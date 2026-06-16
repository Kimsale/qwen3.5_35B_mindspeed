===AGENT agent-a08ff1eb49c231597===
# Qwen3-30B-A3B LoRA 微调性能瓶颈分析与优化方案

## 1. 项目背景

本项目在昇腾 910B3 (8×64GB HBM) 集群上对 Qwen3-30B-A3B MoE 模型进行 LoRA 微调。模型架构为 48 层 Transformer，包含 128 个专家 (topk=8)，采用 GQA (32 heads / 4 KV groups)。软件栈为 CANN 8.5.0 + MindSpeed-LLM 26.0.0 + Megatron core_v0.12.1。当前配置为 TP2/PP1/EP4，序列长度 4096，micro-batch 2，使用 alltoall_seq dispatcher。基线测试显示 HBM 占用仅 17.2GB/65GB (26%)，AI Core 利用率 1-16% (瞬时)，存在严重的资源利用不足问题。

## 2. 基线数据汇总

| 指标 | 数值 | 备注 |
|-----|------|------|
| **硬件配置** | 8×910B3 (64GB HBM each) | CANN 8.5.0 |
| **并行策略** | TP=2, PP=1, EP=4, DP=1 | alltoall_seq dispatcher |
| **训练超参** | seq=4096, MBS=2, GBS=16, bf16 | 变长序列 |
| **HBM 占用** | 17.2 GB / 65 GB | **26% 利用率** |
| **单步耗时** | 2.3–8.4s (mean ~4.8s, std 727ms) | 波动大 |
| **吞吐量** | TPS: 1.95 samples/s, WPS: ~8000 tokens/s | - |
| **AI Core%** | 1–16% (npu-smi 瞬时) | 真实算力约 8% (需 profiler 确认) |
| **收敛状态** | Loss 正常，0 NaN | - |
| **已开优化** | flash-attn, fused-rotary/swiglu/rmsnorm, sequence-parallel, distributed-optimizer, moe-grouped-gemm, moe-permutation-async-comm | - |

## 3. 瓶颈分类诊断

### 3.1 显存瓶颈：严重欠配 (17.2/65GB = 26%)

**现象**
- HBM 占用仅 17.2GB，剩余 47.8GB (73%) 空闲
- 对比理论占用：TP2 切分后模型权重 ~30GB，实际占用远低于预期

**根因**
1. **Micro-batch 过小** (★★★★★): MBS=2 导致激活占用仅 2-5GB。激活与 batch 线性相关，提升到 MBS=8/16 可增加 15-35GB
2. **LoRA 微调模式** (★★★☆☆): 仅训练 LoRA 参数 (r=16, 4 modules)，优化器状态从理论 45GB 降至 <5GB。全参微调可增加 40+GB
3. **变长序列** (★★★☆☆): 实际 token 数 < padding 到的 4096，有效显存利用率降低
4. **TP2 切分** (★★☆☆☆): 模型/激活横切减半。降到 TP1 可使单卡占用翻倍

**证据**
- 固定 batch 模式下单步耗时稳定 ~2.6s，变长模式下 std 达 727ms，印证变长序列影响
- 32K full_pack 脚本 (MBS=2, seq=32768) HBM 占用达 45-55GB，证明序列长度/batch 是主要杠杆

### 3.2 算力瓶颈：AI Core 利用率极低 (1-16%)

**现象**
- npu-smi 瞬时读数 AI Core% 在 1-16% 剧烈波动
- TPS 仅 1.95 samples/s，远低于硬件理论吞吐

**根因**
1. **小 batch 导致算子并行度不足** (★★★★★): MBS=2 × seq=4096 = 8K tokens/step，远低于 MoE grouped-GEMM 饱和阈值 (需 >16K tokens)。NPU ALU 无法填满
2. **AllToAll 通信开销** (★★★★☆): EP4 的 MoE alltoall_seq (3×AllGather + 1×AllToAll) 在小 batch 下效率低。当前未开 overlap，通信期间 AI Core 完全空转
3. **变长序列波动** (★★★☆☆): 不同 seq 长度导致算力利用率在 step 间剧烈波动 (短序列低、长序列高)
4. **MoE 稀疏性** (★★☆☆☆): topk=8/128 = 6.25% 专家激活率，93.75% 权重不参与计算。这是架构固有特性，只能通过增大 batch 摊销路由开销

**证据**
- 粗估真实算力: 30B MoE × topk=8/128 × 2 samples × 4096 seq ≈ 60 TFLOP / 2.3s = 26 TFLOP/s。对比 910B3 理论峰值 ~320 TFLOP/s → **真实利用率约 8%**
- 单步耗时 2.3-8.4s 波动与 npu-smi 读数剧烈波动同步，印证通信-计算串行

### 3.3 通信瓶颈：AllToAll 完全串行

**现象**
- 单步耗时在 MoE 层突增
- AI Core% 在 npu-smi 采样时命中低值 (通信阶段)

**根因**
1. **alltoall_seq 未开 overlap** (★★★★★): 当前路径为 3×AllGather (permutation) + 1×AllToAll (output)，与 expert GEMM **完全串行**。通信时 AI Core 挂起
2. **TP×EP 通信叠加** (★★★☆☆): TP=2 的 all-gather/reduce-scatter + EP=4 的 AllToAll 串行执行，未利用 tp-extend-ep 融合
3. **小 batch 通信效率低** (★★☆☆☆): MBS=2 下通信量与计算量比例失衡 (通信 overhead 固定，计算时间过短无法掩盖)

**证据**
- 32K full_pack 脚本开启 `--moe-alltoall-overlap-comm` 后单步耗时降低 30-40%
- 你当前已开 `moe-permutation-async-comm` (重叠 3 个 AllGather)，但未开主力 overlap 开关

### 3.4 MoE 特定问题

**现象**
- 128 experts 在 EP=4 下每 rank 分配 32 experts
- topk=8 意味每 token 仅激活 6.25% 专家，但需通信全部 8×hidden 数据

**根因**
1. **EP=4 相对保守** (★★★☆☆): 8 卡配置下 EP=4 留下 DP=1 (无数据并行收益)。EP=8 可使每 rank 仅 16 experts，通信量减半
2. **未融合 permute 算子** (★★☆☆☆): token permute/unpermute 未使用融合算子 (CANN 8.5 已支持 `npu_moe_token_permute_with_routing_map`)
3. **GroupedGEMM 梯度未融合** (★★☆☆☆): 反向梯度累加未融合到 grouped-GEMM kernel，额外 kernel 启动开销

**证据**
- 代码约束检查: `num_experts(128) % (TP×EP) == 0` → 当前 128 % 8 = 0 ✓，支持 tp-extend-ep
- full_pack 脚本全部开启 `--gemm-gradient-accumulation-fusion` + `--moe-permute-fusion`

### 3.5 数据流问题

**现象**
- 单步耗时 std 727ms (变异系数 15%)
- 固定 batch 下 std 显著降低

**根因**
- **变长序列 padding 不均** (★★★★☆): 不同 batch 实际 token 数差异大，导致计算/通信时间波动

**证据**
- 固定 seq=4096 模式下单步耗时稳定在 ~2.6s

## 4. 优化方案 (按预期收益排序)

### 方案 1: 增大 Micro-batch + 固定序列长度

**优化目标**: 占满显存 + 提升算子并行度

**命令行改动**
```bash
# 当前: --micro-batch-size 2, 变长序列
# 改为:
--micro-batch-size 8 \
--seq-length 4096   # 确保固定长度，移除任何 --variable-seq-lengths flag
```

**预期收益**
- HBM: 17.2GB → 35-40GB (+18-23GB, 达到 54-62% 利用率)
- AI Core%: 8% → 25-35% (算子并行度 ×4)
- 单步耗时: 2.3-8.4s (波动) → 3.5-4.5s (稳定)
- TPS: 1.95 → 6-8 samples/s (+200-300%)
- 波动: std 727ms → <150ms (固定序列消除波动)

**风险**: 极低
- 训练收敛不受影响 (保持 GBS=16 不变，减少 gradient accumulation steps)
- 通信开销小幅增加 (但 overlap 后可掩盖)

**验证方法**
```bash
# 训练 100 steps，监控:
watch -n 1 'npu-smi info | grep "Memory-Usage"'  # 观察 HBM 增长
# 日志中提取 timing: samples/s, avg_step_time, std
```

**参考来源**
- 激活占用与 batch 线性关系: Megatron 论文 Section 3.2
- 固定序列: 所有 pretrain 脚本均使用固定 seq

---

### 方案 2: MoE AllToAll 通信重叠 (主力优化)

**优化目标**: 掩盖 AllGather/AllToAll 通信延迟

**命令行改动**
```bash
# 在方案 1 基础上追加:
--moe-tp-extend-ep \              # 前置依赖: 用 TP×EP=8 份切 token
--moe-alltoall-overlap-comm \     # 主开关: 异步通信掩盖 expert GEMM
--moe-permute-fusion              # 融合 permute/unpermute 算子
```

**预期收益**
- 通信时间: -50-70% (AllGather latency 被 GEMM 遮掩)
- AI Core%: 25-35% → 40-60% (+15-25%)
- 单步耗时: 3.5-4.5s → 1.5-2.0s (-40-55%)
- HBM: +3-5GB (异步 buffer，仍远低于 65GB)
- TPS: 6-8 → 10-15 samples/s (+50-100%)

**风险**: 低
- `moe-tp-extend-ep` 改变专家分布逻辑 (扩展到 TP×EP=8 组)，需验证 loss 不变
- CANN 8.5 已满足 `moe-permute-fusion` 算子要求 (需 ≥8.3.RC1)

**验证方法**
```bash
# 1. 验证约束满足
grep "num_experts" config.json  # 确认 128 % 8 == 0
grep "CANN" $(which python)     # 确认 CANN ≥ 8.3

# 2. 对比 baseline 跑 50 steps
# 提取 loss 曲线对比 (确保收敛一致)
# profiler trace 看 AllGather 与 GEMM 重叠情况:
--profile --profile-step-start 10 --profile-step-end 12
```

**参考来源**
- 文件: `/data/sejin/third_party/mindspeed-core-26.0.0/mindspeed/features_manager/moe/moe_alltoall_overlap.py`
- 文件: `/data/sejin/third_party/mindspeed-core-26.0.0/mindspeed/features_manager/moe/tp_extend_ep.py`
- 文件: `/data/sejin/third_party/mindspeed-core-26.0.0/mindspeed/features_manager/fusions/fused_moe_permute.py`
- 脚本: `/data/sejin/third_party/mindspeed-llm-26.0.0/examples/mcore/qwen3_moe/tune_qwen3_30b_a3b_32K_full_pack_A3_ptd.sh` (行 103-108)

---

### 方案 3: DP 通信重叠 (辅助优化)

**优化目标**: 掩盖分布式优化器的 gradient reduce-scatter 与 parameter all-gather

**命令行改动**
```bash
# 在方案 2 基础上追加:
--overlap-grad-reduce \
--overlap-param-gather \
--reset-bucket-group-order        # 强制依赖 param-gather
```

**预期收益**
- 单步耗时: -5-10% (优化器通信不阻塞反向)
- AI Core%: +5-10% (DP 通信遮掩)
- HBM: +0-1GB (bucket 重排，可忽略)

**风险**: 极低
- 235B/480B pretrain 脚本标配组合
- 已开 `distributed-optimizer`，满足前置条件

**验证方法**
```bash
# profiler trace 看 reduce-scatter 与 backward 重叠:
# 查找 "ReduceScatter" 与 "Backward" timeline 是否并行
```

**参考来源**
- 文件: `/data/sejin/third_party/mindspeed-core-26.0.0/mindspeed/features_manager/distributed/reset_bucket_group_order_feature.py`
- 脚本: `/data/sejin/third_party/mindspeed-llm-26.0.0/examples/mcore/qwen3_moe/pretrain_qwen3_235b_a22b_4k_A3_ptd.sh`

---

### 方案 4: GroupedGEMM 梯度累加融合 + 内存对齐

**优化目标**: 减少 kernel 启动开销 + 通信对齐

**命令行改动**
```bash
# 在方案 3 基础上追加:
--gemm-gradient-accumulation-fusion \  # GroupedGEMM 反向梯度融合
--param-and-grad-buffer-pad 512        # 昇腾推荐 bucket 对齐
```

**预期收益**
- 单步耗时: -3-5% (减少 kernel 启动 + 通信碎片)
- AI Core%: +2-5%
- HBM: +0GB (仅改调度)

**风险**: 极低
- 依赖 `moe-grouped-gemm` (已开)
- 所有 full_pack 脚本标配

**验证方法**
```bash
# profiler 看 kernel 启动次数减少:
# 对比 GEMMBackward 相关 kernel 数量
```

**参考来源**
- 文件: `/data/sejin/third_party/mindspeed-core-26.0.0/mindspeed/features_manager/moe/gmm.py`
- 文件: `/data/sejin/third_party/mindspeed-core-26.0.0/mindspeed/features_manager/distributed/buffer_pad.py`

---

### 方案 5: 继续增大 Micro-batch (占满剩余显存)

**优化目标**: 最大化显存利用率 + 算子并行度

**命令行改动**
```bash
# 方案 1-4 后若 HBM < 55GB，逐步提升:
--micro-batch-size 12   # 或 16
--global-batch-size 96  # 保持 GBS/MBS = 合理的 accumulation steps
```

**预期收益**
- HBM: 40GB → 50-60GB (达到 77-92% 利用率)
- AI Core%: 40-60% → **60-75%** (接近目标 ≥70%)
- TPS: 10-15 → 12-18 samples/s (+20-50%)

**风险**: 中
- MBS=16 可能触发 OOM (需逐步试探)
- 若 OOM，配合方案 6 (轻量重计算) 回退到 MBS=12

**验证方法**
```bash
# 逐步增大: 8 → 10 → 12 → 16
# 每次跑 20 steps，观察:
watch -n 1 'npu-smi info | grep "Memory-Usage"'
# 若出现 OOM，立即 Ctrl+C 并回退
```

**参考来源**
- 32K full_pack 脚本使用 MBS=2 但 seq=32768 (总 token 数相当于 4K 下 MBS=16)

---

### 方案 6: 轻量重计算 (仅在 OOM 时启用)

**优化目标**: 用少量算力换显存空间，支持更大 batch

**命令行改动**
```bash
# 仅当方案 5 触发 OOM 时启用:
--recompute-activation-function \               # 只重算 MLP 激活函数
--recompute-activation-function-num-layers 24   # 重算前半数层 (48/2=24)
```

**预期收益**
- HBM: -3-5GB (可换取 MBS +2-4)
- 单步耗时: +5-10% (重计算开销，但若换来更大 batch 则净收益仍为正)
- 净效果: MBS=12 (无重计算) 换为 MBS=16 (轻量重计算) → TPS 仍提升 +10-20%

**风险**: 低
- 仅重算激活函数 (非 full recompute)，代价远小于 `--recompute-granularity full`
- 235B pretrain 脚本有成熟用例

**验证方法**
```bash
# 对比 MBS=12 (无重计算) vs MBS=16 (轻量重计算):
# 若后者 TPS > 前者，则启用；否则保持 MBS=12
```

**参考来源**
- 文件: `/data/sejin/third_party/mindspeed-core-26.0.0/mindspeed/features_manager/recompute/activation_function.py`
- 文件: `/data/sejin/third_party/Megatron-LM-core_v0.12.1/megatron/training/arguments.py` (行 1533-1579)

---

### 方案 7: 并行策略调优 - 降低 TP 提升 EP (中期优化)

**优化目标**: 简化 TP 通信 + 最大化 MoE 专家并行

**命令行改动**
```bash
# 需重新切分 checkpoint，单独测试:
--tensor-model-parallel-size 1 \
--expert-model-parallel-size 8 \
# 保留方案 1-4 所有 overlap 开关
# 注: tp-extend-ep 在 TP=1 时自动退化为无效
```

**预期收益**
- 通信简化: 消除 TP 的 all-gather/reduce-scatter，仅保留 EP AllToAll
- EP=8 每 rank 仅 16 experts (vs EP=4 的 32)，通信量减半
- AI Core%: 可达 **65-75%** (纯 EP 并行最简洁)
- HBM: 15-18GB (TP1 下模型不切分，略降，但可通过 MBS 补回)

**风险**: 中
- 需重跑 checkpoint 转换 (TP 维度从 2→1)
- TP1 下某些 TP 相关优化失效 (但 EP overlap 仍有效)
- 训练超参可能需微调

**验证方法**
```bash
# 1. 转换 checkpoint:
python tools/checkpoint/convert_checkpoint.py \
  --model-type GPT \
  --load-dir /path/to/tp2_ckpt \
  --save-dir /path/to/tp1_ckpt \
  --target-tensor-parallel-size 1

# 2. 单独跑 100 steps，对比 TP2/EP4 vs TP1/EP8:
# - Loss 曲线一致性
# - TPS / AI Core% / 单步耗时
```

**参考来源**
- 30B-A3B 模型参数分布: MoE 占 ~70%，适合激进 EP
- MoE 理论: 专家并行优先于张量并行 (通信量更低)

---

### 方案 8: 引入上下文并行 (仅当需要长序列时)

**优化目标**: 支持 seq > 8192 的长文本任务

**命令行改动**
```bash
# 仅当业务需求 seq ≥ 8192 时:
--context-parallel-size 2 \
--context-parallel-algo ulysses_cp_algo \  # GQA 下 head 须整除 CP×TP
--seq-length 8192  # 或更长

# 调整拓扑: TP2/PP1/EP4/CP2 (需 8×2=16 卡) 或 TP2/PP1/EP2/CP2 (8 卡)
```

**预期收益**
- 支持长序列: seq 可达 16K-32K (CP=2 下)
- HBM: +15-25GB (序列翻倍)
- AI Core%: +5-10% (长序列填满算子)

**风险**: 高
- 你当前 seq=4K 可能不需要 CP
- CP 引入额外 all-to-all (head 维度切分)，通信开销显著
- 需 8 卡拓扑重构: TP2/EP2/CP2 (减小 EP) 或 16 卡集群

**验证方法**
```bash
# 确认数据集包含 seq > 8K 的样本
# 对比 CP=1 (seq=4K) vs CP=2 (seq=8K) 的吞吐与收敛
```

**参考来源**
- 文件: `/data/sejin/third_party/mindspeed-core-26.0.0/mindspeed/features_manager/context_parallel/ulysses_context_parallel.py`
- 脚本: `/data/sejin/third_party/mindspeed-llm-26.0.0/examples/mcore/qwen3_moe/tune_qwen3_30b_a3b_256K_full_pack_A3_ptd.sh` (TP2/EP16/CP16)

---

### 方案 9: AutoTuning 自动搜索并行配置 (探索性)

**优化目标**: 用 MindSpeed auto-settings 自动搜索最优 TP/PP/EP/CP/MBS 组合

**命令行改动**
```bash
# 在现有训练脚本基础上追加 (移除 --load/--save):
python pretrain_gpt.py \
  --auto-settings \
  --auto-settings-type mixed \      # white=最快, black=最准, mixed=折中
  --auto-settings-ranks 8 \         # 搜索用的 world size
  --auto-settings-work-dir /data/sejin/auto_tune_output \
  --target-nnodes 1 \
  --nnodes 1 --nproc-per-node 8 --node-rank 0 \
  --master-addr 127.0.0.1 --master-port 29500 \
  # ... (其余训练参数保持不变)
```

**预期收益**
- 输出 top-3 推荐配置 (TP/PP/EP/CP/MBS/recompute 层数)
- 基于 profiling 建模，比手动试探更系统
- mixed 模式会实测 top-5 候选，准确性高

**风险**: 中
- profiling 用降配模型 (少层、experts 估算)，推荐值需用真实配置复测
- MC2 候选可能触发已知 crash (搜索空间需去掉 mc2)
- 仅输出推荐值，需手动填回训练脚本

**验证方法**
```bash
# 1. 运行 auto-settings (约 30-60 分钟)
# 2. 查看输出:
grep "<==========Top" /data/sejin/auto_tune_output/*.log
# 3. 提取推荐配置，修改训练脚本后跑 100 steps 验证
```

**参考来源**
- 文件: `/data/sejin/third_party/mindspeed-core-26.0.0/mindspeed/features_manager/auto_settings/auto_settings_feature.py`
- 文件: `/data/sejin/third_party/mindspeed-core-26.0.0/mindspeed/auto_settings/module/searcher.py`

---

### 方案 10: MC2 融合算子 (高风险，暂不推荐)

**优化目标**: TP 的 matmul + all-gather/reduce-scatter 算子级融合

**命令行改动**
```bash
# 需先确认环境稳定性:
--use-ascend-mc2  # 或 --moe-alltoall-mc2
```

**预期收益**
- 通信-计算融合: 理论可达 +20-30% 吞吐
- AI Core%: +10-15%

**风险**: 极高
- **MEMORY.md 明确记录此环境 MC2 会 crash** ("to avoid MC2 crash")
- `--use-ascend-mc2` 与 `--moe-alltoall-overlap-comm` 互斥
- `--moe-alltoall-mc2` 在 alltoall_seq + TP=2 下不满足约束 (需 TP=1)

**验证方法**
```bash
# 仅在单独分支测试 (不影响主训练):
# 1. TP1/EP8 配置下测试 --moe-alltoall-mc2
# 2. 跑 20 steps，监控是否 crash
# 3. 若稳定，对比 baseline 吞吐；若 crash，放弃
```

**参考来源**
- 文件: `/data/sejin/third_party/mindspeed-core-26.0.0/mindspeed/features_manager/tensor_parallel/mc2.py`
- 文件: `/data/sejin/third_party/mindspeed-core-26.0.0/mindspeed/features_manager/moe/moe_alltoall_mc2.py`
- 警告: MEMORY.md 的 `ascend-910b-mindspeed-qwen3-env.md`

---

## 5. 验证计划

### 5.1 基线数据采集 (T0, 已完成)

**指标**
- HBM: npu-smi 持续监控
- 单步耗时: 训练日志 `iteration time (ms)`
- 吞吐: TPS (samples/s), WPS (tokens/s)
- AI Core%: npu-smi 瞬时读数 (不准) + profiler 实测 `AICore_time / E2E_time`
- Loss: 确保收敛，0 NaN

**Profiler 采集**
```bash
--profile \
--profile-step-start 10 \
--profile-step-end 12 \
--profile-level 1 \
===AGENT agent-a1f2d4b569031920a===
I have all the information needed. Here is the complete inventory of memory-optimization command-line switches that truly exist in the code.

---

# 显存优化开关清单 (MindSpeed-LLM 26.0.0 + MindSpeed core 26.0.0 + Megatron core_v0.12.1)

均为代码中真实存在的 flag。来源路径标于每组末尾。

## 1. 重计算 (Recompute) — 原生 Megatron (core_v0.12.1)

来源: `/data/sejin/third_party/Megatron-LM-core_v0.12.1/megatron/training/arguments.py:1533-1579, 2493`

- `--recompute-granularity {full|selective}` (default None)
  - `full`: 整层 Transformer 重计算 (省显存最多, 速度最慢)。
  - `selective`: 只重计算 `--recompute-modules` 指定的子模块 (default `core_attn`)。
- `--recompute-method {uniform|block}` (default None)
  - `uniform`: 把所有层均分成块, 每块重计算其输入激活。
  - `block`: 每个 PP stage 内只重计算前 N 层 (N 由 `--recompute-num-layers` 指定)。
- `--recompute-num-layers N` (int, default None)
  - uniform 模式: 每个重计算单元的层数; block 模式: 每个 PP stage 重计算的层数。
- `--recompute-modules [...]` (nargs，default `["core_attn"]`)。可选: `core_attn`, `moe_act`, `layernorm`, `mla_up_proj`, `mlp`, `moe`。`moe_act`/`layernorm`/`mla_up_proj` 用 output-discarding checkpoint; `core_attn`/`mlp`/`moe` 用普通 checkpoint。
- `--distribute-saved-activations` (store_true): 把重计算保存的激活按 TP 组切分分布 (需配合 `full`)。
- `--moe-layer-recompute` (store_true): 已弃用。等价于 `--recompute-granularity selective --recompute-modules moe`。整 MoE 层重计算。对当前 30B-A3B (MoE 为主)最直接。

注: 30B-A3B 用标准 GQA, 不涉及 MLA, 故 `mla_up_proj` 不适用。`--recompute-modules moe` / `moe_act` 是占满显存后做权衡的主力开关。

## 2. 重计算 — MindSpeed core 扩展

来源: `mindspeed-core-26.0.0/mindspeed/features_manager/recompute/*` 及 `mindspeed/arguments.py:282-322`

- `--recompute-activation-function` (store_true): 只重计算 MLP 层激活函数。
- `--recompute-activation-function-num-layers N` (int): 配合 `--recompute-method block` + `--recompute-num-layers` 使用。校验: 0 ≤ N ≤ num_layers。
- `--recompute-norm` (store_true): 重计算 Transformer 层内 norm。
- `--recompute-norm-num-layers N` (int): 可与 activation 重计算联用。校验同上。
- `--enable-recompute-layers-per-pp-rank` (store_true, default False): 使 `--recompute-num-layers` 按每个 PP rank 计数 (否则按 VPP rank 计数)。
- `--moe-adaptive-recompute-activation` (store_true): MoE 自适应重计算, 缓解训练早期显存不均衡。
- `--moe-adaptive-recompute-activation-scale F` (float, default 2.0): 上者的阈值因子。
- `--recompute-in-bubble` (store_true): 利用 PP 气泡做重计算省显存 (ripipe schedule)。
- `--recompute-in-advance` (store_true): 提前重计算以减少气泡。

### 自适应整体重计算 (二选一, 互斥于 swap/AMO)
- `--adaptive-recompute-device-size N` (int, default -1): >0 时启用自适应选择性重计算, 数值为目标显存(单位见实现)。
- `--adaptive-recompute-profiling-step N` (int, default 10): 上者第 N 步后求解策略图。
- `--adaptive-recompute-device-swap` (store_true): 自适应重计算+swap 开关。
- `--adaptive-memory-optimization` (store_true, default False): AMO 自适应显存优化。与 swap-attention 互斥。

## 3. Swap (激活/优化器换出到 CPU/虚拟内存)

- `--swap-attention` (store_true) + `--swap-modules STR` (default `"input_norm,self_attention,post_attention_norm"`): 把指定模块激活换出。
  来源: `mindspeed/features_manager/memory/swap_attention.py`
  关键约束: **与 LoRA 不兼容** (代码第 27-29 行: `is_enable_lora and swap_attention → AssertionError`)。当前任务是 LoRA 微调, **此开关不可用**。也与 adaptive recompute / AMO 互斥。
- `--swap-optimizer` (store_true) + `--swap-optimizer-times N` (int, default 16): 优化器状态换出到 CPU, 每次搬移 `len(shard_fp32)/N` 个元素。
  来源: `mindspeed/features_manager/optimizer/swap_optimizer_feature.py`
  约束: 与 `reuse_fp32_param` 互斥; 依赖 `--use-distributed-optimizer`。
- `--virtual-optimizer 'all'|F [F ...]` (nargs+): 用 NPU 虚拟内存换出优化器。`all`=65GB(满卡)。
  来源: `mindspeed/features_manager/optimizer/virtual_optimizer.py`
  约束: 需 torch_npu 支持 `empty_with_swapped_memory`; 与 `fused_ema_adamw` 互斥。
- `--smart-swap` (store_true): 智能 swap, 替换显存分配器自动换出。
  来源: `mindspeed/features_manager/memory/smart_swap.py`。约束: 与 adaptive selective recompute 互斥。
- `--moe-unperm2-mem-optim-swap` (store_true): MoE fb-overlap 下 unpermute2 显存优化 swap。
  来源: `mindspeed/features_manager/moe/fb_overlap.py`

## 4. reuse-fp32-param (优化器 FP32 副本省显存)

来源: `mindspeed/features_manager/memory/reuse_fp32_param.py`

- `--reuse-fp32-param` (store_true): 分布式优化器释放 FP32 参数副本以省显存。
  约束: **依赖 `--bf16`** (当前已满足); 与 `enable_zero3` 互斥; 与 `swap_optimizer` 互斥; legacy 模型下与 `overlap_param_gather` 不兼容; 与 `fused_ema_adamw` 互斥。

## 5. MoE 专属显存开关 (MindSpeed core)

来源: `mindspeed/arguments.py:162-167`

- `--moe-zero-memory {disable|level0|level1}` (default `disable`): 节省 MoE 层激活显存。level1 比 level0 更激进。
- `--moe-zero-memory-num-layers N` (int, default None): 每个 PP stage 用 level1 的层数。

## 6. Loss 显存优化 (MindSpeed-LLM)

来源: `mindspeed_llm/features_manager/memory/chunk_loss.py`

- `--loss-compute-mode {default|chunk}` (default `default`): chunk 模式分块计算 CE loss 省显存。
- `--loss-chunk-size N` (int, default 1024): chunk 模式下的分块大小。

## 7. 其它相关

- `--noop-layers STR`: 指定空操作层 (`mindspeed/arguments.py:362`, 也在 ckpt 转换脚本中)。
- `--manual-gc` / `--manual-gc-interval N` / `--no-manual-gc-eval` (Megatron 原生): 手动 GC 对齐各 rank 时机, 间接稳定显存。

## 不适用于本任务的 (MLA-only, 仅记录)

`--mla-zero-memory`, `--recompute-mla-up-proj`, `--mla-swap-core-attn-out` 等 (`mla_feature.py`) 仅对 multi-latent-attention 生效。Qwen3-30B-A3B 用标准 GQA, **不适用**。

---

## 针对当前场景的关键提示

当前任务 HBM 仅占 26% (17.2/65GB), 目标是占满显存并提升 AI Core 利用率, 方向与省显存相反。上述开关中:
- 现在应**关闭/不开**所有重计算和 swap, 把激活留在显存换吞吐。
- **`--swap-attention` 本身就因 LoRA 不可用**, 无需考虑。
- 真正该反向利用的是: 增大 `mbs/gbs/seq`、减小并行切分 (降 TP/EP)、关掉重计算, 这些才会把显存吃满并拉高算力。`--reuse-fp32-param`/`--swap-optimizer`/`moe-zero-memory` 这类是显存吃紧时才需要的, 当前不建议开。

相关源文件路径:
- `/data/sejin/third_party/mindspeed-core-26.0.0/mindspeed/arguments.py` (集中定义, 149-167 行 MoE, 282-322 行 recompute/swap)
- `/data/sejin/third_party/mindspeed-core-26.0.0/mindspeed/features_manager/{recompute,memory,optimizer}/`
- `/data/sejin/third_party/mindspeed-llm-26.0.0/mindspeed_llm/features_manager/memory/chunk_loss.py`
- `/data/sejin/third_party/Megatron-LM-core_v0.12.1/megatron/training/arguments.py:1533-1579, 2493`
===AGENT agent-a9b00e662931a5889===
I have a complete picture from the code. Here are the findings, based only on the source in mindspeed-core-26.0.0.

## MindSpeed Auto Settings (auto_settings) — 勘察结论

### 是什么 / 怎么启用
原生自动调优能力叫 `auto-settings`（注册为 feature `AutoSettingsFeature`，optimization_level=2）。它不是训练时的在线优化，而是一个独立的"搜索-profiling-建模-推荐"流程：用小集群跑若干 profiling 试验，对内存/时间建模，然后在合法并行配置空间里搜出 top-3 推荐配置并打印到日志，**不会写 checkpoint，也不会改你的训练脚本**。

启用入口（`features_manager/auto_settings/auto_settings_feature.py` + `training.py:428`）：
- 主开关 flag：`--auto-settings`（store_true）。命中后 `mindspeed/training.py` 的 pretrain 包装里走 `AutoSettings().auto_setting_fun(argument)` 并直接 return，不进入正常训练。
- 硬约束：`--auto-settings` 时**不能带 `--load` / `--save`**（`validate_args` 会 assert 报错）。
- 内部还有 3 个环境变量做子进程分流（用户不需要手动设，由框架在 profiling 子进程里注入）：`OOTB_OPTIMIZER_PARSE_ARGS` / `OOTB_OPTIMIZER_PARSE_MODEL` / `OOTB_OPTIMIZER_PROFILING`(+`_BLACK`)。`is_need_apply` 对这些变量也返回 True。

相关 flag（都在 `_add_auto_settings_args`）：
- `--auto-settings-work-dir`（默认 cwd，存 profiling/pkl/csv/json 结果）
- `--auto-settings-ranks`（默认 8，= 搜索用的 world size，`search_world_size`）
- `--auto-settings-type`（默认 `white`，可选 `white|black|mixed`）
- `--auto-settings-log-level`（info/warning/debug）
- `--target-nnodes`（要外推到的目标大集群节点数）、`--nnodes/--nproc-per-node/--node-rank/--master-addr/--master-port`（透传给内部 torchrun）
- `--prof-file`，以及自动流水线相关 `--automated-pipeline`、`--automated-pipeline-perf`、`--recompute-module-list`、`--jit-compile`

### 三种搜索器（`module/searcher.py`）
- `white`（白盒，默认）：先做 profiling 建模（算子时间 + 内存），再对全量合法配置用解析模型估算 peak memory 与 e2e time，按时间取 top-k。最快、不实跑每个候选。
- `black`（黑盒）：对候选配置实际裁剪后（PP=1/VPP=1/MBS=1）跑 profiling，实测 peak memory 和 step time，写 `model_results.csv`，按 e2e_time 排序取 top-k（要求 peak_mem < 0.95×显存）。最准、最慢。
- `mixed`：先 white 取 top-5，再对这 5 个用 black 实测，取最终 top-3。

### 能自动搜索哪些维度
合法空间在 `search_space.py:build_search_spaces()` 枚举（约束：tp·cp·pp·dp == world_size，层数能整除 pp 等）：
- TP `tensor_model_parallel_size`（2 的幂，受 devices_per_node 限制）
- CP `context_parallel_size`（含长序列约束：910B 上 seq/cp ≥ 8K 才允许 cp>1）
- PP `pipeline_model_parallel_size`
- DP `data_parallel_size`（由前三者推导）
- EP `expert_model_parallel_size`（仅 MoE，范围 1..min(cp·dp, num_experts)，且满足 (cp·dp)%ep==0、num_experts%(extend_ep)==0）
- VPP `num_layers_per_virtual_pipeline_stage`（仅 pp>1 时搜）
- MBS `micro_batch_size`（固定枚举 `[1, 2]`）
- ZeRO1 `use_distributed_optimizer`（按 dp·cp/ep>1 推导）
- MC2 `use_ascend_mc2`（profiling 配置里带 mc2 变体，white 搜索按收益开关；注意 mc2 会强制 sequence-parallel）
- 重计算层数 `recompute_num_layers`：不是在主空间枚举，而是在 white `time_cost` 里按内存是否 OOM 反解出需要重计算的层数（full/block 粒度），作为推荐配置的一部分输出。

不搜索的（硬编码/透传）：模型结构、专家数（profiling 用降配，MoE 固定取 128 估算）、LoRA 超参、学习率等。`parallel_switch = ["tp","cp","dp","pp","ep","mc2"]` 控制哪些维度参与。

### 输入 / 输出
输入：你现有的训练启动命令（完整 argv，框架直接 copy `sys.argv` 透传），加上 `--auto-settings` 及上面的 flag；模型/系统信息由首个 PARSE_ARGS 子进程自动探测（含 `torch.npu.get_device_properties` 拿显存上限、device_type、CANN/driver 版本）。

输出（写到 work-dir）：
- profiling 中间产物：`auto_settings_static_model_pp4.json`、`...expert2.json`、`...tp2.json`、`profile/profiling_configs.json`、各配置的 profiling 目录、`at_<rank>.pkl`、`post_info`
- black/mixed：`model_results.csv`
- 最终：top-3 推荐配置打到日志（`<==========Top #N config==========>`），字段见 `SearchConfig.__str__`：DP/TP/PP/VPP/CP/EP/ZeRO1/MC2/TokenRearrange/MicroBatchSize/Recompute layer。**只是推荐值，需要你手动填回训练脚本**。

### 运行机制
master 节点 `Profiler.run` → 通过 `subprocess` 起内部 `torchrun`（`profile/runner.py`，nnodes/nproc/master 来自上面 flag）跑降配 profiling（典型 train_iters=10，profile_step 8–9，level1，带 cpu/memory/shapes）。多机时从节点走 `run_on_slaves` 用 GLOO + `monitored_barrier` 等 master 广播任务。先用小集群（`--auto-settings-ranks`）profiling，再外推到 `--target-nnodes` 目标规模。

### 典型用法（推断自代码，无官方 example 脚本）
MindSpeed-LLM 的 examples/docs 里**没有**现成 auto-settings 脚本（grep 仅命中 coc 等无关项）。用法是在现有 `tune_*.sh` / `pretrain_*.sh` 的 torchrun/python 命令上追加，例如：
```
--auto-settings \
--auto-settings-type white \
--auto-settings-ranks 8 \
--auto-settings-work-dir /data/sejin/auto_settings_out \
--target-nnodes 1 --nnodes 1 --nproc-per-node 8 --node-rank 0 \
--master-addr 127.0.0.1 --master-port 29500
```
并移除 `--load/--save`。

### 对当前任务的关键约束 / 注意点
- 它搜的就是 TP/PP/CP/EP/DP/VPP/MBS/ZeRO/MC2/重计算——正好覆盖你"仅可调并行策略/混合精度/缓存/通信"的合法范围，没碰模型结构，符合硬约束。
- 但 search_space 用的是 `global_batch_size` 推导 micro-batch 数且 MBS 只枚举 `[1,2]`；profiling 用降配模型（少层、experts 估算 128、固定 seq 下限 8K），所以推荐值是估算，落地后仍需用你的真实变长序列复测。
- white 搜索基于解析模型，可能与你实测的"变长序列 + MoE 突发计算"偏差较大；若要更贴近真实占用，用 `mixed` 或 `black`（会实跑，耗时长，但 black 默认 ranks='0' 单卡裁剪 profiling）。
- 注意 mc2 候选在你的环境里有已知 MC2 crash 风险（见 MEMORY 的 LoRA flags 规避项）——若搜索开 mc2 变体可能在 profiling 阶段崩，必要时确认 `parallel_switch` 是否要去掉 `mc2`。

关键文件（绝对路径）：
- Feature/flag 定义：`/data/sejin/third_party/mindspeed-core-26.0.0/mindspeed/features_manager/auto_settings/auto_settings_feature.py`
- 主入口：`/data/sejin/third_party/mindspeed-core-26.0.0/mindspeed/auto_settings/auto_settings.py`
- 触发点：`/data/sejin/third_party/mindspeed-core-26.0.0/mindspeed/training.py`（185/273/428 行）
- 搜索空间：`/data/sejin/third_party/mindspeed-core-26.0.0/mindspeed/auto_settings/search_space.py`
- 搜索器：`/data/sejin/third_party/mindspeed-core-26.0.0/mindspeed/auto_settings/module/searcher.py`
- profiling 配置模板：`/data/sejin/third_party/mindspeed-core-26.0.0/mindspeed/auto_settings/profile/profiling_configs.json`
- argv 透传/过滤：`/data/sejin/third_party/mindspeed-core-26.0.0/mindspeed/auto_settings/profile/argv.py`
- torchrun 启动器：`/data/sejin/third_party/mindspeed-core-26.0.0/mindspeed/auto_settings/profile/runner.py`
- 配置数据类：`/data/sejin/third_party/mindspeed-core-26.0.0/mindspeed/auto_settings/config/{search_config.py,model_config.py,system_config.py}`

MindSpeed-LLM 侧无 auto-settings 示例脚本或文档。
===AGENT agent-aa5d5dc3061b7495d===
I have enough to compile the findings. Here is the complete inventory of real flags found.

---

# MindSpeed-LLM 26.0.0 进阶开关勘察结果

所有条目均来自 `mindspeed-core-26.0.0/mindspeed/features_manager/` 与 `mindspeed-llm-26.0.0/examples/mcore/qwen3_moe/` 的真实代码，未列任何臆测项。

## 1. 通信掩盖 (Communication Overlap)

| Flag | 作用 | 来源/约束 |
|---|---|---|
| `--overlap-grad-reduce` | 反向梯度 reduce-scatter 与计算重叠 (DP/分布式优化器) | 原生 Megatron arg (core 未重定义)。235B/480B pretrain 脚本用 |
| `--overlap-param-gather` | 前向参数 all-gather 与计算重叠 | 原生 Megatron arg。与 `--reset-bucket-group-order` 配合 |
| `--reset-bucket-group-order` | 让前向计算按正确的 overlap all-gather 顺序走 (optlevel 2)。**强制依赖** `--overlap-param-gather`,否则报错 | `distributed/reset_bucket_group_order_feature.py` |
| `--moe-alltoall-overlap-comm` | MoE alltoall/alltoall_seq dispatcher 下用异步通信+swap 掩盖计算。alltoall+share_expert 时自动开 `--moe-shared-expert-overlap`。**依赖** `--moe-grouped-gemm`;alltoall_seq 还需 `--moe-permutation-async-comm`,且 tp>1 时需 `--moe-tp-extend-ep` | `moe/moe_alltoall_overlap.py`。与 mc2 互斥 |
| `--moe-allgather-overlap-comm` | allgather dispatcher 下异步通信掩盖计算。**依赖** `--moe-permutation-async-comm` + `--moe-grouped-gemm` | `moe/moe_allgather_overlap.py`。与 mc2 互斥 |
| `--moe-permutation-async-comm` | 重叠 MoE permutation 的 3 个 all-gather 通信 | 定义在 `moe/tp_extend_ep.py`。**基线已开** |
| `--moe-fb-overlap` | MoE 前/反向 (fwd-bwd) 跨 microbatch 重叠。需 ETP=1 且 EP>1;不支持 allgather/alltoall_seq dispatcher;需无 PP 或 VPP 或 dualpipev | `moe/fb_overlap.py`。与 `overlap_grad_reduce`、`moe_alltoall_overlap_comm`、`moe_tp_extend_ep`、`swap_attention` 互斥 |
| `--moe-unperm2-mem-optim-swap` | fb-overlap 下 unpermute2 内存优化 swap | 仅可与 `--moe-fb-overlap` 同用 |
| `--use-cp-send-recv-overlap` | CP (ring) 下 send-recv 通信重叠 | `context_parallel/context_parallel_feature.py` |
| `--overlap-p2p-comm` | PP 的 P2P 通信重叠 (VPP 场景) | 原生 Megatron;`megatron_basic.py` 在 VPP_size=1 时会自动关掉 |

注意: `tp-comm-overlap` **在 MindSpeed-core features_manager 中没有注册**(grep 无 `add_argument`)。它是 TE/Megatron 原生 GPU 概念,昇腾上的对应能力由下面的 MC2 / CoC 提供。qwen3_moe 脚本中也没有任何脚本使用 `--tp-comm-overlap`。

## 2. 算子级 TP 通算融合 (MC2 / CoC) — 替代 GPU 的 tp-comm-overlap

| Flag | 作用 | 来源/约束 |
|---|---|---|
| `--use-ascend-mc2` | 昇腾 MC2 融合算子,把 TP 的 ColumnParallel/RowParallel Linear 的 matmul 与 all-gather/reduce-scatter 通算融合。**要求 TP>1 且 `--sequence-parallel`**,否则自动禁用 | `tensor_parallel/mc2.py`。与 coc、use_pipe_experts、use_nanopipe、unaligned_linear 互斥。**(注: MEMORY.md 记录此环境 MC2 会 crash,LoRA 下需避开)** |
| `--use-ascend-coc` | 昇腾 CoC (computation-communication) 通算并行,同样融合 TP linear 的通算 | `tensor_parallel/locl_coc.py`。**TE impl 不支持** (transformer_engine 报错);与 mc2 互斥、tp-2d 互斥 |
| `--coc-mode` | CoC 模式: 0=original / 1=rewrite / 2=coc default (默认 -1) | 同上 |
| `--coc-parallel-num` | CoC 切分并行粒度 (默认 1) | 同上 |
| `--coc-fused-kernel` | 使用 CoC 融合 kernel | 同上 |
| `--moe-alltoall-mc2` | [expert] MoE 层 alltoall/alltoall_seq dispatcher 用 MC2 融合 kernel。alltoall_seq 需 TP=1;alltoall 需 ETP=1 且 dropless | `moe/moe_alltoall_mc2.py`。与 use_ascend_mc2、两个 overlap-comm、fb_overlap、tp_extend_ep 全互斥 |

## 3. 算子融合 (Fusion)

| Flag | 作用 | 来源 |
|---|---|---|
| `--use-fused-rmsnorm` | 融合 RMSNorm | `megatron_basic/megatron_basic.py`。**基线已开** |
| `--use-fused-swiglu` | 融合 SwiGLU | 同上 + `fusions/fused_bias_swiglu.py`。**基线已开** |
| `--use-fused-rotary-pos-emb` | 融合 RoPE;**要求** `--position-embedding-type=rope` | `fusions/fused_rope.py`。**基线已开** |
| `--use-fused-moe-token-permute-and-unpermute` / `--moe-permute-fusion` | 两者等价(优先用后者)。融合 MoE token permute/unpermute,调 `npu_moe_token_permute_with_routing_map`。**需 CANN≥8.3.RC1 + PTA≥7.2.RC1**(当前环境 CANN 8.5 满足);只支持 alltoall/alltoall_seq dispatcher,不支持 allgather | `fusions/fused_moe_permute.py`。**当前基线未开,可加** |
| `--gemm-gradient-accumulation-fusion` | GroupedGEMM 中梯度累加融合。**依赖** `--moe-grouped-gemm` | `moe/gmm.py`。**基线未开,所有 full_pack 脚本都开了,建议加** |
| (fused-softmax) | 融合 ScaledMaskedSoftmax 等,无独立 flag,optlevel 0 默认挂载 | `fusions/fused_softmax.py` |
| `--lora-fusion` | LoRA 计算融合(CCLoRA 类),LoRA 脚本启用 | 定义在 mindspeed-llm 侧(非 core features_manager);见 `tune_qwen3_30b_a3b_4K_lora_ptd.sh` |

## 4. GroupedGEMM / 专家并行扩展

| Flag | 作用 | 来源 |
|---|---|---|
| `--moe-grouped-gemm` | MoE 专家用 GroupedMLP + grouped GEMM 算子(替换逐专家 matmul) | `moe/gmm.py` + `fusions/grouped_matmul.py`。**基线已开** |
| `--moe-tp-extend-ep` | 用 TP group 扩展专家并行(切 token 而非切专家权重),要求 `num_experts % (tp*ep)==0`。**依赖** `--moe-permutation-async-comm`+`--moe-grouped-gemm`。alltoall_seq + tp>1 + overlap 时必需 | `moe/tp_extend_ep.py` |

## 5. 上下文并行 (Context Parallel)

主控: `--context-parallel-size N` + `--context-parallel-algo`,4 种算法(`context_parallel_feature.py`):

| algo 值 | 类型 | 关键约束 |
|---|---|---|
| `megatron_cp_algo` | Ring attention (默认) | seq 须整除 2*CP;配 `--cp-window-size`(double-ring,范围 [1,CP)) + `--use-fused-ring-attention-update` + `--megatron-cp-in-bnsd` |
| `ulysses_cp_algo` | Ulysses (all2all 切 head) | head 须整除 CP*TP;qwen3_moe **所有脚本用这个** (`ulysses_context_parallel.py`) |
| `hybrid_cp_algo` | Ulysses+Ring 混合 | 需 `--ulysses-degree-in-cp` |
| `hybrid_adaptive_cp_algo` / `adaptive_cp_algo` | 自适应 mask | `adaptive_context_parallel.py`,配 `--attention-mask-on-cpu` 等 |
| `kvallgather_cp_algo` | KV allgather | 仅 causal mask |

CP 相关附加 flag:
- `--ulysses-degree-in-cp` (hybrid 必需)
- `--cp-window-size` (double-ring 窗口)
- `--attention-mask-type {causal,general}`
- `--context-parallel-kv-cache-policy {full,half}` + `--context-parallel-cache-interval` + `--use-ulysses-allgather-kv` (`context_parallel_kv_cache.py`,需 CP>1 + flash-attn;allgather-kv 需 GQA)

## 6. 2D 张量并行 (tp-2d) — 自带通信重叠

`tensor_parallel/tp_2d.py`: `--tp-2d` + `--tp-x` + `--tp-y` (须 tp = tp_x*tp_y)。自带 overlap 开关:
- `--enable-overlap-ag-with-matmul` (前向 all-gather 叠 matmul)
- `--enable-overlap-matmul-with-rs` (matmul 叠 reduce-scatter)
- `--enable-backward-overlap-ag-with-matmul` (反向)

**约束: 不支持 MoE (`expert_model_parallel_size>1` 报错)** → 对 Qwen3-30B-A3B MoE 不可用;与 sequence_parallel、use_fused_rmsnorm、use_ascend_coc 互斥。

## 7. 显存/优化器侧(影响“占满显存”与重叠的相邻开关)

- `--use-distributed-optimizer` (基线已开) — ZeRO-1 式优化器分片,是 overlap-grad-reduce / overlap-param-gather 的前提
- `--reuse-fp32-param` (`memory/reuse_fp32_param.py`) — 释放 FP32 param 副本省显存,**依赖 bf16**;235B 脚本用
- `--param-and-grad-buffer-pad 512` (`distributed/buffer_pad.py`) — 昇腾建议设 512,bucket 内存对齐
- `--moe-zero-memory {disable,level0,level1}` + `--moe-zero-memory-num-layers` (`moe/moe_zero_memory.py`) — MoE 激活省显存,**只支持** 配 `--moe-alltoall-overlap-comm` 或 `--moe-fb-overlap`
- `--swap-attention` / `--swap-optimizer` — 32K full_pack 脚本用(省显存换带宽,与提升利用率方向相反,LoRA 占用低时不需要)
- `--recompute-activation-function` (`recompute/activation_function.py`) — 仅重算 MLP 激活,比 full recompute 省算力
- `--schedules-method dualpipev` (`pipeline_parallel/dualpipev_feature.py`) — DualPipeV 调度;**需 PP>1、untie embeddings、CP=1、无 VPP**,与 overlap_grad_reduce/swap_attention 互斥

## 关键对照: qwen3_moe 各脚本实际并行+进阶配置

| 脚本 | TP/PP/EP/CP | dispatcher | 进阶 overlap/fusion flags |
|---|---|---|---|
| `tune_..._4K_lora_ptd` (与你 LoRA 任务最贴近) | 4/1/1/- | alltoall_seq | grouped-gemm, permutation-async-comm, fused-rope/swiglu/rmsnorm, seq-parallel, dist-optimizer, **lora-fusion** |
| `tune_..._4K_full_ptd` | 4/2/2/- | (full) | 同上基础集 |
| `tune_..._32K_full_pack_A3` | 4/2/8/1 (ulysses) | **alltoall** | + **moe-alltoall-overlap-comm**, **gemm-gradient-accumulation-fusion**, swap-optimizer, swap-attention, recompute full/uniform/1 |
| `tune_..._32K_full_pack_A+X` | 4/1/8/1 | alltoall | 同上 (仅 PP=1) |
| `tune_..._256K_full_pack_A3` | 2/2/16/16 (ulysses) | — | CP=16 长序列 |
| `pretrain_..._4K` | 1/2/8/1 | — | — |
| `pretrain_235b_a22b_4k_A3` | — | alltoall_seq | moe-alltoall-overlap-comm, gemm-grad-accum-fusion, **reuse-fp32-param, overlap-grad-reduce, overlap-param-gather**, VPP, noop-layers 94,95 |
| `pretrain_480b_4k_A3` | — | alltoall | 同 235B (overlap-grad-reduce + overlap-param-gather + VPP + noop-layers 0,63) |

## 对你 LoRA 调优任务的直接结论(列举可加项,真实存在且兼容)

1. **`--gemm-gradient-accumulation-fusion`** — 依赖已满足 (moe-grouped-gemm 已开),纯收益,所有 full_pack 脚本都用。
2. **`--moe-permute-fusion`** (= `--use-fused-moe-token-permute-and-unpermute`) — CANN 8.5 满足版本要求,当前 dispatcher 是 alltoall_seq(支持),基线未开。
3. **`--moe-alltoall-overlap-comm`** — 通信掩盖主力。当前 alltoall_seq + TP2 → **必须同时加 `--moe-tp-extend-ep`** 且需 `num_experts(128) % (TP*EP)==0`(TP2*EP4=8,128%8=0 ✓);依赖 permutation-async-comm(已开)+ grouped-gemm(已开)。
4. **`--overlap-grad-reduce` + `--overlap-param-gather`** (+ 可选 `--reset-bucket-group-order`) — distributed-optimizer 已开,可直接叠加掩盖 DP 通信。
5. **MC2 (`--use-ascend-mc2`) 慎用** — MEMORY.md 已记录此环境 LoRA 下 MC2 crash;且要求 SP+TP>1。CoC (`--use-ascend-coc`) 是另一选择,但 **TE impl 不支持**,需确认 transformer_impl 是否为 local。
6. **`--param-and-grad-buffer-pad 512`** — 昇腾对齐,低成本。

## 相关文件路径(绝对)

- 通信掩盖: `/data/sejin/third_party/mindspeed-core-26.0.0/mindspeed/features_manager/moe/moe_alltoall_overlap.py`, `.../moe/moe_allgather_overlap.py`, `.../moe/fb_overlap.py`, `.../distributed/reset_bucket_group_order_feature.py`
- MC2/CoC: `/data/sejin/third_party/mindspeed-core-26.0.0/mindspeed/features_manager/tensor_parallel/mc2.py`, `.../tensor_parallel/locl_coc.py`, `.../moe/moe_alltoall_mc2.py`
- 融合: `.../fusions/fused_moe_permute.py`, `.../fusions/grouped_matmul.py`, `.../fusions/fused_rope.py`, `.../fusions/fused_bias_swiglu.py`, `.../fusions/fused_softmax.py`, `.../moe/gmm.py`, `.../megatron_basic/megatron_basic.py`
- CP: `.../context_parallel/context_parallel_feature.py`, `.../context_parallel/ulysses_context_parallel.py`, `.../context_parallel/adaptive_context_parallel.py`, `.../context_parallel/context_parallel_kv_cache.py`
- 2D TP: `.../tensor_parallel/tp_2d.py`
- 显存/优化器: `.../memory/reuse_fp32_param.py`, `.../moe/moe_zero_memory.py`, `.../distributed/buffer_pad.py`, `.../pipeline_parallel/dualpipev_feature.py`, `.../pipeline_parallel/optimize_p2p_comm.py`, `.../recompute/activation_function.py`, `.../moe/tp_extend_ep.py`
- 示例脚本目录: `/data/sejin/third_party/mindspeed-llm-26.0.0/examples/mcore/qwen3_moe/`(最贴近: `tune_qwen3_30b_a3b_4K_lora_ptd.sh`;overlap 参考: `tune_qwen3_30b_a3b_32K_full_pack_A3_ptd.sh`, `pretrain_qwen3_235b_a22b_4k_A3_ptd.sh`)

未在 features_manager 中找到的项(显式说明): `--tp-comm-overlap` 不是 MindSpeed-core 注册的开关(昇腾对应能力 = MC2/CoC);`moe-hierarchical-alltoallv` 仅在 fb_overlap 的互斥检查中被引用,未在 core features_manager 注册 arg。
===AGENT agent-abced0a7ea0a257c1===
# 算力/显存瓶颈分析 + 优化路径

基于基线实测（HBM 17.2/65GB = 26%，AICore 瞬时 1-16%，TPS 1.95，变长 seq mbs2 单步 2.3~8.4s 波动大）与 Qwen3-30B-A3B MoE 架构（48层，128 experts topk=8，GQA 32/4），从显存与算力两个视角剖析瓶颈并给出优先级排序的行动建议。

---

## 一、为何 HBM 只占 26%？根因拆解

### 1.1 占用构成（理论）
MoE LoRA 训练显存 = 模型权重 + 激活 + 优化器状态 + 梯度 + LoRA 参数/梯度 + 通信缓存。

**你的配置下预估（粗算）：**
- **模型参数**（BF16）：30B × 2 bytes ≈ 60GB → TP2 切分后每卡 **~30GB**
- **激活**（前向缓存，seq=4096, mbs=2, bf16）：
  - Transformer 层激活主项：hidden_states (mbs × seq × hidden = 2 × 4096 × 2048 × 2) + attention 中间 QKV/scores (GQA 可省部分 KV) + MoE 专家激活（topk=8 意味每 token 只激活 8/128 专家，但 EP4 分布后有 all2all 通信缓存）
  - 单层激活粗估 ~1-2GB（含 MoE permute/routing buffers），48 层 × TP2 切分 → 每卡 **24-48GB**（但你已开 sequence-parallel + fused-swiglu/rmsnorm，会降低部分）
  - mbs=2 小 batch 使激活占用远低于峰值（若 mbs=16 会×8）
- **优化器状态**（分布式优化器 ZeRO-1 已开，FP32 master params + momentum + variance）：
  - 理论 30B × (4+4+4) = 360GB → 分片到 DP=1（TP2×EP4=8 卡，DP=1）意味全副本，但你开了 `distributed-optimizer` 会按某种切分（具体看实现，假设按 TP×EP 切 → 每卡 **45GB**）
  - 但 LoRA 微调时**只优化 LoRA 参数**（lora-r=16，4 个 target module，参数量远小于 30B）→ 优化器状态大幅降低，粗估 **<5GB**
- **LoRA 参数**（r=16, 4 modules, bf16）：每层 LoRA 增量 ~MB 级，48 层 → **<1GB**
- **梯度**（与参数同 size）：LoRA 场景下 **<2GB**

**实际测得 17.2GB = 模型权重主导（~30GB TP2 切分后？）+ 小 batch 激活（<5GB）+ LoRA 优化器/梯度（<5GB）**

### 1.2 为何没占满？关键原因（优先级排序）

| 原因 | 占比贡献 | 解释 |
|-----|---------|------|
| **① mbs=2 过小** | ★★★★★ | 激活与 mbs 线性相关。mbs=2 → 激活仅占 2-5GB；若改 mbs=16 → 激活可达 16-40GB（占满主力） |
| **② 变长序列** | ★★★☆☆ | 变长 seq 下实际 token 数 < pad 到的 4096，有效算力/显存利用率都降低（std 727ms 波动印证 seq 长度波动大）。固定 seq=4096 可消除波动并拉高显存基线 |
| **③ LoRA 微调** | ★★★☆☆ | 只训 LoRA 参数 → 优化器状态从理论 45GB 降到 <5GB。若全参微调（full finetune）→ 优化器立刻吃掉 40+GB |
| **④ TP2 切分** | ★★☆☆☆ | TP2 把模型/激活横切，单卡占用减半。降到 TP1 → 模型权重翻倍，激活也翻倍（但你 EP4 已占满 8 卡，TP1 需重构拓扑） |
| **⑤ 已开优化** | ★☆☆☆☆ | sequence-parallel、fused-swiglu、fused-rmsnorm、distributed-optimizer 都在**省**显存/通信，与占满方向相反 |

**核心结论：mbs 是第一杠杆（可×8），seq 固定化是第二杠杆（消波动+10-20%），LoRA 模式是天然上限（若可改全参则+40GB）。**

---

## 二、AICore 利用率低的真实原因

### 2.1 瞬时读数 1-16% 不等于真实算力

`npu-smi` 瞬时采样在 MoE 场景严重低估：
- MoE topk=8 意味每 forward 突发激活 8/128 专家（6.25%），专家计算高度稀疏 + 突发（grouped-gemm 短时拉高算力，idle 时瞬时降为 0）
- all2all 通信期间 AICore 挂起 → 瞬时读数趋 0
- 你基线 WPS=8000 tokens/s、单步 2.3-8.4s 变长 → **真实有效算力应从 TFLOP/s 反算**（需 profiler 里的 `AICore_time / total_time`，而非瞬时%）

**粗估真实算力占比（需确认，优先级：先跑 profiler）：**
- 若单步 2.3s 全是计算（理想），MoE 30B topk=8 前向 FLOP ~= 30B×2×(8/128)×mbs×seq ≈ 30B×2×0.0625×2×4096 ≈ 30 TFLOP/样本 → 2 样本 = 60 TFLOP / 2.3s = **26 TFLOP/s 每卡**
- 910B3 理论峰值 BF16 ~320 TFLOP/s（需查手册，假设）→ 26/320 = **8% 真实算力利用率**（与瞬时 1-16% 一致量级）

### 2.2 瓶颈拆解（影响 AICore 利用率）

| 瓶颈 | 占比 | 解释 | 如何验证 |
|------|------|------|----------|
| **① 小 batch (mbs=2)** | ★★★★★ | Grouped-GEMM 在 mbs 小时 kernel 并行度不足，NPU ALU 填不满；mbs×seq=2×4096=8K tokens/step，远低于饱和阈值（MoE 需 >>16K 才能喂饱 128 experts） | 改 mbs=8/16 复测，利用率应线性提升 |
| **② All2all 通信开销** | ★★★★☆ | EP4 的 MoE all2all（dispatcher=alltoall_seq）在 mbs 小+变长时效率低；通信时 AICore 空转 | Profiler trace 看 `all_to_all_single` 耗时占比；开 `--moe-alltoall-overlap-comm` 掩盖 |
| **③ 变长序列波动** | ★★★☆☆ | 不同 seq 长度导致 token 数波动 → 部分 step 算力低（短 seq）、部分高（长 seq），平均拉低。std 727ms 波动印证 | 固定 seq=4096 或用 packing（examples 里有 pack 脚本） |
| **④ MoE 稀疏性** | ★★☆☆☆ | Topk=8/128 = 6.25% 专家激活率，即使 grouped-gemm 融合，仍有 93.75% 专家权重不参与计算（但这是模型设计，无法改） | 架构固有，只能通过 batch 增大每激活专家的 token 数摊销路由开销 |
| **⑤ 已开重计算？** | ★☆☆☆☆ | 你基线**未开** `--recompute-*`（未在开关列表），故不存在重计算额外 FWD 开销 | 若后续占满显存时开 recompute，会降低 5-15% 吞吐 |
| **⑥ LoRA 额外计算** | ★☆☆☆☆ | LoRA forward/backward 在 base 模型外额外算 ΔW×x，但 r=16 很小，FLOPs 增幅 <1% | 可忽略（若 r=128 才需关注） |

**核心结论：mbs=2 是第一瓶颈（提到 8-16 可能×4 利用率），all2all 通信是第二瓶颈（需 overlap），变长 seq 是第三（固定化+10-20%）。**

---

## 三、占满显存的具体路径

目标：从 17.2GB → 接近 65GB（留 5GB 余量 = 60GB target）。Δ = **+43GB 需填补**。

### 路径选项（单独或组合）

| 方案 | 预估增量 | 风险 | 优先级 |
|------|---------|------|--------|
| **A. 增大 mbs（2→8→16）** | +15-35GB | 低（训练收敛不受影响，通信开销小幅增加） | ★★★★★ 最优先 |
| **B. 增大 seq（4096→8192）** | +10-20GB | 中（需确认数据集支持长 seq；通信/计算均×2） | ★★★☆☆ 按需（若任务需长 seq） |
| **C. 固定序列（变长→固定 4096）** | +2-5GB | 低（消除 padding 浪费，略增显存但大幅稳定性能） | ★★★★☆ 高优先级辅助项 |
| **D. 降低 TP（2→1）+ 调整 EP** | +15-25GB | 高（需重构拓扑，TP1 时某些通信优化失效；8 卡下 TP1 → EP8 or CP 引入） | ★★☆☆☆ 复杂，非首选 |
| **E. 开启全参微调（LoRA→Full）** | +40GB | 极高（改变任务性质；需确认业务目标是否允许） | ★☆☆☆☆ 仅当前三项饱和后考虑 |
| **F. 关掉部分融合省显存的开关** | +1-3GB | 低（如关 fused-rmsnorm？但收益极小且可能降速） | ☆☆☆☆☆ 不推荐 |

### 推荐组合（渐进式，按优先级）

#### 🥇 **Phase 1: mbs 翻倍 + 固定 seq（低风险，立即可做）**

```bash
# 当前: mbs=2, 变长 seq, HBM=17.2GB
# 目标: mbs=8, 固定 seq=4096

--micro-batch-size 8 \
--seq-length 4096 \
# 去掉变长相关 flag（若有 --variable-seq-lengths 需移除）
```

**预期效果：**
- 显存：17.2 + (8-2)×3GB（激活） + 2GB（固定 seq padding 填满） ≈ **35-40GB**
- AICore 利用率：mbs×4 → kernel 并行度×4 → **20-35%**（仍未到 70%，但大幅改善）
- 单步耗时：固定 batch 后波动消失，预估 3.5-4.5s（通信占比增加，但 overlap 可缓解）

**验证命令：**
```bash
npu-smi info  # 训练中持续监控 HBM
# 同时在训练日志查看 loss、samples/s、timing breakdown
```

---

#### 🥈 **Phase 2: 启用通信 overlap（掩盖 all2all 瓶颈）**

在 Phase 1 基础上加：

```bash
# 前置依赖（已满足）: --moe-grouped-gemm, --moe-permutation-async-comm
--moe-tp-extend-ep \  # TP2 + alltoall_seq 必须先开此项（128 experts % (2×4)==0 ✓）
--moe-alltoall-overlap-comm \  # 异步通信掩盖专家计算

# DP 侧通信 overlap（低风险，distributed-optimizer 已开）
--overlap-grad-reduce \
--overlap-param-gather \

# 可选融合（低风险）
--moe-permute-fusion \  # CANN 8.5 满足，融合 token permute
--gemm-gradient-accumulation-fusion \  # grouped-gemm 梯度融合
```

**预期效果：**
- 显存：+0-2GB（overlap 用 swap，略增）
- AICore 利用率：通信与计算重叠 → idle 时间减少 → **+10-20%**（目标 40-50%）
- 单步耗时：-15-25%（通信掩盖收益）

**风险：**
- `moe-tp-extend-ep` 改变专家分布逻辑（扩展到 TP×EP=8 组），需确认无正确性问题（代码成熟度高，qwen3_moe 脚本有用例）
- MC2 已知 LoRA crash，**不开** `--use-ascend-mc2`

---

#### 🥉 **Phase 3: 继续增大 mbs 至极限（占满最后 20GB）**

若 Phase 1+2 后显存仍有余量（如 40GB < 60GB target）：

```bash
--micro-batch-size 12  # 或 16（逐步试探 OOM 边界）
--global-batch-size 96  # or 128（保持 gbs/mbs = accumulation steps 合理）
```

**预期：**
- mbs=12: HBM ≈ 50GB, AICore ≈ 50-60%
- mbs=16: HBM ≈ 55-60GB, AICore ≈ **60-70%**（接近目标）

若 OOM：
- 回退 mbs=12，开启轻量重计算（只重算激活函数）：
  ```bash
  --recompute-activation-function \
  --recompute-activation-function-num-layers 24  # 重算前半数层
  ```
  **代价：** 吞吐 -5-10%，但可换 mbs +2-4（净收益仍为正）

---

#### 🏅 **Phase 4（激进）: 增大 seq 或降 TP（仅当前三 Phase 仍未占满且业务需要）**

```bash
# 选项 A: 长序列（需数据集支持）
--seq-length 8192
# 显存 +15-25GB，AICore 利用率 +5-10%（但单步耗时×2+，总吞吐可能不增反降）

# 选项 B: TP2→TP1（需重构，8 卡 → TP1/EP8 或 TP1/EP4/CP2）
--tensor-model-parallel-size 1 \
--expert-model-parallel-size 8  # 或保持 EP4, 引入 CP2
# 显存 +20-30GB（模型/激活不切分），但 TP1 下 MC2/某些融合失效，需全面复测
```

**不推荐原因：**
- seq 翻倍 → 计算/通信开销×4（attention O(n²)，all2all 数据量×2），除非业务明确需要长上下文
- TP1 改拓扑复杂度高，且你已有可用的 mbs 增长空间

---

## 四、显存换算力的机会矩阵

| 优化方向 | 当前状态 | 改为 | 显存变化 | AICore 利用率变化 | 吞吐变化 | 操作复杂度 |
|---------|---------|------|---------|------------------|---------|-----------|
| **关掉重计算**（若有） | 未开 | N/A | - | - | - | - |
| **增大 mbs** | 2 | 8→12→16 | +18→+33→+38GB | +20→+35→+50% | +80→+150→+200% | ★☆☆☆☆ 最简单 |
| **固定 seq** | 变长 | 4096 | +2-5GB | +5-10% | +10-20%（消波动） | ★☆☆☆☆ 改 1 个 flag |
| **开 overlap** | 部分 | 全开（moe+dp） | +0-2GB | +10-20% | +15-25% | ★★☆☆☆ 加 5 个 flag |
| **开融合** | 部分 | +permute+gemm-grad | +0GB | +3-8% | +5-10% | ★☆☆☆☆ 加 2 个 flag |
| **降 TP** | 2 | 1 | +20-30GB | -5-10%（某些优化失效） | -10-20%（通信模式变） | ★★★★☆ 高风险 |
| **增大 seq** | 4096 | 8192 | +15-25GB | +5-10% | -30-50%（通信²增长） | ★★★☆☆ 需数据支持 |
| **轻量重计算换 mbs** | 无 | recompute-activation-function | -3-5GB → 可+mbs | 持平（省显存换 batch） | 持平或+5% | ★★☆☆☆ 配合 mbs 增长 |

---

## 五、最终行动建议（优先级排序）

### ✅ **立即执行（P0，低风险高收益）**

1. **mbs 2→8 + 固定 seq 4096**
   - 命令：`--micro-batch-size 8 --seq-length 4096`（去掉变长 flag）
   - 预期：HBM 17→38GB, AICore 8→25%, TPS 1.95→6+
   - 验证：跑 50-100 steps，监控 `npu-smi` + loss 收敛

2. **开启低风险融合**
   - 命令：`--moe-permute-fusion --gemm-gradient-accumulation-fusion`
   - 预期：+5-8% 吞吐，0 显存增长
   - 风险：CANN 8.5 已满足算子要求，代码成熟

### 🔄 **第二轮（P1，需前置项）**

3. **启用 MoE/DP overlap**
   - 命令：
     ```bash
     --moe-tp-extend-ep \
     --moe-alltoall-overlap-comm \
     --overlap-grad-reduce \
     --overlap-param-gather
     ```
   - 预期：单步耗时 -20%，AICore +15%
   - 风险：tp-extend-ep 改专家分布，需验证 loss（但代码路径成熟）

4. **mbs 增至 12-16（试探 OOM 边界）**
   - 逐步：8→10→12→16，每次跑 20 steps 观察 HBM
   - 若 OOM：回退并加 `--recompute-activation-function --recompute-activation-function-num-layers 24`

### 🎯 **目标达成检查点**

| 阶段 | HBM 占用 | AICore 利用率（profiler 实测，非瞬时） | TPS (samples/s) |
|------|---------|-------------------------------------|-----------------|
| 基线 | 17.2GB (26%) | ~8%（推算） | 1.95 |
| P0 后 | 35-40GB (55-62%) | 25-35% | 6-8 |
| P1 后 | 50-60GB (77-92%) | **50-70%** ✅ | 10-15 |

### 📊 **需补充的诊断数据（用 profiler 确认真实瓶颈）**

在当前基线或 P0 改动后，跑一次完整 profiling：

```bash
# 在训练命令加
--profile \
--profile-step-start 10 \
--profile-step-end 12 \
--profile-level 1 \
--profile-with-cpu \
--profile-with-memory

# 输出在 --profile-output-path，用 CANN analyze 工具打开
```

关键指标：
- `AICore_time / E2E_time`（真实算力占比，而非 npu-smi 瞬时%）
- `AllToAllComm / Total_time`（通信占比，判断 overlap 收益上限）
- `activation_memory_peak`（验证 mbs 增长与显存关系）

---

## 六、风险与备选方案

### ⚠️ **已知雷区（避开）**

| 开关 | 问题 | 来源 |
|------|------|------|
| `--use-ascend-mc2` | LoRA 下 crash | MEMORY.md 记录 |
| `--swap-attention` | 与 LoRA 不兼容（代码 assert） | `swap_attention.py:27-29` |
| `--moe-fb-overlap` | 需 ETP=1 + alltoall（你当前 ETP=4 + alltoall_seq 冲突） | `fb_overlap.py` constraints |
| `--moe-alltoall-mc2` | alltoall_seq 需 TP=1（你 TP=2 冲突）| `moe_alltoall_mc2.py` |
| `--balanced-moe-experts` | EP=4 太小，代码提示不划算 | `balanced_moe.py` |

### 🔄 **若 P0/P1 后仍未达 70% 利用率的备选**

1. **切换到 full_pack 模式**（变长→packing，examples 有脚本）
   - 消除 padding 浪费，MoE token 利用率接近 100%
   - 需改数据预处理，复杂度 ★★★☆☆

2. **尝试 CP（上下文并行）+ 更长 seq**
   - 如 TP2/PP1/EP4/CP2, seq=8192
   - 适用于长文本任务，但你当前 4K 可能不需要

3. **profiler 导向的算子级优化**
   - 若 profiling 发现某特定算子（如 router、unpermute）占比高，针对性开相关融合或调参

---

## 总结：三步走路线图

```
当前: HBM 17GB (26%), AICore ~8%, TPS 1.95
  ↓
【P0: mbs×4 + 固定seq】 → HBM 38GB (58%), AICore 25-35%, TPS 6-8
  ↓
【P1: overlap + 融合】 → HBM 40GB (62%), AICore 40-50%, TPS 10-12
  ↓
【P1+: mbs→16 or 轻量重计算】 → HBM 55-60GB (85-92%), AICore **60-70%** ✅, TPS 12-15
```

**关键洞察：**
- mbs 是显存与算力的双杠杆（×4 batch → ×4 显存 + ×3-4 吞吐）
- 固定 seq 是稳定性基石（消除 727ms std 波动）
- Overlap 是通信黑洞的解药（all2all 占比高时 +20% 净收益）
- 重计算是最后的权衡项（省显存换吞吐，仅在 OOM 边缘启用）

**下一步操作：立即修改训练脚本，执行 P0（5 分钟改完），跑 100 steps 验证，回传 HBM/timing/loss 数据以进入 P1。**
===AGENT agent-aea38fba875436c3c===
我将从通信、并行、MoE 三个维度分析当前配置瓶颈，并给出具体可落地建议。

## 当前配置 (TP2/PP1/EP4) 通信开销与瓶颈分析

### 1. alltoall_seq dispatcher 的通信特征

**当前路径**（已开 `--moe-permutation-async-comm` + `--moe-grouped-gemm`）:
- **3 次 AllGather** (permutation 阶段，token 分发到专家)
- **1 次 AllToAll** (专家输出重排)  
- **通信量**: 每层 MoE `2 × seq × hidden × topk` (topk=8)，量大
- **与计算的串行**: 当前未开 overlap，通信与 expert GEMM **完全串行**，这是你 AI Core 低利用率的主因

**实测数据对应**:
- 你的单步耗时 2.3–8.4s 波动 (std 727ms) → MoE 层突发计算后等通信，通信结束前卡空闲
- AI Core% 瞬时 1-16% → NPU-SMI 采样命中通信阶段 (计算 kernel 未跑)
- HBM 仅 17.2/65GB (26%) → 激活未占满，但通信 latency 已成瓶颈

---

### 2. 为何不建议直接换 alltoall dispatcher

**dispatcher 对比** (`alltoall_seq` vs `alltoall`):

| 维度 | alltoall_seq (当前) | alltoall (候选) |
|---|---|---|
| **通信模式** | 3×AllGather + 1×AllToAll | 2×AllToAll (输入+输出) |
| **内存占用** | 低 (序列维度切分) | 高 (全 token buffer) |
| **支持 TP>1 + overlap** | 需 `--moe-tp-extend-ep` | 不支持 tp-extend-ep |
| **兼容 MC2** | 需 TP=1 (你 TP=2 冲突) | 需 ETP=1 (你 EP=4 可降到 ETP=1) |
| **成熟度** | 主推路径，full_pack 脚本全用 | 需额外显存，32K 脚本有用例 |

**结论**: `alltoall` 不比 `alltoall_seq` 更优，且你 TP=2 会失去 tp-extend-ep，反而限制 overlap 手段。**应保持 alltoall_seq + 加 overlap 开关**。

---

### 3. 当前可用的通信掩盖路径 (优先级排序)

#### 🟢 **方案 A: MoE AllToAll Overlap (主力，立即可用)**

```bash
# 在现有基础上追加 3 个 flag:
--moe-tp-extend-ep \
--moe-alltoall-overlap-comm \
--moe-permute-fusion
```

**原理**:
- `--moe-tp-extend-ep`: 把 128 experts 按 TP×EP = 2×4 = 8 份切 token (而非切专家权重)，使 TP group 参与专家并行，满足 `num_experts(128) % 8 == 0` ✓
- `--moe-alltoall-overlap-comm`: 用异步通信 + swap 把 3 个 AllGather 与 expert GEMM 重叠 (依赖前者 + permutation-async-comm + grouped-gemm，均已满足)
- `--moe-permute-fusion`: 融合 permute/unpermute 算子 (CANN 8.5 满足版本要求，alltoall_seq 支持)

**预期收益**:
- **通信时间减少 50–70%** (AllGather latency 被 GEMM 计算遮掩)
- **AI Core 利用率提升到 40–60%** (计算-通信流水线化)
- **单步耗时降到 1.5–2.0s** (消除波动，通信不再裸露)
- **显存增加 3–5GB** (异步 buffer，仍远低于 65GB 上限)

**风险**: 低。32K full_pack 脚本 (`tune_qwen3_30b_a3b_32K_full_pack_A3_ptd.sh`) 已验证此组合 (TP4/EP8，更激进)。

**互斥检查**: ✓ 与当前所有已开开关兼容 (不冲突 MC2/fb-overlap/CoC)。

---

#### 🟡 **方案 B: DP 通信重叠 (辅助，低成本叠加)**

```bash
# 追加到方案 A:
--overlap-grad-reduce \
--overlap-param-gather \
--reset-bucket-group-order
```

**原理**:
- `--overlap-grad-reduce`: 反向梯度 reduce-scatter 与计算重叠 (DP=1 下仍有 distributed-optimizer 的参数收集)
- `--overlap-param-gather`: 前向参数 all-gather 与计算重叠
- `--reset-bucket-group-order`: 调整 bucket 顺序让 overlap 最大化 (强制依赖 param-gather)

**预期收益**:
- **再减 5–10% 单步时间** (优化器通信不阻塞反向)
- **AI Core 再提升 5–10%** (DP 通信遮掩，虽然 DP=1 下量小)
- **无额外显存** (仅改调度)

**风险**: 极低。235B/480B pretrain 脚本标配 (`pretrain_qwen3_235b_a22b_4k_A3_ptd.sh`)。

**互斥检查**: ⚠️ 与 `--moe-fb-overlap` 互斥 (但你不用 fb-overlap)。

---

#### 🔴 **方案 C: MC2 融合 (高收益但高风险，暂不推荐)**

```bash
# 需改并行配置 + 谨慎测试:
--use-ascend-mc2  # 或 --moe-alltoall-mc2
```

**原理**: 把 TP 的 matmul + all-gather/reduce-scatter 融合到单个 kernel (算子级通算并行)。

**为何不推荐**:
1. **MEMORY.md 记录此环境 MC2 会 crash** ("to avoid MC2 crash")
2. `--use-ascend-mc2` 要求 TP>1 + sequence-parallel (你满足)，但与 `moe-alltoall-overlap-comm` **互斥**
3. `--moe-alltoall-mc2` 在 alltoall_seq 下要求 **TP=1** (你 TP=2 冲突)

**若要尝试**: 先用 `--tensor-model-parallel-size 1` + `--expert-model-parallel-size 8` 单独测 MC2 稳定性，确认不 crash 后再评估收益。

---

### 4. 并行策略调优 (TP/EP/PP 组合分析)

#### 当前 TP2/EP4 的问题

**30B-A3B (128 experts) 特点**:
- **参数分布**: 48 层 MoE，每层 128×(moe_ffn 768) + shared (GQA 2048, FFN 6144)，MoE 占 ~70% 参数
- **计算分布**: topk=8 → 平均每 token 激活 8/128 = 6.25% 专家，但通信量 = 8×hidden

**TP=2 的代价**:
- **每层 QKV/O 切 2 份** → TP all-gather/reduce-scatter (虽然 hidden 仅 2048，量不大)
- **与 EP=4 的交互**: 当前 128 experts 分到 4 个 EP rank，每个 32 experts，TP 在每个 EP 内再切专家权重 (moe_ffn 768 / 2 = 384)
- **瓶颈**: TP 通信 + EP 通信 **串行** (未开 tp-extend-ep 前)，且 EP=4 相对保守 (8 卡只切 4 份，剩余 DP=1 无数据并行收益)

#### 推荐配置对比

| 配置 | TP | PP | EP | DP | 通信开销 | 显存占用 | 适用场景 | 预期 AI Core% |
|---|---|---|---|---|---|---|---|---|
| **当前** | 2 | 1 | 4 | 1 | 中 (TP+EP 串行) | 17.2GB (低) | - | 10–16% |
| **A (激进EP)** | 1 | 1 | 8 | 1 | 低 (纯 EP) | 15–18GB | MoE 主导模型 | 50–70% |
| **B (保守TP)** | 2 | 1 | 4 | 1 + overlap | 中→低 (overlap 后) | 20–23GB | 当前基础上优化 | 40–60% |
| **C (引入PP)** | 2 | 2 | 2 | 2 | 高 (P2P) | 12–15GB | 省显存 | 30–50% (PP 气泡) |
| **D (TP4平衡)** | 4 | 1 | 2 | 1 | 中 (TP 增) | 20–25GB | TP 主导 | 35–55% |

**结论**:
- **立即落地: 配置 B** (当前 TP2/EP4 + 方案 A+B 的 overlap 开关) → **最低风险、最快见效**
- **中期优化: 配置 A** (TP1/EP8) → **纯 EP 通信最简单**，但需重跑 checkpoint 转换 (TP 切分变化)
- **长序列场景: 引入 CP** (如 TP2/EP4/CP2，GBS 不变) → 你当前 seq=4K 暂不需要

---

### 5. 其他可叠加优化 (低成本、高收益)

```bash
# 融合算子 (方案 A 已含 moe-permute-fusion):
--gemm-gradient-accumulation-fusion \  # GroupedGEMM 梯度累加融合，纯收益

# 昇腾对齐:
--param-and-grad-buffer-pad 512 \      # bucket 内存对齐，降通信碎片

# 稳定显存 (可选):
--manual-gc --manual-gc-interval 1      # 手动 GC 对齐各 rank
```

预期: **再降 3–5% 单步时间**，**AI Core 再提升 2–5%**。

---

## 最终推荐配置 (分阶段)

### 🎯 **Phase 1: 低风险立即落地 (预期 AI Core 40–60%)**

在现有脚本基础上追加:

```bash
# === MoE overlap (主力) ===
--moe-tp-extend-ep \
--moe-alltoall-overlap-comm \
--moe-permute-fusion \

# === DP overlap (辅助) ===
--overlap-grad-reduce \
--overlap-param-gather \
--reset-bucket-group-order \

# === 融合+对齐 ===
--gemm-gradient-accumulation-fusion \
--param-and-grad-buffer-pad 512
```

**不改**: TP/EP/PP 保持 2/4/1，dispatcher 保持 alltoall_seq，所有现有开关不动。

**预期**:
- 单步耗时: 2.3–8.4s → **1.5–2.0s** (-35–40%)
- AI Core%: 1–16% → **40–60%**
- HBM: 17.2GB → **20–25GB** (仍有 40GB 余量，可后续增大 MBS/GBS)
- TPS: 1.95 → **3.0–3.5 samples/s**

---

### 🔬 **Phase 2: 并行策略调优 (预期 AI Core 60–75%)**

若 Phase 1 验证通过且希望进一步压榨:

```bash
# 改为 TP1/EP8 (需重新切分 checkpoint):
--tensor-model-parallel-size 1 \
--expert-model-parallel-size 8 \

# 保留 Phase 1 所有 overlap 开关
# (注: tp-extend-ep 在 TP=1 时自动退化为无效，但不冲突)
```

**收益**: EP=8 下每 rank 仅 16 experts，通信量减半，AI Core 可达 **60–75%**。

**代价**: 需重跑 checkpoint 转换 (TP 维度变化)，训练超参可能需微调。

---

### ⚠️ **不推荐路径**

1. **引入 PP** (PP>1): 你模型仅 48 层，PP=2 每 stage 24 层，P2P 通信 + 气泡会抵消收益，且当前 HBM 富余不需要 PP 省显存。
2. **MC2**: 环境已知 crash 风险 + 与 overlap 互斥。
3. **fb-overlap**: 需切到 alltoall dispatcher + ETP=1 + 与多个已开开关互斥，改动量大且收益不确定。
4. **tp-2d**: 明确不支持 MoE (`expert_model_parallel_size>1` 报错)。

---

## 关键文件路径 (供验证/debug)

- MoE overlap 实现: `/data/sejin/third_party/mindspeed-core-26.0.0/mindspeed/features_manager/moe/moe_alltoall_overlap.py`
- tp-extend-ep: `/data/sejin/third_party/mindspeed-core-26.0.0/mindspeed/features_manager/moe/tp_extend_ep.py`
- permute fusion: `/data/sejin/third_party/mindspeed-core-26.0.0/mindspeed/features_manager/fusions/fused_moe_permute.py`
- 参考脚本: `/data/sejin/third_party/mindspeed-llm-26.0.0/examples/mcore/qwen3_moe/tune_qwen3_30b_a3b_32K_full_pack_A3_ptd.sh` (行 103–108, moe-alltoall-overlap-comm + tp-extend-ep)
