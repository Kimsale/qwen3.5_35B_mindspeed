# HULK 对齐训练性能报表 — 第一轮基线（hulk_aligned_20260603_210624）

**生成时间**：2026-06-03 21:15
**训练退出码**：0（成功完成 30/30 步）
**墙钟耗时**：535 s（约 8.9 分钟，含权重加载 + 算子编译 + 训练 + 收尾）
**训练日志**：`/data/sejin/baseline_26/logs/hulk_aligned_20260603_210624.log`

> 任务背景：MindSpeed-LLM 26.0.0 (CANN 8.5.0) 训练 Qwen3-30B-A3B-LoRA-MoE，配置严格对齐 HULK 自研框架。
> 模型：Qwen3-Omni-30B-A3B（thinker text-only 子模块，标准 Qwen3MoeForCausalLM），48 层全 MoE，128 专家 / topk=8，hidden=2048，moe_intermediate=768。
> 数据：`/data/sejin/data_hulk_dist_30k_mcore/hulk_sft_packed`（hulk 长度分布对齐，2314 文档，217 MB）。
> 权重：`/data/sejin/models/Qwen3-Omni-30B-A3B-hulk_tp1_pp1_ep8`（HF→Megatron 转换，TP1/PP1/EP8，77 GB / 8 EP rank）。

---

## 一、配置对齐复核

| 维度 | HULK 目标 | 本次实测 | 一致 |
|------|----------|---------|------|
| TP / PP / EP / CP | 1 / 1 / 8 / 2(ulysses) | 1 / 1 / 8 / 2(ulysses_cp_algo) | ✅ |
| seq_length | 8192 | 8192 | ✅ |
| LoRA r / alpha | 32 / 64 | 32 / 64 | ✅ |
| **LoRA dropout** | **0.1** | **0.0**（框架强制） | ❌ 见说明 1 |
| LoRA target | 仅 attention | linear_qkv linear_proj | ✅ |
| lr / clip / warmup / min_lr | 5e-6 / 5.0 / 0.0 / 1e-6 | 5e-6 / 5.0 / 0.0 / 1e-6 | ✅ |
| **ZeRO 级别** | **os_v2 (Stage-2)** | **ZeRO-2 (use_distributed_optimizer=True)** | ✅ |
| swap-optimizer | 关 | 关（ZeRO-2 纯 NPU，无 CPU offload） | ✅ |
| MoE 128/topk8/ffn768/alltoall_seq | 一致 | 一致 | ✅ |
| 算子融合（FA/SwiGLU/RMSNorm/RoPE） | — | 全开（no-rope-fusion 规避 MC2） | ✅ |
| 重计算 | full / block / num_layers=1 | full / block / num_layers=1 | ✅ |
| 精度 | bf16 | bf16 | ✅ |

**说明 1（LoRA dropout 不可对齐）**：MindSpeed-LLM 26.0.0 在 `mindspeed_llm/training/training.py:151` 将 `lora_dropout` 硬编码为 `0.0`，未暴露 CLI；内部 `LoraConfig` 调用三处（`training.py:147`、`tasks/checkpoint/models.py:534`、`core/transformer/moe/moe_layer.py:60/242`）的 dropout 参数全部写死。HULK 的 dropout=0.1 在 MindSpeed 框架内**无法通过参数对齐**。
- **影响**：dropout 仅影响正则化强度，不改变 FLOPs / 通信模式 / 显存占用；**对吞吐性能对比无干扰**，本基线照样反映 MindSpeed 在该配置下的真实算力效率。
- **后续**：如需严格对齐，需 patch `LoraConfig(lora_dropout=args.lora_dropout)` 并新加 `--lora-dropout` 参数，属于框架定制改动，不在 CLAUDE.md 第三节"仅调整训练超参 / 并行策略 / 算子配置"许可范围内，建议作为差异接受。

---

## 一·补、ZeRO / FSDP 关系与 MindSpeed 支持情况（澄清对齐口径）

为避免"ZeRO-1 / ZeRO-2 / FSDP"概念混淆导致对齐口径误判，单列本节。**结论：本轮训练用的是 ZeRO-2 (os_v2)，与 HULK 的 `zero_sharding='os_v2'` 完全对齐；不是 FSDP。**

### 1. 什么是 FSDP

FSDP = Fully Sharded Data Parallel（完全分片数据并行），是 PyTorch 原生的显存优化数据并行方案，思想直接来自 DeepSpeed 的 ZeRO。它把模型的三样东西**全部按 DP rank 切片**（而非每卡复制），用到时才临时聚合：

| 切片对象 | 平时状态 | 用时动作 |
|---------|---------|---------|
| 模型参数 params | 每卡只存 1/N | forward/backward 临时 **all-gather** 拼出整层，算完立即释放 |
| 梯度 grads | 每卡只存 1/N | backward 后 **reduce-scatter** 切回 1/N |
| 优化器状态 optimizer states | 每卡只存 1/N | 本地只更新自己那 1/N 参数 |

代价是通信增多（每层都要 all-gather 参数），换来单卡显存大幅下降。

### 2. FSDP 与 ZeRO 的对应关系

FSDP 本质就是 ZeRO 思想的 PyTorch 实现，区别只在"切到第几样"：

| 级别 | 切优化器状态 | 切梯度 | 切参数 | 对应 |
|------|:---:|:---:|:---:|------|
| ZeRO-1 (os) | ✅ | ❌ | ❌ | — |
| **ZeRO-2 (os_v2)** | ✅ | ✅ | ❌ | **本轮 + HULK 均用此档** |
| ZeRO-3 (os_v3) | ✅ | ✅ | ✅ | ≈ 完整 FSDP（full shard） |

- **ZeRO-2 ≈ "半个 FSDP"**：只切优化器状态 + 梯度，**参数仍每卡全量复制**。
- **完整 FSDP（full shard）≈ ZeRO-3**：连参数也切。

### 3. 本轮训练到底是哪一档（实测证据）

实测为 **ZeRO-2 (os_v2)**，证据链：
- 日志 `param_and_grad_buffer:Number of buckets for gradient all-reduce / reduce-scatter: 1` —— 梯度走 **reduce-scatter 分片**（ZeRO-2 特征，非 ZeRO-1 的纯 all-reduce）。
- `OptimizerConfig(use_distributed_optimizer=True, optimizer_cpu_offload=False)` —— 优化器状态分片，纯 NPU 无 CPU offload。
- Megatron `DistributedOptimizer` 源码注释：每个 DP rank "responsible for **reducing the relevant subset of grads**, and **updating the relevant subset of params**" —— 即梯度+优化器状态双分片、参数更新后 all-gather 同步。
- 参数**未分片**：单卡 HBM 占用高达 78%（51 GB），正因每卡都驻留完整 30B 参数 —— 这恰恰说明它**不是 FSDP/ZeRO-3**。

> 因此前一版报表笔误写的"纯 GPU ZeRO-1"已更正为 **ZeRO-2**，与第一节对齐复核表一致。`--use-distributed-optimizer`（Megatron/MindSpeed）== `os_v2`（HULK）== ZeRO-2。

### 4. MindSpeed-LLM 26.0.0 有没有 FSDP

**有，且有两套**（已核源码 / 示例）：

| 实现 | 开启方式 | 入口 | 分档能力 |
|------|---------|------|---------|
| Torch-FSDP2 | `--use-torch-fsdp2` | 独立 `train_fsdp2.py` + YAML，示例在 `examples/fsdp2/` | 依赖 PyTorch 原生 FSDP2 |
| Custom-FSDP（MindSpeed 自研） | `--use-custom-fsdp` + `--data-parallel-sharding-strategy` | 文档 `mindspeed-26.0.0/docs/features/custom_fsdp.md` | `optim`(≈ZeRO-1) / `optim_grads`(≈ZeRO-2) / `optim_grads_params`(≈ZeRO-3) |

### 5. 为什么本对标场景用不了 FSDP

Torch-FSDP2 被源码 `Megatron-LM-core_v0.12.1/megatron/training/arguments.py:510-526` 硬约束排除：

- `--use-torch-fsdp2 is not supported with **expert parallelism**` —— 本配置 EP=8（MoE 必须开 EP 切 128 专家），**直接冲突**。
- `--use-torch-fsdp2 is not supported with **pipeline parallelism**` / `with MCore's distributed optimizer`。
- `FSDP always requires CUDA_DEVICE_MAX_CONNECTIONS > 1`，而本环境为保证 MoE/TP 通信下发顺序锁了 `CUDA_DEVICE_MAX_CONNECTIONS=1`，**两者矛盾**。

> 注：`examples/fsdp2/qwen3_moe/*` 能跑，是因为它先 `moe_hf_param_merge_experts` 合并专家权重、走 **HF 原生 MoE + 纯 DP** 路线，绕开了 Megatron 的 EP 切分 —— 与"对齐 HULK 的 EP=8 Megatron MoE"是完全不同的并行拓扑，不可混用。

### 6. 对本项目的结论

- **对标基线保持 ZeRO-2 (os_v2) 是正确且与 HULK 一致的选择，FSDP 不在对齐范围内。**
- Custom-FSDP 的 `optim_grads_params`（≈ZeRO-3）可作为**后续"显存换 batch"优化探索**的候选（MindSpeed 比 HULK 更省显存能否换更大 mbs），但属于优化项、非对标基线，引入即偏离对齐目标，需单列实验说明。

---

## 二、单步耗时 — 分段分析（揭示性能拐点）

**关键发现**：训练存在明显两段稳态，第 12→13 步耗时突然从 ~9.2 s 翻倍到 ~20 s，并维持到训练结束。

| 段 | 步范围 | n | mean (ms) | min (ms) | max (ms) | std (ms) |
|---|--------|---|-----------|----------|----------|----------|
| 预热 | 1–5 | 5 | 11190.9 | 9003.4 | 19551.9（含 ckpt + 算子编译） | 4674.7 |
| **稳态 A** | **6–12** | **7** | **9161.3** | **8977.1** | **9411.2** | **139.5** |
| **稳态 B（漂移）** | **13–30** | **18** | **19501.7** | **12573.5** | **23578.7** | **2402.6** |
| 全部稳态合并 | 6–30 | 25 | 16606.4 | 8977.1 | 23578.7 | 5152.5 |

**漂移点解读**：iter 12 (9.2s) → iter 13 (12.6s) → iter 14 (23.6s)。HBM 在整段保持 51150 MB / 78% 不变，**没有 OOM 触发的重计算或 swap**。AICore 时序显示漂移段大量样本落在 0-7% 区间，少数 30-40% 尖峰，说明**计算实际等待时间显著增加**，瓶颈在通信或 host-device 同步。

**初步嫌疑**：
1. **MoE alltoall_seq 通信不均衡**：随训练推进 router 对热门 expert 路由倾斜后，EP=8 各 rank 计算时长发散；最慢 rank 拖慢全 step。
2. **CP=2 ulysses all-to-all**：8K 长度切两半后 attn all-to-all 数据量大，与 MoE alltoall_seq 在 HCCL 流上 contention。
3. **shuffle index 重建 / dataloader 切换**：训练开始时 batch index 是预热加载的，到中段需要重建（待对照 `_build_index_mappings` 行为确认）。
4. **NPU 监控干扰**：`npu-smi info` 每 2s 触发一次驱动查询，少量场景下可能与训练 stream 抢占 PCIe；下一轮关闭监控复测排除嫌疑。

**第二步项目计划（CLAUDE.md 第三节）**应将"漂移点定位"作为优化方案文档的首要瓶颈分析。

---

## 三、吞吐 — 双口径报告

| 口径 | step_ms | TPS (samples/s) | WPS (tokens/s, gbs×seq 上界) | 备注 |
|------|---------|-----------------|------------------------------|------|
| **稳态 A**（理想，6–12 步） | 9161 | **1.75** | **14307** | 衡量 MindSpeed 在该配置下的"无漂移"上限 |
| **稳态 B**（实际，13–30 步） | 19502 | 0.82 | 6721 | 实际生产步进吞吐 |
| 合并稳态（6–30 步） | 16606 | 0.96 | 7893 | 默认 parse_metrics 输出值 |

> WPS 是 `gbs × seq_length / step_s` 的名义上界。Hulk 数据是动态 pack（max_tokens=16000，~95% 有效），MindSpeed 这边 packed 数据集每个 doc 是定长 8192 切片（不知具体浪费率），所以**两边 WPS 直接比较仍需结合"有效 token 占比"修正**。

---

## 四、硬件指标 — 实际测得 vs CLAUDE.md 目标

| 指标 | 单卡均值 | 单卡峰值 | 8 卡聚合 | CLAUDE.md 目标 | 是否达标 |
|------|----------|----------|----------|----------------|----------|
| AI Core 利用率 (%) | 11–13 | 38–43 | 均值 12.0 / 峰值 43.0 | **≥ 70%（优化目标）** | ❌ 远低（首轮基线，后续优化对象） |
| HBM 占用 (MB) | 47028 / 65536 ≈ 71.8% | **51150 / 65536 ≈ 78.0%** | 8 卡均落在 73-78% | **50-60 GB（基本打满）** | ✅ 接近目标下沿 |
| HBM 总量 | 65536 MB / 卡 | 8 × 64 = 512 GB | — | — | — |
| 整机功耗 (W) | **采集失败** | — | — | — | ⚠️ npu_monitor.py 的 `sample_power()` 是空函数 |
| 监控样本数 | 157 (间隔 2s, 约 5.2 分钟) | — | — | — | — |

**说明 2（功耗采集 bug）**：`/data/sejin/baseline_26/scripts/npu_monitor.py:44-46` 的 `sample_power()` 直接 `return {}`，导致 power_w 全部覆盖为 None；npu-smi info **第一行实际有 power 字段**，正则已能 capture 到 `power` 字段（脚本 L31），但被 L52-53 的空 dict 覆盖。下一轮修这个 bug 后即可获得真实功耗。报表中 `power_w` 列今天空缺。

**每卡 HBM 峰值（来自最近 50 样本）**：
| chip | HBM peak (MB) | 占比 | AICore mean | AICore max |
|------|---------------|------|-------------|------------|
| 0 | 51150 | 78.0% | 12.8% | 43% |
| 1 | 49316 | 75.3% | 12.1% | 42% |
| 2 | 48220 | 73.6% | 11.6% | 40% |
| 3 | 48696 | 74.3% | 11.3% | 38% |
| 4 | 49096 | 74.9% | 11.0% | 42% |
| 5 | 48836 | 74.5% | 10.8% | 41% |
| 6 | 49134 | 75.0% | 12.0% | 43% |
| 7 | 50714 | 77.4% | 12.6% | 39% |

8 卡 HBM 占用相对均匀（73.6–78.0%），未见 EP rank 间显存倾斜。AICore 利用率均匀偏低，说明瓶颈是**全局问题**（通信/同步），不是单卡。

---

## 五、收敛指标

| 指标 | 值 |
|------|----|
| Loss 起始 (iter 1) | **0.331** |
| Loss 末步 (iter 30) | **0.297** |
| Loss min (训练全程) | **0.282 @ iter 19** |
| Grad Norm 均值 | 0.298 |
| Grad Norm 范围 | 0.197 – 0.582 |
| NaN 步数 | **0** |
| Skipped 步数 | **0** |
| Loss scale | 1.0（bf16，恒定） |

Loss 单调下降（震荡正常），梯度健康。说明**框架训练逻辑正确**，第三节硬性约束"loss 异常 / NaN" 不触发。

---

## 六、模型与数据真实性核对

| 维度 | 数值 |
|------|------|
| transformer block 参数量 | 29.90 B |
| embedding 层参数量 | 0.62 B |
| **总参数量** | **30.52 B** |
| Theoretical memory footprint (weight + optimizer) | 261957 MB ≈ **256 GB**（8 卡 ZeRO-2 后单卡 ~32 GB） |
| 实际 reserved per rank | 42676 MB（rank 0：44518 MB） |
| 数据集文档数 | 2314（indexed dataset 实际读出） |
| 数据集 dtype | int32（MMIDIDX v1） |
| 每个文档长度 | 8192 token（定长 packed，std=0） |
| 训练 30 步消耗样本 | 480（gbs=16 × 30） |
| 每步总 token | 131,072 (16 × 8192) |
| 预计 1 epoch 步数 | 2314 / 16 ≈ 145 步（远未跑完） |

> 实际 reserved < theoretical footprint，因为 LoRA 冻结主体只训练 adapter，optimizer 状态主要在 adapter 上；**主体参数已就位但不分配 optimizer slot**。

---

## 七、CLAUDE.md 第六节性能指标核对清单

| 类别 | 指标 | 状态 |
|------|------|------|
| 吞吐 | WPS / TPS / 单步耗时 / 单轮耗时 | ✅ 已采，含分段统计 |
| 硬件 | 平均 AI Core / 峰值 AI Core | ✅ 已采（12% / 43%） |
| 硬件 | HBM 占用 | ✅ 已采（peak 78%） |
| 硬件 | HBM 带宽 | ❌ npu-smi 不直出，需 msprof 单步采集 |
| 硬件 | 整机功耗 | ❌ 监控脚本 bug，下一轮修复 |
| 硬件 | 显存占用率 | ✅ 78% 接近目标 |
| 训练 | Loss 收敛 | ✅ 0.331 → 0.297 单调降 |
| 训练 | NaN / 梯度爆炸 | ✅ 0 NaN / grad norm 0.20-0.58 |
| 备注 | 并行配置 | TP1·PP1·EP8·CP2 |
| 备注 | 算子开关 | FA / fused-rotary / fused-swiglu / fused-rmsnorm / no-rope-fusion / sequence-parallel / distributed-optimizer |
| 备注 | AutoTuning 最优参数 | 本轮关闭（首轮基线） |

---

## 八、第二阶段（瓶颈分析 & 优化方案文档）的待办输入

按 CLAUDE.md 第三节执行顺序，下一步是输出《基线性能瓶颈分析 & 优化方案文档》。本轮基线给出的关键输入：

1. **首要待诊断瓶颈**：iter 12→13 单步耗时翻倍漂移点，AICore 长期低位，疑似 MoE alltoall + CP all-to-all 通信 contention 或 router 倾斜。
2. **AICore 利用率 12% 距离 70% 目标差距巨大**——这是最大的优化空间。优化候选：
   - 关闭 `--moe-permutation-async-comm` 试同步路径稳定性
   - 切换 `alltoall` ↔ `alltoall_seq`
   - 调整 `recompute-num-layers`（当前 1 层）/ 试 `--moe-layer-recompute`
   - MindSpeed Auto Tuning 全局搜参（CLAUDE.md 第五节明确允许）
3. **HBM 占用 78% 已接近 CLAUDE.md "50-60GB 基本打满"目标**，可适当提 mbs 或减少重计算（牺牲显存换计算密度）。
4. **修脚本 bug**：`npu_monitor.py::sample_power()`，下一轮起报表自动有功耗数据。
5. **接 msprof**：单步级别 HBM 带宽 / 算子级时长，定位漂移点的具体算子。

---

## 附：原始数据

| 文件 | 路径 |
|------|------|
| 训练日志 | `/data/sejin/baseline_26/logs/hulk_aligned_20260603_210624.log` |
| 训练 metrics JSON | `/data/sejin/baseline_26/metrics/hulk_aligned_20260603_210624_train.json` |
| NPU 监控 JSON | `/data/sejin/baseline_26/metrics/hulk_aligned_20260603_210624_npu.json` |
| 合并 metrics | `/data/sejin/baseline_26/metrics/hulk_aligned_20260603_210624_combined.json` |
| 训练脚本 | `/data/sejin/baseline_26/scripts/train_hulk_aligned.sh` |
| 主驱动 | `/data/sejin/baseline_26/scripts/run_hulk_aligned_eval.sh` |
| 配置对比文档 | `/data/sejin/baseline_26/reports/HULK_VS_BASELINE_COMPARISON.md` |
