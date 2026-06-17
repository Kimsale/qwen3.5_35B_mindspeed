# Hulk 对齐配置 - 执行计划与风险评估

**生成时间**: 2026-06-03 17:45
**目标**: 将 MindSpeed-LLM 基线配置严格对齐到 Hulk 训练参数

---

## 一、已完成的准备工作

### 1.1 权重转换脚本
- **位置**: `/data/sejin/baseline_26/scripts/convert_weights_tp1_pp1_ep8.sh`
- **状态**: 正在后台运行（PID 1712183）
- **源**: `/data/xuchen2/model/Qwen3-Omni-30B-A3B-Captioner` (HF 格式)
- **目标**: `/data/sejin/checkpoints/qwen3_omni_30b_a3b_mcore_tp1_pp1_ep8` (MCore 格式 TP1/PP1/EP8)
- **预计耗时**: 15-30 分钟
- **监控命令**: `tail -f /data/sejin/baseline_26/logs/convert_tp1_pp1_ep8_*.log`

### 1.2 训练配置脚本
- **位置**: `/data/sejin/baseline_26/scripts/train_hulk_aligned.sh`
- **状态**: 已生成，等待权重转换完成和数据路径配置

---

## 二、配置改动清单（7 项对齐）

| # | 维度 | 基线 | Hulk | 改动状态 |
|---|---|---|---|---|
| ~~1~~ | ~~模型架构~~ | ~~MoE-A3B~~ | ~~MoE-A3B~~ | ✅ 不需要改（已确认两边一致）|
| 2 | **并行策略** | TP2/PP1/EP4/CP1 | TP1/PP1/EP8/CP2 | ✅ 已改 |
| 3 | **swap-optimizer** | 开启（32 次分段）| 不开 | ✅ 已去掉 |
| 4 | **LoRA 配置** | r16/α32/dropout0/target含MLP | r32/α64/dropout0.1/仅attn | ✅ 已改 |
| 5 | **序列长度** | 4096 | 8192 | ✅ 已改 |
| 6 | **数据集** | 1024短样本 | 382K真实分布 | ⏳ 等你提供路径 |
| 7 | **超参** | lr=1.25e-5, clip=1.0, warmup=0.01 | lr=5e-6, clip=5.0, warmup=0.0 | ✅ 已改 |

---

## 三、风险评估与缓解措施

### 3.1 显存 OOM 风险：**中等**

**分析**：
- 基线 TP2/EP4/swap 配置下 HBM 峰值 **57.4 GB**（占满 88%）
- 新配置 TP1/EP8/无swap，理论显存占用变化：
  - ✅ **降低项**：
    - 去掉 swap-optimizer → 优化器状态在 GPU 分片（DP=4），每卡约 **12-15 GB**（vs 基线全在 CPU）
    - EP8 vs EP4 → 专家权重切分更细，单卡分片减半
  - ❌ **升高项**：
    - TP1 vs TP2 → 激活值/中间结果不再 TP 切分，翻倍
    - seq 8192 vs 4096 → 激活值翻倍（KV cache、attention 中间结果）
    - LoRA r32 vs r16 → 可训练参数翻倍
    - CP2 Ulysses → 引入 all-to-all 通信缓冲（按 head 切分，额外约 2-4 GB）
  
  **预估峰值**：60-68 GB（可能超 64 GB 上限 5-10%）

**缓解措施**：
1. **优先级 1**：若 OOM，立即加回 `--swap-optimizer --swap-optimizer-times 16`（减半分段数，比基线开销小）
2. **优先级 2**：若仍 OOM，降 MBS 2→1（但会降低吞吐）
3. **优先级 3**：若仍 OOM，考虑改 `recompute-num-layers 1 → 4`（增加重计算层数，牺牲速度换显存）

### 3.2 CP=2 Ulysses 崩溃风险：**中等**

**分析**：
- MindSpeed-LLM 26.0.0 的 Ulysses CP 在**昇腾 910B + 8192 序列**下**未在文档中明确验证**
- 对比文档提到"昇腾 Ulysses CP 是否触发 MC2 类崩溃需要实测"
- Hulk 侧在相同配置下已跑通（CP2 + seq8192），说明**理论上可行**
- 但 Hulk 用的是 Theta 框架，MindSpeed 实现可能有差异

**可能的崩溃场景**：
1. 启动阶段报错：`context-parallel-algo ulysses_cp_algo` 参数不识别或与其他参数冲突
2. 训练中途崩溃：all-to-all 通信触发 HCCL 超时或算子 bug
3. 精度问题：CP 切分导致 loss 异常（NaN / 不收敛）

**缓解措施**：
1. **优先级 1**：小规模测试（10 步），观察 loss 是否正常、是否崩溃
2. **若崩溃**：立即去掉 CP 参数（`--context-parallel-size 1`，删掉 `--context-parallel-algo`），降级为基线的 CP1 配置
3. **替代方案**：若 CP 不可用，考虑用 `--moe-alltoall-overlap-comm`（MoE 通信重叠）部分弥补性能差距

### 3.3 性能预期：单步耗时**可能变慢 20-40%**

**分析**：
- 基线单步 **12.4s**（但有 swap H2D/D2H 开销 + 固定 pad 浪费）
- 新配置变化：
  - ✅ **加速项**：
    - 去掉 swap → 消除 H2D/D2H 传输开销（约 -15% 单步时间）
    - EP8 vs EP4 → MoE all-to-all 通信更分散（专家分布更细），理论上减少通信热点
  - ❌ **减速项**：
    - seq 8192 vs 4096 → 计算量翻倍（attention O(n²)，MLP O(n)）
    - TP1 vs TP2 → 无 TP 内通信重叠，串行度上升
    - CP2 Ulysses → 引入额外的 all-to-all 通信（按 head 切 attention）
    - LoRA r32 vs r16 → 可训练参数翻倍，forward/backward 时间增加
  
  **综合预期**：seq 翻倍主导，单步约 **16-18s**（vs 基线 12.4s，+29-45%）

**但注意**：有效 token 吞吐会**大幅提升**：
- 基线：12.4s/step, 16 样本×88 token = **1408 有效 token** → **113 tokens/s**
- 新配置：假设 17s/step，动态 pack 后约 **7800 有效 token** → **459 tokens/s** (**+305%**)

**结论**：单步变慢是正常的（序列翻倍+计算量翻倍），但**有效 token 吞吐提升 3 倍以上**（pack 消除浪费）

---

## 四、执行步骤（你接下来要做的）

### 步骤 1：等待权重转换完成（预计还需 10-20 分钟）

```bash
# 实时监控转换进度
tail -f /data/sejin/baseline_26/logs/convert_tp1_pp1_ep8_*.log

# 或检查进程是否还在运行
ps -p 1712183

# 转换完成后，确认输出目录
ls -lh /data/sejin/checkpoints/qwen3_omni_30b_a3b_mcore_tp1_pp1_ep8/
```

**预期输出**：
- `iter_0000001/` 目录（MCore checkpoint 格式）
- `latest_checkpointed_iteration.txt` 文件
- 总大小约 **60 GB**（30B 模型 bf16 权重）

### 步骤 2：配置数据路径

你正在按 Hulk 的样本长度分布准备新数据。数据准备完成后：

```bash
# 方式 A：环境变量（推荐，不改脚本）
export HULK_DATA_PATH="/data/sejin/data/YOUR_NEW_DATA_PREFIX"

# 方式 B：直接修改训练脚本（第 18 行）
# 把 DATA_PATH="${HULK_DATA_PATH:-/data/sejin/data/PLACEHOLDER_HULK_DATA}"
# 改为 DATA_PATH="/data/sejin/data/YOUR_NEW_DATA_PREFIX"
```

### 步骤 3：小规模测试（10 步，验证配置）

```bash
cd /data/sejin/baseline_26/scripts
chmod +x train_hulk_aligned.sh

# 小规模测试：10 步，MBS=1，GBS=8
MBS=1 GBS=8 ITERS=10 LOG_FILE=/data/sejin/baseline_26/logs/hulk_smoke_test.log \
  ./train_hulk_aligned.sh
```

**关键观察点**：
- [ ] 是否成功启动（无 CP 参数错误、无权重加载失败）
- [ ] 首步是否 OOM（若 OOM，执行缓解措施 3.1）
- [ ] 是否崩溃于 CP all-to-all（若崩溃，执行缓解措施 3.2）
- [ ] Loss 是否正常下降（首步约 2.0-2.5，10 步后应降到 1.5-1.8 范围）
- [ ] 单步耗时是否在预期范围（15-20s，首步更长因为编译）

### 步骤 4：若测试通过，放大规模

```bash
# 完整训练：60 步，MBS=2，GBS=16（与 Hulk 对齐）
MBS=2 GBS=16 ITERS=60 ./train_hulk_aligned.sh
```

### 步骤 5：采集指标并与 Hulk 对比

训练完成后，从日志提取：
- 稳定段单步耗时（step 10-50 均值±std）
- 有效 token 吞吐（tokens/s，基于实际数据长度分布）
- HBM 峰值（若有监控）
- Loss 收敛曲线

---

## 五、回滚方案（若出现问题）

### 场景 A：OOM（显存不足）

**回滚优先级**：
1. 加回 swap-optimizer（但减半分段数）：
   ```bash
   # 在 OPTIMIZE_ARGS 中加回:
   --swap-optimizer \
   --swap-optimizer-times 16 \
   ```
2. 降低 MBS：`MBS=1 GBS=8`
3. 增加重计算层数：`--recompute-num-layers 4`
4. 最后手段：降序列长度到 6144（但这会偏离 Hulk）

### 场景 B：CP 崩溃（context-parallel 不可用）

**回滚**：
```bash
# 在 MODEL_PARALLEL_ARGS 中:
# 改为: --context-parallel-size 1
# 删掉: --context-parallel-algo ulysses_cp_algo
```

**补偿**：考虑加 MoE 通信优化（若 LoRA 兼容）：
```bash
# 检查是否能用 moe-alltoall-overlap-comm（基线因 LoRA 冲突未用）
# 若能用，加到 MOE_ARGS:
--moe-alltoall-overlap-comm \
```

### 场景 C：单步过慢（>25s）

**诊断**：
- 检查 AI Core 利用率（`npu-smi info`，持续 <5% 说明有瓶颈）
- 检查是否有 swap I/O（若加回了 swap）
- 检查 CP all-to-all 是否成为瓶颈（profiler）

**优化方向**（需实测验证）：
- 调整 recompute 策略（full → selective）
- 调整 CP 实现（若有其他 algo 可选）

---

## 六、成功标准

对标基线**成功**的标志：
1. ✅ 训练稳定运行 60 步无崩溃
2. ✅ Loss 正常收敛（与 Hulk 曲线形状相似）
3. ✅ 单步耗时在合理范围（15-20s，考虑 seq 翻倍）
4. ✅ 有效 token 吞吐显著高于基线（>300 tokens/s）
5. ✅ HBM 占用不超 64 GB（或通过 swap 缓解后不超）

---

## 七、文件清单

| 文件 | 路径 | 状态 |
|---|---|---|
| 权重转换脚本 | `/data/sejin/baseline_26/scripts/convert_weights_tp1_pp1_ep8.sh` | 运行中 |
| Hulk 对齐训练脚本 | `/data/sejin/baseline_26/scripts/train_hulk_aligned.sh` | 就绪 |
| 转换日志 | `/data/sejin/baseline_26/logs/convert_tp1_pp1_ep8_*.log` | 监控中 |
| 对比报告（修正后）| `/data/sejin/baseline_26/reports/HULK_VS_BASELINE_COMPARISON.md` | 已更新 |
| 本执行计划 | `/data/sejin/baseline_26/reports/HULK_ALIGNMENT_EXECUTION_PLAN.md` | 当前文件 |

---

**下一步行动（你）**：
1. 等待权重转换完成（~10-20 分钟）
2. 配置数据路径（`export HULK_DATA_PATH=...`）
3. 运行小规模测试（10 步）
4. 根据测试结果决定是否需要回滚/调整
5. 若通过，运行完整训练（60 步）并采集指标
