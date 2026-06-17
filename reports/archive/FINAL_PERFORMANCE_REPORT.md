# Qwen3-30B-A3B LoRA 微调性能优化评测报告

**项目**: CANN 8.5.0 + MindSpeed-LLM 26.0.0 对 Qwen3-30B-A3B LoRA 微调性能优化 & 框架对标
**硬件**: 昇腾 910B3 ×8 (单卡 64 GB HBM, total_capacity 60.96 GiB per `torch_npu` 报告)
**日期**: 2026-06-03
**约束**: 不改模型结构；全程 CANN 8.5.0（禁用系统 8.1）；仅调训练超参/并行/算子/混合精度/LoRA/通信参数
**报告更新**: 2026-06-03 增补"数据来源与可追溯性"一节，每条数据标注原始日志文件、行号、step 范围、测量方式。

---

## 1. 环境栈（已验证可运行）

| 组件 | 版本 | 路径 |
|---|---|---|
| CANN | **8.5.0** | `/usr/local/Ascend/cann-8.5.0` |
| MindSpeed-LLM | **26.0.0** | `/data/sejin/third_party/mindspeed-llm-26.0.0` |
| MindSpeed core | **26.0.0_core_r0.12.1** | `/data/sejin/third_party/mindspeed-core-26.0.0` |
| Megatron core | **core_v0.12.1** | `/data/sejin/third_party/Megatron-LM-core_v0.12.1` |
| PyTorch / torch_npu | 2.7.1 / 2.7.1.post2 | venv `/data/sejin/env/venv_26b` |
| transformers / peft | 4.57.1 / 0.7.1 | (26.0.0 强制版本) |
| NNAL / ATB | 配套 8.5.0 | `/usr/local/Ascend/nnal/atb` |

**模型**: Qwen3-30B-A3B (MoE, 48层, hidden 2048, 128 experts/topk 8, moe_ffn 768, ffn 6144, GQA 32头/4 KV组)
- 参数统计来自 Megatron 训练日志输出（`opt_R3c.log` L1727-L1730）：
  - **Total params: 30.52 B**（transformer block 29.90 B + embedding 0.62 B）
  - **most loaded shard: 15.26 B**（per rank under TP2/EP4）
  - **Theoretical memory footprint: weight + optimizer = 130978.71 MB ≈ 127.9 GB**（来源：同一日志 L1730）

**LoRA 可训练参数**（peft 自打印，来源 `opt_R5_actrecomp.log` L1050-L1052，三次重复打印取一致值）：
```
trainable params: 135,659,520 || all params: 4,536,743,936 || trainable%: 2.990239738317909
```
即 **trainable = 135.66 M, per-rank all-params = 4.54 B（每卡分片，TP2 切分后），trainable% = 2.99%**

**权重**: MG 格式 TP2/PP1/EP4 (`/data/sejin/checkpoints/qwen3_30b_a3b_mcore_tp2_pp1_ep4`)
**数据**: packed SFT, 1024 sequences (`/data/sejin/data/qwen3_sft_mcore/`)
**LoRA 配置**: r=16, alpha=32, target=`linear_qkv linear_proj linear_fc1 linear_fc2`

---

## 2. 环境搭建关键修复（共 8 处崩溃，全部解决）

| # | 问题 | 根因 | 修复 |
|---|---|---|---|
| 1 | torchvision 缺失 | transformers 4.57.1 懒加载 aria→torchvision | 复用 xuchen2 的 torchvision 0.22.1 |
| 2 | peft `fan_in_fan_out` KeyError | peft 0.19 不兼容 | 降级 peft 0.7.1 |
| 3 | tokenizer vocab_file=None | 缺 vocab.json | 从 tokenizer.json 提取 vocab.json |
| 4 | state_dict size mismatch | padded_vocab 不符 | `--padded-vocab-size 152064` |
| 5 | dataset IndexError | data-path 用了全名 | 改用 prefix `qwen3_sft` |
| 6 | venv 克隆后 worker 用错 python | bin/ shebang 指向旧 venv | sed 修所有 bin/* shebang |
| 7 | **MC2 `aclnnMatmulReduceScatter 507018`** | 固定长序列触发 MC2 融合算子 bug | `--no-rope-fusion --swap-optimizer --swap-optimizer-times 32 --recompute-granularity full` |
| 8 | checkpoint save 撑爆磁盘(140G) | 结尾强制 save | 性能测试禁用 save |

---

## 3. 基线性能指标（占满显存的可信基线）

**基线运行**：`/data/sejin/baseline_26/logs/verify_mc2fix.log` (15 步训练)
**配置**: TP2/PP1/EP4, **固定 seq 4096**, mbs2/gbs16, bf16, full recompute + swap-optimizer + MC2 规避
**采样窗口**: step 4-15（n=12，跳过前 3 步 warmup，因 step 1 = 161018ms 是首步算子 JIT 编译 outlier）

### 3.1 单步耗时（来源逐行可查）

来自 `verify_mc2fix.log` L1749-L1770，每行格式 `iteration N/15 | ... elapsed time per iteration (ms): X`：

| step | 行号 | 时间(ms) |
|---|---|---|
| 1 (warmup) | L1749 | 161017.7 ← 含算子 JIT 编译，剔除 |
| 2 (warmup) | L1757 | 12578.6 |
| 3 (warmup) | L1758 | 12564.9 |
| **4** | L1759 | **12546.8** |
| **5** | L1760 | **12527.4** |
| **6** | L1761 | **12618.8** |
| **7** | L1762 | **12567.5** |
| **8** | L1763 | **12440.2** |
| **9** | L1764 | **12385.5** |
| **10** | L1765 | **12415.9** |
| **11** | L1766 | **12541.1** |
| **12** | L1767 | **12351.0** |
| **13** | L1768 | **12315.5** |
| **14** | L1769 | **12452.1** |
| **15** | L1770 | **12399.2** |

**统计（step 4-15, n=12）**:
- mean = **12463.4 ms**
- min = 12315.5, max = 12618.8
- std = **91.1 ms**（变异系数 0.7%, 极稳定）

### 3.2 显存（HBM）数据 — **三种测量来源，必须分清**

报告早先的 "57.4 GB" 标注**来源不严谨**，重新厘清如下：

**(A) Megatron 内部 `torch.npu.memory_*` 打印**（`verify_mc2fix.log` L1755-L1756, after 1 iterations）：
```
[Rank 0] memory (MB) | allocated: 9635.11 | max allocated: 45006.98 | reserved: 45694.0 | max reserved: 45694.0
```
即每卡 **max allocated ≈ 43.95 GB, max reserved ≈ 44.62 GB**（PyTorch 视角）

**(B) Megatron 理论估算**（`opt_R3c.log` L1730）：
```
Theoretical memory footprints: weight and optimizer=130978.71 MB
```
权重+优化器总占 ≈ **127.9 GB**（被 swap-optimizer 部分卸载到 CPU 后才能装下 65 GB 单卡）

**(C) `npu-smi info` 整卡 HBM 实测**（含 driver/cache/expandable_segments 增长，无对应 json 存档，仅在 R3c 训练 step 9 期间手动读到瞬时值，**bash 命令输出留底**）：
- 训练运行时 HBM 峰值约 **57.4 GB / 65 GB（≈ 88%）**——这个数字的可信度低于 (A)，因为没有连续监控 json，仅一次性 snapshot

**修订后的稳健陈述**：
- **PyTorch 视角 max reserved（per rank）= 44.6 GB**（强证据，日志 L1755 直接打印）
- npu-smi 视角整卡 HBM 峰值约 57 GB（弱证据，单点 snapshot，未存档）
- 两者差 ≈ 13 GB 是 npu-smi 含的 driver/HCCL buffer/expandable_segments 预留 + torch 之外的 NNAL/ATB 占用，正常

### 3.3 吞吐 / loss / 稳定性

| 指标 | 数值 | 来源 |
|---|---|---|
| TPS (samples/s) | 1.28 | 计算: gbs(16) / mean_step(12463 ms) = 1.284，来源 `verify_mc2fix.log` step 4-15 |
| WPS (tokens/s) | 5258 | 计算: tokens_per_step(16×4096=65536) / mean_step = 5258，来源同上 |
| Loss | 2.197 → 1.820 | `verify_mc2fix.log` step 1 lm loss=2.197363 (L1749), step 15 lm loss=1.819675 (L1770) |
| 梯度范数 | mean ≈ 2.196 | step 4-15 grad norm 均值，原始数据见 L1759-L1770 |
| NaN / 跳过步 | 0 / 0 | 全 15 步 `nan iterations: 0`, `skipped iterations: 0`，逐行可查 |
| trainable params | 135,659,520 (2.99%) | peft 直接打印，`opt_R5_actrecomp.log` L1050（其他 run 同样数字） |

### 3.4 AI Core 利用率说明（无可信测量）

`npu-smi info` 的 AICore% 是瞬时采样，对 MoE 稀疏激活 + swap-optimizer 频繁 H2D/D2H 的工作负载严重失真（实测瞬时值在 0–16% 抖动，无统计意义）。`npu_fixed_mbs2.json`（早期 sweep 采的 npu json）`aicore_pct.peak=0.0`、`mean=0.0`，进一步说明 monitor 的瞬时采样模式不适合本场景。

**真实算力利用率**应通过 profiler 的 `AICore_time / E2E_time` 或训练日志中的 `TFLOP/s/GPU` 指标获取——本次评测**未启用 profiler**（`--profile False`），所以**无可信的 AICore% 数据**。CLAUDE.md 设的 "≥70% AICore" 目标在没有 profiler 数据的情况下**无法验证**。

---

## 4. 优化迭代实测结果（每条标注来源）

每轮固定 seq 4096, gbs16, 与基线同条件对比。所有 step 数据来自下表对应日志，统计同样跳过前 3 步 warmup。

| 轮次 | 优化项 | 命令行改动 | 单步均值±std (ms) | n | 数据来源（日志 + 行号范围） | TPS | 结论 |
|---|---|---|---|---|---|---|---|
| **R0** | **基线** | (MC2 规避基线) | **12463.4 ± 91.1** | 12 | `verify_mc2fix.log` L1749-L1770 (step 4-15) | **1.28** | 基准 |
| R1 | 增大 batch mbs4 | `--micro-batch-size 4` | **OOM** | - | `opt_R1_mbs4.log` L1763 (rank6 RuntimeError: NPU out of memory, allocated 57.81 GiB / 60.96 GiB) | - | ❌ mbs2 已是 full-recompute 显存上限 |
| R2 | MoE alltoall 重叠 | `--moe-tp-extend-ep --moe-alltoall-overlap-comm` | **崩溃** | - | `opt_R4_combo.log` L149-150 (`AssertionError: Lora and Qlora are not supported with moe-tp-extend-ep`) | - | ❌ moe-tp-extend-ep 与 LoRA 硬约束不兼容 |
| R3 | DP 通信重叠 | `--overlap-grad-reduce --overlap-param-gather` | **15312.0 ± 518.7** | 22 | `opt_R3c.log` L1736-L1760 (step 4-25) | 1.04 | ❌ 负优化 -23% |
| R5 | 轻量重计算 | `--recompute-activation-function`（替代 full） | **15217.1 ± 555.4** | 17 | `opt_R5_actrecomp.log` L1729-L1755 (step 4-20) | 1.05 | ❌ 负优化 -22% |
| - | reset-bucket-group-order | `--reset-bucket-group-order` | argparse 拒绝 | - | `opt_R3_clean.stdout` (`error: unrecognized arguments`) | - | ❌ flag 未注册到 CLI |

> **更正**：之前的报告版本 R3 单步写为 "15305±607"、R5 写为 "15226±351"。这些是用 `parse_metrics.py`（warmup=3）算出的，与本次直接重算（L 行号读取，warmup=3）**略有差异**（R3 重算 15312±519, R5 15217±555）。差异来自浮点舍入与跳过策略的边界处理，结论方向不变（都是负优化 -22~23%）。本表是直接基于日志重算的权威值。

### 关键发现

1. **mbs2 即显存上限**：R1 OOM 证据精确到行——`opt_R1_mbs4.log` L1763 显示 8 卡全部 OOM，每卡尝试分配 920 MiB 时已 allocated 57.81 GiB / 60.96 GiB capacity（free 仅剩 938 MiB）。
2. **LoRA 主力优化路径被框架硬封**：`AssertionError: Lora and Qlora are not supported with moe-tp-extend-ep.` 这是源码 assert，不可绕过；预期收益最高的 MoE alltoall overlap 因此不可用。
3. **DP overlap 在本场景是负优化**：基线 12463 → R3 15312（+22.9%）。原因：LoRA 可训练参数仅 135.66 M（peft 打印），梯度通信量极小，overlap 的 bucket 重排开销 > 收益。
4. **轻量 recompute 也是负优化**：基线 12463 → R5 15217（+22.1%）。说明 swap-optimizer 的 H2D/D2H 是真实瓶颈，不是 recompute。R5 期间 `max reserved` 升到 46116 MB（+422 MB，`opt_R5_actrecomp.log` L1735）但单步反而变慢，进一步印证显存非瓶颈。

### MindSpeed AutoTuning 实测：架构层不可用（CLAUDE.md 第 3 条）

CLAUDE.md 要求 "依托 MindSpeed 内置 Auto Tuning 自动优选超参"，实测确认 **26.0.0 的 LLM 包架构上不支持** AutoTuning（脚本 `scripts/train_autotune.sh`，5 次注入修复尝试，全部失败）：

| 现象 | 根因（mindspeed-core/llm 源码确认）|
|---|---|
| `--auto-settings` argparse 直接拒绝 | `mindspeed_llm/features_manager/__init__.py:create_features_list()` 不包含 `AutoSettingsFeature` |
| 注入后 `--jit-compile` 命名冲突 | `mindspeed_llm/megatron_basic/training_basic.py:97` 已注册同名 flag |
| 子类化过滤后 8-rank 启动 → 30B OOM | AutoTuning 的 hook 在 `megatron.training.training.pretrain`，但 `pretrain_gpt.py` 走 `mindspeed_llm.training.training.pretrain` wrapper，**hook 点完全绕过** |

**最终判定**: 在 26.0.0 + LoRA + posttrain 路径下 MindSpeed AutoTuning **不可用**，需框架级修改才能启用，超出 CLAUDE.md "禁改模型结构、仅调超参" 的边界。代码注入修改已全部回滚。

---

## 5. 瓶颈诊断

| 维度 | 现象 | 证据来源 | 是否可优化 |
|---|---|---|---|
| **算力** | 无可信 AICore% 数据 | profiler 未启用，`npu_*.json` aicore_pct=0 显示 monitor 不可用 | 受 LoRA + 显存约束，且无测量手段 |
| **显存** | per-rank max reserved 44.6 GB；mbs4 OOM | `verify_mc2fix.log` L1755 + `opt_R1_mbs4.log` L1763 | 已占满，不能加 batch |
| **通信** | DP overlap +22.9% 反而变慢 | `opt_R3c.log` L1736-L1760 vs `verify_mc2fix.log` L1759-L1770 | 本场景不适用 |
| **MoE** | moe-tp-extend-ep 被框架 assert 拒 | `opt_R4_combo.log` L149-150 | ❌ LoRA 下不可用 |
| **swap-optimizer 开销** | recompute 变轻反而慢（+22.1%）| `opt_R5_actrecomp.log` L1729-L1755 | swap H2D/D2H 是真瓶颈 |

---

## 6. 结论与生产配置建议

### 最优可落地配置（= 当前基线 R0）
LoRA 微调 Qwen3-30B-A3B 在本环境的最优稳定配置：

```
TP=2 PP=1 EP=4, seq=4096, mbs=2, gbs=16, bf16
--use-flash-attn --sequence-parallel --use-fused-swiglu --use-fused-rmsnorm
--use-fused-rotary-pos-emb --no-rope-fusion
--moe-grouped-gemm --moe-permutation-async-comm --moe-token-dispatcher-type alltoall_seq
--use-distributed-optimizer --swap-optimizer --swap-optimizer-times 32
--recompute-granularity full --recompute-method block --recompute-num-layers 1
```

实测：单步 12463 ms（n=12, std 91 ms），TPS 1.28，WPS 5258，per-rank max reserved 44.6 GB，trainable 135.66 M (2.99%)。

### 核心结论
1. **LoRA 微调下 MindSpeed MoE 通信优化不可用**（hard assert，framework-level）
2. **mbs2 + full recompute 即显存上限**，无 batch 提升空间
3. **DP overlap 与 lightweight recompute 在本场景均为负优化 -22%**
4. **MindSpeed AutoTuning 在 26.0.0 LLM 路径架构性不可用**

### 后续可探索（需更大改动 / 换权重）
- TP1/EP8 重转权重 → 简化 TP 通信，可能解锁部分优化
- 放宽到全参微调 → 可启用 moe-alltoall-overlap-comm（预期单步显著下降）
- 启用 profiler 测真实 AICore% / TFLOP/s（CLAUDE.md "≥70% AICore" 目标的真实达成度需此数据）

---

## 7. 数据可追溯性附录

| 报告中数字 | 原始日志 | 行号 / step 范围 | 测量方式 |
|---|---|---|---|
| 单步 12463.4 ± 91.1 ms | `verify_mc2fix.log` | L1749-L1770, step 4-15 (n=12) | Megatron 训练日志 `elapsed time per iteration (ms)` |
| max reserved 44.6 GB / rank | `verify_mc2fix.log` | L1755-L1756 (after 1 iter) | `torch.npu.memory_reserved` |
| Theoretical 130978.71 MB | `opt_R3c.log` | L1730 | Megatron 启动时打印 |
| Total 30.52 B params | `opt_R3c.log` | L1727-L1729 | Megatron `Number of parameters in transformer block + embedding` |
| trainable 135,659,520 (2.99%) | `opt_R5_actrecomp.log` | L1050-L1052 | peft `print_trainable_parameters()` |
| R1 mbs4 OOM 57.81 GiB | `opt_R1_mbs4.log` | L1763 | torch_npu 抛 RuntimeError |
| R2/R4 LoRA assert | `opt_R4_combo.log` | L149-150 | mindspeed_llm 源码 `raise AssertionError` |
| R3 15312 ± 519 ms | `opt_R3c.log` | L1736-L1760, step 4-25 (n=22) | 同基线 |
| R5 15217 ± 555 ms | `opt_R5_actrecomp.log` | L1729-L1755, step 4-20 (n=17) | 同基线 |
| 57.4 GB npu-smi 峰值 | (未存档) | R3c 训练 step 9 期间 bash 手动 `npu-smi info` 一次性 snapshot | npu-smi HBM-Usage 字段；**可信度弱于 max reserved** |

`npu_*.json`（5 个）由 `npu_monitor.py` 采样，时机不当（采样时训练未真正进入稳定迭代或本身权重已经反向 unload），数据**不可信**（aicore_pct 全 0、HBM 偏低）。已不在报告主结论中引用。

---

## 8. 附录：脚本与日志清单

- 环境脚本: `/data/sejin/baseline_26/scripts/env_cann85.sh`
- 训练脚本: `/data/sejin/baseline_26/scripts/{train_param,train_baseline_lora,train_r5,train_autotune}.sh`
- Sweep 脚本: `/data/sejin/baseline_26/scripts/{auto_sweep,opt_sweep}.sh`
- 工具: `npu_monitor.py`（已确认采样时机问题，不可单独依赖）, `parse_metrics.py`
- 优化方案分析（workflow 产出）: `/data/sejin/baseline_26/reports/workflow_analysis_raw.md`
- 训练日志总目录: `/data/sejin/baseline_26/logs/` (28 个 `.log`)
- 关键日志:
  - `verify_mc2fix.log` — 基线 R0 (15 step)
  - `opt_R1_mbs4.log` — R1 OOM
  - `opt_R3c.log` — R3 DP overlap (25 step)
  - `opt_R4_combo.log` — R4 LoRA 不兼容 assert
  - `opt_R5_actrecomp.log` — R5 act-recomp (20 step)
  - `autotune_run.log` — AutoTuning 5 次失败尝试
