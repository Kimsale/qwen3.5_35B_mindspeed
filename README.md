# Pack 格式优化分支 (feat/llm-pad-to-pack-recompute)

> **分支用途**: LLM 序列 pad→pack 改造 + recompute 策略验证  
> **核心成果**: WPS 2111 (+86%)，HBM 40GB (-29%)，历史最高吞吐  
> **完整报告**: [`pack_format_validation_report.md`](pack_format_validation_report.md)

---

## 一、核心成果

| 配置 | WPS | HBM/卡 | 单步耗时 | 状态 | 收益 |
|------|-----|--------|----------|------|------|
| **pack mbs=1 rc_off** | **2111** | 40 GB | 3.6s | ✅ 80步稳定 | vs pad1133: **+86% WPS, -29% HBM** |
| pack mbs=1 rc_on | 1475 | 33 GB | 5.0s | ✅ 80步稳定 | HBM -7GB, WPS -30% |
| pack mbs=2 rc_off | N/A | N/A | N/A | ❌ FSDP2 hang | 跨 rank 序列长度不一致 |
| pack mbs=2 rc_on | N/A | N/A | N/A | ❌ FSDP2 hang | 同上 |

**历史最高吞吐**: Pack mbs=1 rc_off, **WPS 2111**, HBM 40GB, 单步 3.6s

---

## 二、Pack 格式原理

### 核心机制

**Pad 格式**（传统）:
```
Sample 1: [audio1] text1 <pad><pad><pad>     # 长度 800，补到 1536
Sample 2: [audio2] text2 <pad><pad><pad><pad> # 长度 600，补到 1536
Sample 3: [audio3] text3 <pad>                # 长度 1200，补到 1536
--> 3 个样本，3×1536 = 4608 tokens，其中 1872 个 padding (40.6% 浪费)
```

**Pack 格式**（本分支）:
```
Packed: [audio1] text1 [audio2] text2 [audio3] text3  # 总长度 2600
--> 1 个拼接序列，2600 真实 tokens，0 padding (0% 浪费)
```

### 关键技术

1. **多样本拼接**: collator 将多个样本 concat 成单序列
2. **Position IDs 重启**: 每个样本的 position_ids 从 0 开始，边界处重启
3. **FA2 varlen**: position_ids 触发 transformers 原生 FA2 varlen 路径（`npu_flash_attn_varlen_func`）
4. **Cu_seqlens 推导**: 从 position_ids 的跳变点推导 cu_seqlens（累积序列长度）

**代码位置**:
- `mindspeed_mm/fsdp/models/qwen3_5_audio/modeling_qwen3_5_audio.py`: forward 支持 pack 检测
- `mindspeed_mm/fsdp/data/dataloader/packed_collator_wrapper.py`: 拼接逻辑
- `mindspeed_mm/fsdp/data/dataloader/data_collator.py`: 注册 `qwen3vl_packed`

---

## 三、Recompute 策略

### Layer-wise Recompute

**配置**:
```yaml
parallel:
  recompute: true
  recompute_plan:
    apply_modules:
    - model.language_model.layers.{*}  # 只 checkpoint 每一层
```

**原理**:
- PyTorch `checkpoint` 包装每个 Transformer layer 的 forward
- 前向不存激活，backward 时重跑前向再算梯度
- **不包 `language_model` 整体**：避免 checkpoint layer loop 导致中间态显存回升

**实测效果**:
- HBM: 40GB → 33GB (-7GB)
- WPS: 2111 → 1475 (-30%)
- 适用场景：显存受限，吞吐可牺牲

---

## 四、快速开始

### 1. 环境准备

```bash
# 加载 CANN 8.5 环境
source scripts/env_cann85.sh
source /data/sejin/env/venv_cann85/bin/activate

# 验证环境
npu-smi info | head -5
python3 -c "import torch_npu; print(torch_npu.__version__)"  # 应输出 2.6.0.post1+cann85
```

### 2. 配置文件

**Pack rc_off（最优吞吐）**:
```bash
# 188 机器配置
/data/sejin/third_party/mindspeed-mm-26.0.0/examples/qwen3_5_audio/perf_tuning/ep8_pack_188.yaml
```

**Pack rc_on（显存受限）**:
```bash
/data/sejin/third_party/mindspeed-mm-26.0.0/examples/qwen3_5_audio/perf_tuning/ep8_mbs1_ga4_rc_on_pack_188.yaml
```

### 3. 启动训练

```bash
# 必须设置环境变量（数据预处理需要）
export AUDIO_PLACEHOLDER="<|AUDIO|>"

# 启动训练（80步验证）
cd /data/sejin/third_party/mindspeed-mm-26.0.0
bash examples/qwen3_5_audio/scripts/train_qwen35_audio.sh \
  --config examples/qwen3_5_audio/perf_tuning/ep8_pack_188.yaml \
  --max_steps 80
```

### 4. 监控性能

```bash
# 实时监控 NPU
watch -n 1 npu-smi info

# 查看训练日志（WPS/HBM/Loss）
tail -f /data/sejin/output/qwen35_audio_ckpt/train.log
```

---

## 五、配置说明

### Pack 格式核心配置

```yaml
# data.dataloader_param.collate_param 段
collate_param:
  model_name: qwen3vl_packed          # ← 启用 pack collator
  ignore_pad_token_for_loss: true
  # 不要设 pad_to_multiple_of（pack 拼接不需要 padding）

# model 段
attn_implementation: flash_attention_2  # 必需，触发 NPU varlen FA2

# parallel 段（rc_off 配置）
parallel:
  recompute: false                    # 吞吐优先
  expert_parallel_size: 8
  ep_plan:
    apply_modules:
    - model.language_model.layers.{*}.mlp.experts
    dispatcher: fused                 # 或 mc2（待验证）

# parallel 段（rc_on 配置）
parallel:
  recompute: true                     # 显存优先
  recompute_plan:
    apply_modules:
    - model.language_model.layers.{*}
```

---

## 六、已知限制

### 1. mbs=2 FSDP2 hang

**现象**: pack 格式在 `micro_batch_size: 2` 时，训练卡在 FSDP2 lazy init 的 all-gather

**根因**: 各 rank collator 独立拼接，导致跨 rank 序列长度不一致，FSDP2 all-gather 需要统一 shape

**解决方向**: 在 `PackedCollatorWrapper` 加跨 rank 全局长度对齐
- 各 rank 在 collate 前通过 `dist.all_reduce` 同步全局 max_seq_length
- 所有 rank 统一 pad 到该长度（仅跨 rank 对齐，样本内仍保持 pack）

### 2. WPS 统计口径差异

**Pad 格式**: WPS 包含 padding token（虚高）  
**Pack 格式**: WPS 只统计真实 token（准确）

对比时需注意：pack WPS 2111 vs pad WPS 1133 不是单纯 +86%，pad 的 1133 中约 40% 是 padding。

---

## 七、与其他方案对比

| 方案 | WPS | HBM/卡 | 实测状态 | 适用场景 |
|------|-----|--------|----------|---------|
| **Pack rc_off** | 2111 | 40 GB | ✅ 本分支 | 吞吐优先，HBM 充足 |
| Pack rc_on | 1475 | 33 GB | ✅ 本分支 | 显存受限（<40GB） |
| Pad1536 rc_off | 1133 | 56.4 GB | ✅ mc2-perf-eval | Pad 最优稳定配置 |
| Pad + MC2 | 1230-1290 | 55-60 GB | ⏳ 预期 | Pad 通信优化 |
| **Pack + MC2** | 2320+ | ~40 GB | 🎯 待验证 | **理论最优** |

---

## 八、下一步方向

### 🎯 Priority 1: Pack + MC2 组合

在 pack rc_off 基础上启用 MC2 通信-计算重叠:
```yaml
parallel:
  ep_plan:
    dispatcher: mc2  # ← 改为 mc2
```

预期: WPS 2111 → 2320+ (+10%)

### ⏳ Priority 2: Pack mbs>1

解决 FSDP2 hang，解锁更高 batch 吞吐

### 📋 Priority 3: Selective recompute

探索更细粒度 recompute（只 checkpoint MLP 或 attention），平衡 HBM/WPS

---

## 九、参考资料

### 完整报告

- **本分支**: [`pack_format_validation_report.md`](pack_format_validation_report.md)
- **早期版本**: [`reports/qwen35_audio_llm_pack_perf_20260616.md`](reports/qwen35_audio_llm_pack_perf_20260616.md)

### 项目文档（main 分支）

- [项目约束 (CLAUDE.md)](https://github.com/Kimsale/qwen3.5_35B_mindspeed/blob/main/CLAUDE.md)
- [项目状态 (STATUS.md)](https://github.com/Kimsale/qwen3.5_35B_mindspeed/blob/main/STATUS_QWEN35_AUDIO_TRAINING.md)
- [快速开始](QWEN35_AUDIO_TRAINING_GUIDE.md)

### 其他分支

- **main**: https://github.com/Kimsale/qwen3.5_35B_mindspeed
- **mc2-perf-eval**: Pad 格式 38 轮扫描 + MC2 代码接通
- **feat/llm-pad-to-pack**: Pack 格式初版（WPS 2069）

---

**最后更新**: 2026-06-17  
**分支状态**: 活跃开发  
**下次同步**: Pack + MC2 组合验证完成后
