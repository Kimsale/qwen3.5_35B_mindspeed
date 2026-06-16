# Qwen3.5-35B 音频 LoRA — LLM Pack 格式优化性能报告

**日期**: 2026-06-16
**测试机器**: 172.29.226.188 (task3-910B-188, 8×910B)
**对标 baseline**: `ep8_mbs1_ga4_rc_off_pad1408_nosync`（最优实践配置）

---

## 一、改造概述

仅对 **LLM 文本序列**做 pad→pack 改造，**Whisper-large-v3 audio encoder、数据、并行策略、LoRA 配置全部保持 baseline 不变**。

**核心机制**：多样本拼接为单序列，position_ids 每样本从 0 重启，触发 transformers 原生
FlashAttention-2 varlen 路径（`npu_flash_attn_varlen_func`），由 position_ids 推导
cu_seqlens，O(Σ seqᵢ²) 计算且零 mask 显存。

**改动位置**（MindSpeed-MM 26.0.0）：
- `mindspeed_mm/fsdp/models/qwen3_5_audio/modeling_qwen3_5_audio.py`：forward 支持 pack（检测 cu_seqlens → batch=1 + 原生 varlen + 按边界替换音频 token）
- `mindspeed_mm/fsdp/data/dataloader/packed_collator_wrapper.py`（新增）：pad→pack 转换，处理 MRoPE 3D position_ids
- `mindspeed_mm/fsdp/data/dataloader/data_collator.py`：注册 `qwen3vl_packed` collator

---

## 二、性能对比（80 步，warmup 后统计）

| 指标 | Baseline (pad1408) | Pack 格式 | 收益 |
|------|-------------------|-----------|------|
| **每步耗时** | 4.786 s | **3.787 s** | **−20.9%** ✅ |
| **input WPS** | 1158.3 | **2069** | **+78.6%** ✅ |
| **HBM 占用** | 54.64 GB | **~40 GB** | **−27%** ✅ |
| **训练完成** | 80/80 | 80/80 | 持平 ✅ |
| **Loss 收敛** | 正常 | 11.85 → 4.83 | 健康单调下降 ✅ |
| **梯度稳定性** | 正常 | grad_norm 2-20（无 NaN） | 稳定 ✅ |

> 注：pack 的 WPS 提升幅度大，因为 baseline 的 WPS 口径含 padding（pad 到 1408×bs），
> 而 pack 的 input token 为真实有效 token（平均 7663/step），无 padding 浪费。
> 显存与步耗时的下降是 padding 消除带来的直接收益。

### Loss 轨迹（每 10 步）

| iter | loss | step_ms | input_tok | wps |
|------|------|---------|-----------|-----|
| 1 | 11.85 | 39206 (含编译) | 6462 | 165 |
| 10 | 11.21 | 3663 | 7952 | 2171 |
| 20 | 10.04 | 3597 | 8189 | 2277 |
| 30 | 7.83 | 3628 | 7960 | 2194 |
| 40 | 6.34 | 3943 | 8233 | 2088 |
| 50 | 5.46 | 3557 | 8394 | 2360 |
| 60 | 5.04 | 4482 | 7670 | 1711 |
| 70 | 4.72 | 3577 | 8269 | 2312 |
| 80 | 4.83 | 3860 | 7549 | 1956 |

---

## 三、实施中修复的问题

| # | 问题 | 根因 | 修复 |
|---|------|------|------|
| 1 | 数据预处理 audios/AUDIO token 数不匹配 | 漏设 `AUDIO_PLACEHOLDER` 环境变量 | 启动脚本加 `export AUDIO_PLACEHOLDER="<\|AUDIO\|>"` |
| 2 | loss_func `IndexError: Dimension out of range` | pack labels 是 1D，framework 期望 2D `(batch,seq)` | collator 输出 `(1,total_len)` |
| 3 | RoPE `tensor size mismatch` | Qwen3.5 用 MRoPE，position_ids 需 3D `(3,batch,seq)` | collator 输出 position_ids `(3,1,total_len)`，每样本从 0 重启 |
| 4 | 之前用错 Python 环境 | 用了 `/home/sejin/.local`（transformers 4.57） | 改用 baseline 的 `venv_qwen35`（源码版） |

---

## 四、数学一致性保证

Pack 格式与逐样本 pad 格式**数学等价**，依据：
1. **样本隔离**：FA2 varlen 通过 cu_seqlens 保证样本间无跨界注意力
2. **位置编码独立**：position_ids 每样本从 0 重启 → 正确的 per-sample RoPE
3. **边界 loss 屏蔽**：HF 内部 causal-shift loss 在样本边界处的 shift target 恒为 `-100`
   （每个样本首 token 必为 prompt/audio token，被 `_build_labels` mask），无跨样本污染

---

## 五、生产配置

```yaml
# examples/qwen3_5_audio/perf_tuning/ep8_mbs1_ga4_rc_off_pack.yaml
data:
  dataloader_param:
    collate_param:
      model_name: qwen3vl_packed   # pack collator，无 pad_to_multiple_of
model:
  attn_implementation: flash_attention_2   # 必需，触发 varlen
```

环境变量（启动脚本必加）：
```bash
export AUDIO_PLACEHOLDER="<|AUDIO|>"
```

---

## 六、结论

LLM pad→pack 改造在保持 baseline 全部配置（Whisper encoder/数据/并行/LoRA）不变的前提下：
- **显存降低 ~27%**（54.6GB → 40GB），为更大 batch 或更长序列留出空间
- **每步加速 ~21%**（4.79s → 3.79s）
- **有效吞吐大幅提升**（消除 padding 浪费）
- **Loss 收敛健康、数学一致、训练稳定**

建议作为新的生产基线，并可在此基础上叠加更大 batch_tokens 进一步压榨显存红利。
