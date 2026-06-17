# Pack Format Implementation - Quick Start

## 改动总结

### 1. 模型层（`mindspeed_mm/fsdp/models/qwen3_5_audio/modeling_qwen3_5_audio.py`）
- **新增 `cu_seqlens` 参数**：forward 自动检测 pack/pad 格式
- **Pack 格式处理**：
  - 给 `inputs_embeds`/`position_ids`/`labels` 加 batch dim `(1, total_len, ...)`
  - `attention_mask` 置 None（触发原生 FA2 varlen 路径）
  - 音频 token 替换按 `cu_seqlens` 边界逐样本处理（`_replace_audio_tokens_packed`）
- **完全向后兼容**：Pad 格式逻辑保持不变，Whisper encoder/projector 完全不动

### 2. 数据层
- **新增 `packed_collator_wrapper.py`**：
  - 包装原 `MultiModalDataCollatorForSeq2Seq`
  - 先走 pad 逻辑，再转 pack：去 padding、拼接、生成 `cu_seqlens`/`position_ids`
- **注册新 collator**（`data_collator.py`）：
  - `DataCollatorForQwen2vlPacked`
  - 注册到 `DATA_COLLATOR["qwen3vl_packed"]`

### 3. 配置文件
- **新增 `ep8_mbs1_ga4_rc_off_pack.yaml`**（基于 pad1408_nosync 复制）
- **关键改动**：
  - `collate_param.model_name: qwen3vl_packed`
  - `attn_implementation: flash_attention_2`（必需，触发 varlen）
  - 删掉 `pad_to_multiple_of: 1408`
  - `cache_dir` 改为独立路径避免冲突

### 4. Recompute 方案
- **新增 `ep8_mbs1_ga4_rc_on_pack.yaml`**
- **关键改动**：
  - `parallel.recompute: true`
  - `parallel.recompute_plan.apply_modules: [model.language_model.layers.{*}]`
  - 只把重计算下沉到单层 Transformer，不把整个 `language_model` 外包进 checkpoint
  - 目标是尽量对标 `--recompute-granularity full` 的显存收益，但避免把 layer loop 整体包住导致显存回升

## 运行

### Smoke Test（10 steps 快速验证）

```bash
cd /data/sejin/third_party/mindspeed-mm-26.0.0

# Pack 格式
bash examples/qwen3_5_audio/smoke_test_pack.sh pack

# Pack + recompute
bash examples/qwen3_5_audio/smoke_test_pack.sh pack-rc

# Pad 格式（对比基线）
bash examples/qwen3_5_audio/smoke_test_pack.sh pad
```

日志在 `logs/train_<timestamp>_smoke_{pack|pad}.log`。

### 完整训练

```bash
cd /data/sejin/third_party/mindspeed-mm-26.0.0

# 使用 pack 格式配置
torchrun --nproc_per_node 8 \
    mindspeed_mm/fsdp/train/trainer.py \
    examples/qwen3_5_audio/perf_tuning/ep8_mbs1_ga4_rc_off_pack.yaml

# 使用 pack + recompute 配置
torchrun --nproc_per_node 8 \
    mindspeed_mm/fsdp/train/trainer.py \
    examples/qwen3_5_audio/perf_tuning/ep8_mbs1_ga4_rc_on_pack.yaml
```

## 验证检查点

### 数值一致性
对比 pack vs pad 前 10 steps 的 loss，期望差异 < 1e-3（浮点误差范围）。

### 性能指标
- **显存**：pack 应显著降低 HBM 占用（消除 padding 浪费）
- **吞吐**：期望 pack ≥ pad（取决于序列长度分布）
- **AICore 利用率**：观察 npu-smi 的 AICore 占用率

### 常见问题排查

1. **`cu_seqlens` 未传递到模型**
   - 检查 collator 是否正确注册：YAML 里 `model_name: qwen3vl_packed`
   - 检查 `packed_collator_wrapper.py` 的 `_pad_to_pack` 返回了 `cu_seqlens`

2. **Attention mask 相关报错**
   - 确认 YAML 里 `attn_implementation: flash_attention_2`（不是 `sdpa`）
   - Pack 格式下 `attention_mask` 必须是 None

3. **音频 token 数量不匹配**
   - 检查 `_replace_audio_tokens_packed` 的边界索引
   - 验证 projector 输出的 token 数与 `<|audio_pad|>` 占位数一致

4. **Loss 为 NaN**
   - 检查 position_ids 是否每个样本从 0 重启（pack 格式的关键信号）
   - 验证 `cu_seqlens` 的累积长度正确

## 改动文件列表

```
mindspeed_mm/fsdp/models/qwen3_5_audio/modeling_qwen3_5_audio.py
mindspeed_mm/fsdp/data/dataloader/packed_collator_wrapper.py          (新增)
mindspeed_mm/fsdp/data/dataloader/data_collator.py
examples/qwen3_5_audio/perf_tuning/ep8_mbs1_ga4_rc_off_pack.yaml      (新增)
examples/qwen3_5_audio/smoke_test_pack.sh                              (新增)
```

## 技术细节

### 为什么必须用 FlashAttention-2？

Pack 格式通过 **position_ids 重启**（每个样本从 0 开始）触发 transformers 的原生 varlen packing 检测（`_is_packed_sequence`）。检测通过后，框架自动路由到 `npu_flash_attn_varlen_func`，传入由 position_ids 推导的 `cu_seqlens`。这是 **O(Σ seq_i²)** 的真正 varlen 注意力，无需显式 mask。

如果用 SDPA，transformers 不支持 varlen packing，会回退到构造稠密 block-diagonal mask——这是 **O(total_len²)** 的，在 batch_tokens=100k 时单个 mask 约 20GB，瞬间 OOM。

### 数学等价性

Pack 格式的 loss 等价于逐样本计算的 pad 格式，依据：
1. **样本隔离**：varlen FA2 通过 cu_seqlens 保证样本间无跨界注意力
2. **独立位置编码**：position_ids 每样本从 0 重启 → 正确的 RoPE
3. **边界 loss 屏蔽**：HF 内部 causal-shift loss 在样本边界处的污染被 `-100` 标签过滤（每个样本首 token 必为 `-100`，因为 prompt/audio tokens 被 mask）

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
