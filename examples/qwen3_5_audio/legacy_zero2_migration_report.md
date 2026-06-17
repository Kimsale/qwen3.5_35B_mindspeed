# Qwen3.5-35B-A3B + Whisper-large-v3 Pack/Legacy Zero2 Migration Report

本报告记录 `feat/llm-pad-to-pack-legacy-zero2` 分支里围绕
`Qwen3.5-35B-A3B + Whisper-large-v3 LoRA pack 微调` 做过的全部主要尝试、当前问题与结果。

## 目标

- 保持模型结构、MoE 路由、专家数不变
- 保持 LoRA-only
- 让 pack 版训练在 8 卡 Ascend 910B3 上尽快跑到首个 iteration
- 在 HBM 55-60G 约束下尽量复现/对标原始 zero2 训练入口

## 当前结论

- `from_pretrained()` 的 meta 初始化卡点已经处理掉
- Whisper encoder 权重加载已经处理掉
- LoRA 注入阶段的 meta tensor 报错已经处理掉
- `legacy zero2 + custom_fsdp` 现在推进到了 FSDP buffer 初始化，但在 `ParamAndGradBuffer` 里 OOM
- `nofsdp` 路径虽然能加载模型和 LoRA，但在 optimizer 初始化时仍然遇到 meta 参数
- 截至当前，仍未进入首个 iteration

## 主要改动

### 1. 音频模型加载链路

- 在 [`mindspeed_mm/fsdp/models/qwen3_5_audio/modeling_qwen3_5_audio.py`](/data/sejin/third_party/mindspeed-mm-26.0.0/mindspeed_mm/fsdp/models/qwen3_5_audio/modeling_qwen3_5_audio.py) 添加了 `initialize_weights()` fast-path，只初始化新增的 `audio_projector`
- 在 [`mindspeed_mm/fsdp/models/qwen3_5_audio/whisper_encoder.py`](/data/sejin/third_party/mindspeed-mm-26.0.0/mindspeed_mm/fsdp/models/qwen3_5_audio/whisper_encoder.py) 改成直接从 safetensors 读取 encoder 权重，避免嵌套 `from_pretrained()`
- 在 [`mindspeed_mm/models/transformers_model.py`](/data/sejin/third_party/mindspeed-mm-26.0.0/mindspeed_mm/models/transformers_model.py) 增加了 meta tensor 递归实体化

### 2. LoRA / PEFT 兼容

- 在 [`mindspeed_mm/tasks/finetune/lora/lora_patch.py`](/data/sejin/third_party/mindspeed-mm-26.0.0/mindspeed_mm/tasks/finetune/lora/lora_patch.py) 中：
  - 对 LoRA 目标模块做递归 meta 扫描
  - 对 `LoraLayer.update_layer*()` 做了 safe patch，避免 `.to(weight.device)` 在 meta tensor 上炸掉
  - 对 `LoraModel._replace_module()` 做了 safe patch，避免替换后再次触发设备搬运

### 3. custom_fsdp 修复

- 在 [`/data/sejin/third_party/Megatron-LM-core_v0.12.1/megatron/core/distributed/custom_fsdp/param_and_grad_buffer.py`](/data/sejin/third_party/Megatron-LM-core_v0.12.1/megatron/core/distributed/custom_fsdp/param_and_grad_buffer.py) 补了 `TransformerEngineBaseModule` fallback 定义，避免 TE 未安装时的 `NameError`

## 已做尝试与结果

| 尝试 | 配置 / 入口 | 结果 |
|---|---|---|
| 原始 pack legacy zero2 | `ep8_mbs1_ga4_pack_legacy_zero2_smoke.yaml` | 卡在 `from_pretrained()` 的 meta 初始化 / `initialize_weights()` 链路 |
| nofsdp smoke | `ep8_mbs1_ga4_pack_nofsdp_smoke.yaml` | 模型能加载并完成 LoRA 注入，但 optimizer 初始化时仍遇到 `torch.meta.BFloat16Tensor` |
| fastinit / meta materialize 迭代 | `pack_nofsdp_fastinit*` | 逐步把问题从 `from_pretrained()` 挪到 LoRA / PEFT 注入阶段，再挪到 optimizer 阶段 |
| legacy zero2 + TE fallback | `ep8_mbs1_ga4_pack_legacy_zero2_smoke.yaml` | 通过了模型加载和 LoRA 注入，但在 `ParamAndGradBuffer` 初始化时 OOM |

## 现在遇到的问题

1. `legacy zero2 + custom_fsdp` 在 `ParamAndGradBuffer._init_each_parameter_group_buffers()` 分配大 buffer 时 OOM。
2. `nofsdp` 仍有 meta 参数没在 optimizer 初始化前完全实体化，导致 `Float16OptimizerWithFloat16Params` 拒绝接收 `torch.meta.BFloat16Tensor`。
3. 当前分支仍在和 `custom_fsdp` / `optimizer` 的参数组织方式耦合，说明如果继续走 zero2，需要继续收紧 buffer / 参数搬运逻辑。

## 关键日志

- [`qwen3_5_audio_pack_legacy_zero2_20260617_105405.log`](/data/sejin/baseline_26/logs/qwen3_5_audio_pack_legacy_zero2_20260617_105405.log)
- [`qwen3_5_audio_pack_nofsdp_20260617_111013.log`](/data/sejin/baseline_26/logs/qwen3_5_audio_pack_nofsdp_20260617_111013.log)
- [`qwen3_5_audio_pack_nofsdp_fastinit4_20260617_112610.log`](/data/sejin/baseline_26/logs/qwen3_5_audio_pack_nofsdp_fastinit4_20260617_112610.log)
- [`qwen3_5_audio_legacy_zero2_after_nameerror_fix_20260617_114723.log`](/data/sejin/baseline_26/logs/qwen3_5_audio_legacy_zero2_after_nameerror_fix_20260617_114723.log)
- [`qwen3_5_audio_meta_fix_nofsdp_smoke_20260617_114121.log`](/data/sejin/baseline_26/logs/qwen3_5_audio_meta_fix_nofsdp_smoke_20260617_114121.log)

## 下一步建议

- 继续压 `custom_fsdp` 的 parameter/ggrad buffer 体积，看是否能把 OOM 收掉
- 或者回到 `nofsdp` 线，专门清理 optimizer 初始化前的残余 meta 参数

