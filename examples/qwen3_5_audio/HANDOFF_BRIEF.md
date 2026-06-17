# Qwen3.5-35B-A3B + Whisper-large-v3 迁移接手简报

这个分支目前的目标是把 `Qwen3.5-35B-A3B + Whisper-large-v3` 的 LoRA 语音微调，
从 pack 版继续往 `legacy zero2` / `nofsdp` 两条路径推进，并尽量对标
`--recompute-granularity full --method uniform` 的显存行为。

## 你需要知道的结论

- `from_pretrained()` 的 meta 初始化卡点已处理
- Whisper encoder 的 nested load 已处理
- LoRA / PEFT 的 meta tensor 注入问题已处理
- `custom_fsdp` 的 `TransformerEngineBaseModule` `NameError` 已处理
- 但 `legacy zero2 + custom_fsdp` 仍会在 `ParamAndGradBuffer` 初始化时 OOM
- `nofsdp` 仍会在 optimizer 初始化时碰到残余 `torch.meta.BFloat16Tensor`
- 目前还没真正进入首个 iteration

## 当前分支和仓库

- 主仓库：`/data/sejin/third_party/mindspeed-mm-26.0.0`
- 主分支：`feat/llm-pad-to-pack-legacy-zero2`
- Megatron 依赖仓库：`/data/sejin/third_party/Megatron-LM-core_v0.12.1`
- Megatron 提交：`db9188b Add TransformerEngineBaseModule fallback`

## 最小复现步骤

1. 进入 CANN 8.5 环境
```bash
source /usr/local/Ascend/cann-8.5.0/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
cd /data/sejin/third_party/mindspeed-mm-26.0.0
```

2. 确认权重目录
- Qwen3.5-35B-A3B
- Whisper-large-v3（128-mel 版本）

3. 先看 smoke 配置
- `examples/qwen3_5_audio/perf_tuning/ep8_mbs1_ga4_pack_legacy_zero2_smoke.yaml`
- `examples/qwen3_5_audio/perf_tuning/ep8_mbs1_ga4_pack_nofsdp_smoke.yaml`

4. 跑对应启动脚本
```bash
bash examples/qwen3_5_audio/finetune_qwen3_5_audio_pack_legacy_zero2.sh
```

## 现在分别卡在哪里

### legacy zero2

- 能走过模型加载和 LoRA 注入
- 卡在 `ParamAndGradBuffer._init_each_parameter_group_buffers()`
- 表现是 buffer 分配 OOM

### nofsdp

- 能走过模型加载和 LoRA 注入
- 卡在 optimizer 初始化
- 表现是还有残余 meta 参数没被实体化

## 关键日志

- `qwen3_5_audio_legacy_zero2_after_nameerror_fix_20260617_114723.log`
- `qwen3_5_audio_meta_fix_nofsdp_smoke_20260617_114121.log`

## 下一步建议

1. 继续压 `custom_fsdp` 的 buffer 体积，确认 OOM 的最小触发点
2. 或者回到 `nofsdp`，专门清掉 optimizer 初始化前的残余 meta tensor
3. 如果要做性能对标，优先沿现有 pack 版配置，把变化控制在一个变量上
