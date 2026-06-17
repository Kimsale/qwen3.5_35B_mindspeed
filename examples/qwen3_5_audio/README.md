# Qwen3.5-35B-A3B + Whisper-large-v3 语音多模态 LoRA SFT

在 **MindSpeed-MM 的 FSDP2 栈**上，把 Whisper-large-v3 的 audio encoder 接到
Qwen3.5-35B-A3B（linear+full 混合注意力 MoE 文本塔）上，做"语音 wav + 文本 txt"
的多模态 LoRA 指令微调。

> 严格遵循项目约束：全程锁定 **CANN 8.5**，禁用 CANN 8.1；不引入 GPU/CUDA 方案。

## 当前状态

这条分支已经把 pack 版迁移、legacy zero2 迁移、recompute 尝试和 meta tensor 清理都跑过一轮。

- `from_pretrained()` 的 meta 初始化卡点已经收掉
- Whisper encoder 的 nested load 已经收掉
- LoRA / PEFT 的 meta tensor 注入问题已经收掉
- `custom_fsdp` 的 `TransformerEngineBaseModule` `NameError` 已经补掉
- 但是 `legacy zero2 + custom_fsdp` 仍在 `ParamAndGradBuffer` 初始化阶段 OOM
- `nofsdp` 仍会在 optimizer 初始化时碰到残余 `torch.meta.BFloat16Tensor`
- 截至目前，训练还没有真正进入首个 iteration

完整尝试记录和日志索引见：

- [`legacy_zero2_migration_report.md`](/data/sejin/third_party/mindspeed-mm-26.0.0/examples/qwen3_5_audio/legacy_zero2_migration_report.md)
- [`HANDOFF_BRIEF.md`](/data/sejin/third_party/mindspeed-mm-26.0.0/examples/qwen3_5_audio/HANDOFF_BRIEF.md)

---

## 一、架构

```
wav ─► Whisper feature_extractor(128-mel) ─► WhisperAudioTower(冻结)
                                                   │ (B, T_mel/2, 1280)
                                                   ▼
                                          AudioProjector(可训)
                                          AvgPool 2x + LayerNorm + MLP
                                                   │ (total_audio_tokens, 2048)
                                                   ▼ masked_scatter 填到 <|audio_pad|>
input_ids ─► embed_tokens ─► text_embeds ──────────┘
                                   │ 融合后的 inputs_embeds
                                   ▼
                       Qwen3.5-MoE 文本塔(LoRA) ─► loss
```

**下采样数量一致性（关键）**：每条音频展开的 `<|audio_pad|>` 数量由数据侧
`mm_plugin.py` 决定：

```
conv_len  = (mel_valid_frames - 1) // 2 + 1   # Whisper 2 层 Conv1d(stride=2)
audio_len = (conv_len          - 2) // 2 + 1   # projector 的 AvgPool1d(k=2,s=2)
```

模型侧 projector 的 `AvgPool1d` 严格复刻第二个 2x，使产出的音频向量数逐条等于
`<|audio_pad|>` 数，`masked_scatter` 才不会报数量不匹配。

**冻结策略**：Whisper encoder 冻结（保留 ASR 表征）、Qwen 文本塔走 LoRA、
projector 全量训练。

---

## 二、新增文件清单

| 文件 | 作用 |
|---|---|
| `mindspeed_mm/fsdp/models/qwen3_5_audio/__init__.py` | 注册 `model_id: qwen3_5_audio` |
| `mindspeed_mm/fsdp/models/qwen3_5_audio/whisper_encoder.py` | Whisper-large-v3 encoder 封装 |
| `mindspeed_mm/fsdp/models/qwen3_5_audio/projector.py` | audio→LLM 维度对齐 + 2x 下采样 |
| `mindspeed_mm/fsdp/models/qwen3_5_audio/modeling_qwen3_5_audio.py` | 顶层融合模型（masked_scatter） |
| `mindspeed_mm/fsdp/models/qwen3_5_audio/convert_weights.py` | 合并 Qwen+Whisper+projector 成 DCP |
| `examples/qwen3_5_audio/qwen3_5_35B_audio_config.yaml` | 训练配置 |
| `examples/qwen3_5_audio/finetune_qwen3_5_35B_audio.sh` | 启动脚本 |
| `examples/qwen3_5_audio/audio_sft_demo.jsonl` | 数据样例 |

---

## 三、操作步骤

### 0. 进入 CANN 8.5 环境
```bash
bash
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH
source /usr/local/Ascend/cann-8.5.0/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
cd /data/sejin/third_party/mindspeed-mm-26.0.0
```

### 1. 准备权重
- Qwen3.5-35B-A3B：`/mnt/shared_data_196/sejin/models/Qwen3.5-35B-A3B`（已就位）
- Whisper-large-v3：`/mnt/shared_data_196/sejin/models/whisper-large-v3`
  - 需含 `config.json` + `model.safetensors`（128-mel 版本，d_model=1280）

### 2. 合并权重为 DCP（必做，否则 whisper/projector 会是未初始化的垃圾值）
```bash
python -m mindspeed_mm.fsdp.models.qwen3_5_audio.convert_weights \
    --qwen_hf_dir    /mnt/shared_data_196/sejin/models/Qwen3.5-35B-A3B \
    --whisper_hf_dir /mnt/shared_data_196/sejin/models/whisper-large-v3 \
    --dcp_dir        /mnt/shared_data_196/sejin/models/Qwen3.5-35B-A3B-audio-dcp \
    --llm_hidden_size 2048
# 产出目录含 release/ + latest_checkpointed_iteration.txt
```

> 原理：FSDP2 用 meta-device 建空壳 + DCP 部分加载（`allow_partial_load`）。
> 不在 DCP 里的参数不会报错但保持未初始化，所以必须把 whisper encoder 和 projector
> 一并烘进 DCP。projector 在转换时按默认 init 初始化（对齐阶段从零训练）。

### 3. 准备数据
JSONL，每行一条样本，音频用 `<|AUDIO|>` 占位（数量须与 `audios` 列表长度一致）：
```json
{"id": "x1", "audios": ["./data/audio/a.wav"], "messages": [
  {"role": "user", "content": "<|AUDIO|>\n请转写这段语音。"},
  {"role": "assistant", "content": "转写结果……"}]}
```
- 音频任意采样率均可，数据管线会用 librosa 重采样到 16kHz 再过 Whisper feature_extractor。
- 支持纯文本样本（去掉 `audios` 字段即可），实现语音/文本混合训练。
- 把 demo 数据和 wav 放到 `./data/` 下，或改 YAML 里的 `dataset` 路径。

### 4. 启动训练
```bash
bash examples/qwen3_5_audio/finetune_qwen3_5_35B_audio.sh
```

### 5. 训练后合并 LoRA / 转回 HF
LoRA 权重以 `lora_only` 模式保存在 `save_path_qwen3_5_audio`。文本塔可用框架的
`Qwen35Converter dcp_to_hf` 转回 HF；audio_tower/projector 为自定义模块，按 DCP
key 前缀（`audio_tower.* / audio_projector.*`）单独提取即可。

---

## 四、显存调优（对齐项目"打满 50-60G"目标）

初始可调档位（910B 单卡 ~64G）：
- `training.micro_batch_size`：1 → 2/4
- `training.gradient_accumulation_steps`：8（控制等效 batch）
- `data.basic_parameters.cutoff_len`：4096 → 更长以容纳长音频 token
- `parallel.recompute`：true（显存吃紧时保留；想换吞吐可关）
- `model.use_grouped_expert_matmul` / `use_triton_gdn`：MoE/线性注意力算子加速，保持 true

逐步加 batch/序列长度，用 `tools.memory_profile` 观察占用，逼近 55-60G。

---

## 五、已知前置依赖
- `peft`（LoRA 注入，框架 `add_lora_to_model` 依赖）
- `librosa`（音频读取重采样）
- Whisper-large-v3 权重（128-mel 版本）

## 五点五、迁移追踪

这部分是当前分支最重要的上下文，方便在别的机器上直接复现判断。

### 已尝试的路径

| 路径 | 结果 |
|---|---|
| 原始 pack + legacy zero2 | 卡在 `from_pretrained()` 的 meta 初始化链路 |
| pack + nofsdp | 能加载模型和 LoRA，但 optimizer 初始化仍遇到 meta 参数 |
| pack + legacy zero2 + `TransformerEngineBaseModule` fallback | 通过模型加载和 LoRA 注入，但在 `ParamAndGradBuffer` 初始化时 OOM |

### 已做的修复

- [`mindspeed_mm/fsdp/models/qwen3_5_audio/modeling_qwen3_5_audio.py`](/data/sejin/third_party/mindspeed-mm-26.0.0/mindspeed_mm/fsdp/models/qwen3_5_audio/modeling_qwen3_5_audio.py)
  - `initialize_weights()` fast-path，只初始化新增 `audio_projector`
- [`mindspeed_mm/fsdp/models/qwen3_5_audio/whisper_encoder.py`](/data/sejin/third_party/mindspeed-mm-26.0.0/mindspeed_mm/fsdp/models/qwen3_5_audio/whisper_encoder.py)
  - 直接从 `model.safetensors` 装 Whisper encoder，避免嵌套 `from_pretrained()`
- [`mindspeed_mm/models/transformers_model.py`](/data/sejin/third_party/mindspeed-mm-26.0.0/mindspeed_mm/models/transformers_model.py)
  - 递归实体化残余 meta tensor
- [`mindspeed_mm/tasks/finetune/lora/lora_patch.py`](/data/sejin/third_party/mindspeed-mm-26.0.0/mindspeed_mm/tasks/finetune/lora/lora_patch.py)
  - 递归扫描 LoRA 目标
  - 规避 PEFT 在 meta tensor 上的 `.to()` 行为
- [`/data/sejin/third_party/Megatron-LM-core_v0.12.1/megatron/core/distributed/custom_fsdp/param_and_grad_buffer.py`](/data/sejin/third_party/Megatron-LM-core_v0.12.1/megatron/core/distributed/custom_fsdp/param_and_grad_buffer.py)
  - 补 `TransformerEngineBaseModule` fallback，避免 TE 未安装时直接 `NameError`

### 结果日志

- [`qwen3_5_audio_legacy_zero2_after_nameerror_fix_20260617_114723.log`](/data/sejin/baseline_26/logs/qwen3_5_audio_legacy_zero2_after_nameerror_fix_20260617_114723.log)
  - 通过了前序加载和 LoRA 注入
  - 在 `ParamAndGradBuffer._init_each_parameter_group_buffers()` 时 OOM
- [`qwen3_5_audio_meta_fix_nofsdp_smoke_20260617_114121.log`](/data/sejin/baseline_26/logs/qwen3_5_audio_meta_fix_nofsdp_smoke_20260617_114121.log)
  - 模型和 LoRA 能走完
  - optimizer 初始化时仍遇到残余 meta 参数

### 结论

当前更现实的方向仍然是继续沿 `legacy zero2` 收 buffer / 参数初始化问题，而不是继续在 `nofsdp` 上硬绕。

---

## 五点六、对框架的两处必要改动（已落地）
Qwen3.5 的 `AutoProcessor` 是 **vision-only**（`Qwen3VLProcessor`，无 `feature_extractor`），
而数据侧 `Qwen2OmniPlugin` 需要它把 wav 转 mel 谱。为此对数据栈做了**向后兼容**的小改：

| 文件 | 改动 |
|---|---|
| `mindspeed_mm/fsdp/data/data_utils/func_utils/model_args.py` | `ProcessorArguments` 新增可选字段 `audio_feature_extractor_path` |
| `mindspeed_mm/fsdp/data/data_utils/func_utils/convert.py` | `load_tokenizer` 在该字段非空且 processor 无 `feature_extractor` 时挂上 Whisper 的 `AutoFeatureExtractor` |

字段不填则行为完全不变，不影响其它模型。本配置已在 YAML 的 `preprocess_parameters`
里指向 whisper 目录。

---

## 七、数学一致性声明
- 未改动 Qwen3.5 MoE 网络结构、专家数、路由规则；仅新增冻结的 audio encoder、
  可训 projector、LoRA 适配器。
- 音频特征通过 `masked_scatter` 注入文本序列的 `<|audio_pad|>` 占位处，未改动
  文本塔的 attention/loss 计算路径（顶层 forward 委托父类原生实现）。
- projector 的下采样与数据侧 token 数公式严格自洽（已用 11 组 mel 长度数值验证
  逐条相等），保证 token 对齐无歧义。
- **位置编码简化**：audio token 占据序列中的顺序位置（text-like），不采用 Omni 的
  audio-mrope 特殊位置。这是 SLAM-LLM 风格的标准做法，数学上等价于把声学 token 当
  普通序列 token，不破坏因果性，对 ASR/语音指令任务足够；后续若要做音视频时间对齐
  再引入 mrope。
