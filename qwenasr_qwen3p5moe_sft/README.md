# Qwen3-ASR + Qwen3.5-MoE 语音翻译训练 (LMDB版)

## 项目概述

基于 Qwen3-ASR-1.7B (音频编码器) + Qwen3.5-35B-A3B MoE (语言模型) 的语音翻译训练流程。使用 Expert Parallelism (EP=8) 在华为昇腾 NPU 上进行分布式训练。

### 模型架构

```
音频 PCM → Feature Extractor (mel) → Audio Encoder (Qwen3OmniMoe) → Audio Embeddings
                                                                          ↓
Prompt + Audio Tokens → LLM Embedding → 替换 audio_pad 为 Audio Embeddings → Qwen3.5 MoE LLM → 译文
```

- **Audio Encoder**: Qwen3OmniMoeAudioEncoder, 3层CNN下采样 + 24层Transformer, 输出2048维
- **LLM**: Qwen3.5-35B-A3B MoE, 256个expert, EP=8 每卡32个expert

### 任务格式

中→英翻译:
```
<|im_start|>user
参考识别内容（可能为空），识别:【识别文本】，将音频里面中文翻译成英文||热词1->翻译1||热词2->翻译2||
<|audio_start|><|audio_pad|>...<|audio_end|>
<|im_end|>
<|im_start|>assistant
<think>

</think>

翻译结果<|im_end|>
```

- 识别文本: 当 ASR acc >= 阈值时填入，否则留空
- 热词: 从 LMDB 中读取，格式 `src->tgt`，无热词时省略
- Label: 仅译文部分计算loss

## 相比原版的改动

### 1. LMDB数据加载 (替代JSONL+WAV文件)

**文件**: `train_ep.py`

原版从 JSONL 逐行读取 `audio_path` 加载 WAV 文件。改为从 LMDB 读取所有数据:

| LMDB字段 | 内容 | 格式 |
|----------|------|------|
| wav | PCM音频 | ASRProto, int16 → float32 |
| ed_label | 译文 | ASRProto, int32 token ids |
| ed_label_src | 原文 | 同上 |
| ed_label_asr | 识别文本 | 同上 |
| ed_label_acc | 识别准确率 | ASRProto, int32 (0-100) |
| ed_label_kws | 翻译热词 | 同上, 格式: `将X翻译为Y:将A翻译为B:` |
| ce_label | phone序列 | 同上 (当前未使用，预留) |

关键实现:
- `LmdbDatasetEntry`: 自动发现并打开 train.json 中的所有 LMDB 字段，未来新增字段无需改代码
- `LmdbEnvWrapper`: 延迟打开 + fork安全（DataLoader多worker兼容）
- **双tokenizer**: LMDB数据用 GemmaTokenizer 编码，需要先 decode 回文本，再用 Qwen3.5 tokenizer 处理

### 2. 动态Batch (替代固定batch_size)

**文件**: `train_ep.py` - `DynamicBatchSampler`

原版使用 `DistributedSampler` + 固定 `batch_size`。改为基于 token 数量的动态 batch:

- `batch_tokens`: 一个 batch 内所有样本的 lengths 之和上限
- `max_batch_size`: 硬上限，防止短样本堆叠导致 OOM (因为 feature_extractor 固定输出 3000 帧 mel)
- 各 rank 的 batch 数量对齐（padding到相同数量），防止 EP all-to-all 通信不同步

当前推荐配置: `--batch_tokens 100000 --max_batch_size 2`，实际由 `max_batch_size` 主导。

### 3. Audio Encoder 开启 Flash Attention + 真实长度Concat

**文件**: `model_ep.py`

原版将 batch 内音频 padding 到相同长度再拼接。改为 concat 真实长度 + Flash Attention:

```python
# audio encoder 配置
audio_config._attn_implementation = 'flash_attention_2'
```

```python
# collator: 直接concat真实长度，无padding浪费
batch["input_features"] = torch.cat(audio_list, dim=1)  # (128, sum_of_real_lens)
batch["feature_lens"] = torch.tensor([real_len_1, real_len_2, ...])
```

**为什么需要 Flash Attention**:

Audio Encoder 内部使用 `cu_seqlens` 标记样本边界来隔离 attention。但 SDPA 模式下 `cu_seqlens` **被忽略**，所有 token 全局 attend，导致样本间信息泄露。FA2 模式通过 NPU 的 `npu_fusion_attention` 算子正确实现 varlen attention。

验证结果 (NPU bf16):
```
Per-sample FA2 vs Concat FA2:  max_diff = 5.1e-3  (bf16精度误差，一致)
Per-sample FA2 vs Concat SDPA: max_diff = 2.5e-2  (信息泄露，差5x)
```

**自动降级**: 当 FA2 不可用时，自动回退到逐样本编码模式:

```python
if self.audio_encoder.config._attn_implementation == 'flash_attention_2':
    # concat模式: 一次调用，cu_seqlens隔离
    audio_outputs = self.audio_encoder(input_features, feature_lens=feature_lens)
else:
    # per-sample模式: 逐样本编码，避免泄露
    for b in range(batch_size):
        ...
```

## 文件说明

```
.
├── train_ep.py          # 训练脚本 (LMDB数据集 + 动态batch + collator)
├── model_ep.py          # EP模型 (音频编码 + LLM + MoE EP)
├── infer_ep.py          # 推理脚本
├── run_ep.sh            # 训练启动脚本
├── run_infer_ep.sh      # 推理启动脚本
├── test_fa2_on_npu.py   # FA2一致性验证脚本
├── protofiles/          # ASRProto protobuf定义
├── file_io/             # LMDB/Lengths读写工具
└── lmdbdata/
    ├── train.json       # 训练数据配置 (LMDB路径列表)
    ├── lmdb/            # LMDB数据文件
    └── res/             # LMDB decode用的tokenizer (GemmaTokenizer)
```

## 训练参数

```bash
torchrun --nproc_per_node=8 train_ep.py \
    --batch_tokens 100000 \
    --max_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-5 \
    --num_epochs 50 \
    --max_seq_length 512 \
    --save_steps 600 \
    --output_dir output_ep_fast
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--batch_tokens` | batch内lengths之和上限 | 1024 |
| `--max_batch_size` | batch内样本数硬上限 | None |
| `--asr_acc_threshold` | ASR acc阈值，>=该值时填入识别文本 | 2.0 |
| `--lmdb_tokenizer_path` | LMDB数据的decode tokenizer路径 | lmdbdata/res |
| `--max_audio_length` | mel特征最大帧数 | 2400 |
| `--ep_size` | Expert Parallelism size | 8 |

## 环境依赖

- 华为昇腾 NPU + CANN 8.5.0
- torch 2.10.0 + torch_npu 2.10.0
- transformers (含 Qwen3OmniMoe 支持)
- lmdb, protobuf 3.20.3
