# Qwen3.5-35B 音频训练 - 快速开始指南

**生成时间**: 2026-06-13
**任务**: Qwen3.5-35B-A3B + Whisper Large v3 语音多模态 LoRA 微调
**框架**: MindSpeed-MM 26.0.0 (FSDP2)
**硬件**: 单机 8×昇腾910B3

---

## 一、当前状态

### ✅ 已准备就绪的资源

1. **模型权重**（新位置 `/mnt/shared_data_196/sejin/models/`）
   - ✅ Qwen3.5-35B-A3B-audio-dcp (69GB, MCore DCP格式)
   - ✅ whisper-large-v3 (2.9-5.8GB, HF格式)

2. **训练配置**
   - ✅ `/data/sejin/baseline_26/scripts/train_qwen35_audio.yaml`
   - ✅ `/data/sejin/baseline_26/scripts/train_qwen35_audio.sh`

3. **环境**
   - ✅ venv_qwen35（已验证，transformers fc91372, torch_npu 2.7.1.post2）

4. **历史经验**
   - ✅ 6月12日成功跑通 Qwen3.5-35B 纯文本 LoRA（60步，HBM 59.36GB/64GB）

### ⏳ 需要准备的

**关键缺失项：音频训练数据**

---

## 二、数据准备（必须完成）

### 2.1 数据格式要求

音频 SFT 数据格式（JSONL，每行一个样本）：

```jsonl
{"id": "sample_001", "audios": ["path/to/audio1.wav"], "messages": [{"role": "user", "content": "<|AUDIO|>\n请转写这段语音。"}, {"role": "assistant", "content": "今天天气很好。"}]}
{"id": "sample_002", "audios": ["path/to/audio2.wav"], "messages": [{"role": "user", "content": "<|AUDIO|>\n这段音频说了什么？"}, {"role": "assistant", "content": "会议定于下周三召开。"}]}
```

**字段说明**：
- `id`: 样本唯一标识
- `audios`: 音频文件路径列表（相对或绝对路径，支持 .wav/.mp3/.flac）
- `messages`: 对话格式
  - `<|AUDIO|>` 是特殊 token（会被替换为 whisper 提取的音频特征）
  - user content 可以在 `<|AUDIO|>` 后添加指令
  - assistant content 是期望的回答（用于计算 loss）

### 2.2 音频要求

- **采样率**: 16kHz（Whisper 要求）
- **时长**: 建议 ≤30秒（配置中 `max_audio_length: 30.0`）
- **格式**: .wav / .mp3 / .flac（HuggingFace datasets 自动支持）

### 2.3 数据准备步骤

**选项 A：如果你已有音频数据**

1. 创建数据目录：
   ```bash
   mkdir -p /data/sejin/baseline_26/data_audio
   ```

2. 准备 JSONL 文件：
   - 训练集：`/data/sejin/baseline_26/data_audio/train.jsonl`
   - 验证集（可选）：`/data/sejin/baseline_26/data_audio/val.jsonl`

3. 确保音频文件可访问（JSONL 中的路径必须有效）

**选项 B：使用官方示例数据（快速测试）**

MindSpeed-MM 提供了示例数据：

```bash
# 复制示例数据到 baseline_26
cp /data/sejin/third_party/mindspeed-mm-26.0.0/examples/qwen3_5_audio/audio_sft_demo.jsonl \
   /data/sejin/baseline_26/data_audio/train.jsonl

# 创建音频文件目录
mkdir -p /data/sejin/baseline_26/data_audio/audio

# 准备示例音频文件（需要你提供真实的 .wav 文件）
# 或者生成合成音频用于测试
```

**选项 C：如果你有现成的 ASR/语音对话数据集**

常见公开数据集：
- **Aishell-1/2**：中文 ASR
- **Common Voice**：多语言 ASR
- **VCTK**：英文语音
- **自有数据**：你的业务数据

转换脚本示例（需根据你的数据格式调整）：

```python
import json

# 假设你有 CSV 格式数据：audio_path, transcription
with open('your_data.csv') as f_in, open('train.jsonl', 'w') as f_out:
    for i, line in enumerate(f_in):
        audio_path, text = line.strip().split(',')
        sample = {
            "id": f"asr_{i:06d}",
            "audios": [audio_path],
            "messages": [
                {"role": "user", "content": "<|AUDIO|>\n请转写这段语音。"},
                {"role": "assistant", "content": text}
            ]
        }
        f_out.write(json.dumps(sample, ensure_ascii=False) + '\n')
```

---

## 三、训练启动流程

### 3.1 确认数据已准备

```bash
# 检查数据文件是否存在
ls -lh /data/sejin/baseline_26/data_audio/train.jsonl

# 检查数据格式（读取前3行）
head -3 /data/sejin/baseline_26/data_audio/train.jsonl

# 检查音频文件是否可访问
# (从 JSONL 中提取第一个音频路径并检查)
```

### 3.2 启动训练（小规模测试）

**首次建议：先跑 10 步测试，验证配置正确**

修改配置文件 `/data/sejin/baseline_26/scripts/train_qwen35_audio.yaml`：

```yaml
training:
  max_steps: 10  # 改为 10（原来是 500）
  save_interval: 5  # 改为 5（快速保存测试）
```

启动训练：

```bash
cd /data/sejin/baseline_26/scripts
chmod +x train_qwen35_audio.sh
./train_qwen35_audio.sh
```

**关键观察点**：
- [ ] 训练是否成功启动（无模型加载错误）
- [ ] 音频数据是否正常加载（无路径错误）
- [ ] 首步是否完成（无 OOM、无崩溃）
- [ ] Loss 是否正常（首步约 2-5，后续下降）
- [ ] HBM 占用是否合理（预期 55-62GB/64GB，与之前纯文本训练接近）
- [ ] 单步耗时是否合理（预期 2-5s，音频预处理会略慢于纯文本）

### 3.3 完整训练（测试通过后）

改回完整配置：

```yaml
training:
  max_steps: 500  # 恢复 500 步
  save_interval: 100  # 每 100 步保存
```

重新启动训练。

---

## 四、预期结果

### 4.1 训练输出

**Checkpoint 位置**：
- `/data/sejin/baseline_26/output/qwen35_audio_ckpt/checkpoint-100/`
- `/data/sejin/baseline_26/output/qwen35_audio_ckpt/checkpoint-200/`
- ...
- `/data/sejin/baseline_26/output/qwen35_audio_ckpt/checkpoint-500/`

**LoRA adapter**（最终产物）：
- `checkpoint-500/lora_adapter_model.safetensors` (~44MB，与之前纯文本 LoRA 类似)

**日志**：
- `/data/sejin/baseline_26/logs/qwen35_audio_YYYYMMDD_HHMMSS.log`

### 4.2 性能预期（基于之前经验）

| 指标 | 预期值 | 说明 |
|---|---|---|
| 单步耗时 | 2-5s | 音频预处理比纯文本略慢（Whisper 特征提取） |
| HBM 占用 | 55-62GB | 接近之前纯文本训练（59.36GB），audio_tower 冻结 |
| 首步编译 | ~30-60s | Triton 编译，仅首步 |
| Loss 初始值 | 2-5 | 取决于任务（ASR vs 对话） |
| Loss 收敛 | 0.5-1.5 | 500步后，取决于数据质量 |

### 4.3 推理验证（训练完成后）

使用训练好的 LoRA adapter 进行推理：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# 加载基座模型
base_model = AutoModelForCausalLM.from_pretrained(
    "/mnt/shared_data_196/sejin/models/Qwen3.5-35B-A3B",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 加载 LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    "/data/sejin/baseline_26/output/qwen35_audio_ckpt/checkpoint-500"
)

# 推理（伪代码，具体 API 需查 MindSpeed-MM 文档）
# audio_input = load_audio("test.wav")
# response = model.generate(audio_input, prompt="<|AUDIO|>\n请转写")
```

---

## 五、故障排查

### 5.1 常见问题

**问题 1：数据加载失败**

```
FileNotFoundError: [Errno 2] No such file or directory: 'path/to/audio.wav'
```

**解决**：检查 JSONL 中的 `audios` 路径是否正确（相对路径 vs 绝对路径）

---

**问题 2：OOM（显存不足）**

```
RuntimeError: NPU out of memory
```

**解决**：
1. 降低 `micro_batch_size`（1 → 1，已是最小）
2. 启用 activation offload（在 yaml 中设 `activation_offload: true`）
3. 降低 `max_seq_length`（2048 → 1024）

---

**问题 3：音频格式不支持**

```
ValueError: Unsupported audio format
```

**解决**：
1. 转换为 .wav 16kHz：`ffmpeg -i input.mp3 -ar 16000 output.wav`
2. 或安装音频处理库：`pip install soundfile librosa`

---

**问题 4：模型加载失败**

```
KeyError: 'audio_tower' or similar
```

**解决**：
1. 确认 `model_path` 指向的是 **Qwen3.5-35B-A3B-audio-dcp**（带音频的版本）
2. 确认 `whisper_path` 指向正确
3. 检查 plugin 注册是否正确（`mindspeed_mm/fsdp/models/qwen3_5_audio`）

---

## 六、下一步行动清单

**立即执行**（按顺序）：

- [ ] **Step 1**: 准备音频训练数据
  - 选择数据来源（选项 A/B/C）
  - 生成 `/data/sejin/baseline_26/data_audio/train.jsonl`
  - 验证数据格式和音频文件可访问性

- [ ] **Step 2**: 小规模测试（10步）
  - 修改配置 `max_steps: 10`
  - 执行 `./train_qwen35_audio.sh`
  - 检查启动、数据加载、首步完成

- [ ] **Step 3**: 完整训练（500步）
  - 恢复配置 `max_steps: 500`
  - 重新启动训练
  - 监控 loss 收敛和 HBM 占用

- [ ] **Step 4**: 验证产出
  - 检查 checkpoint 保存
  - 提取 LoRA adapter
  - 推理验证效果

---

## 七、关键文件位置

| 类型 | 路径 |
|---|---|
| 模型权重（LLM） | `/mnt/shared_data_196/sejin/models/Qwen3.5-35B-A3B-audio-dcp` |
| 模型权重（Whisper） | `/mnt/shared_data_196/sejin/models/whisper-large-v3` |
| 训练配置 | `/data/sejin/baseline_26/scripts/train_qwen35_audio.yaml` |
| 训练脚本 | `/data/sejin/baseline_26/scripts/train_qwen35_audio.sh` |
| 训练数据 | `/data/sejin/baseline_26/data_audio/train.jsonl` |
| Checkpoint 输出 | `/data/sejin/baseline_26/output/qwen35_audio_ckpt/` |
| 训练日志 | `/data/sejin/baseline_26/logs/qwen35_audio_*.log` |
| 环境 | `/data/sejin/env/venv_qwen35` |

---

## 八、联系与支持

- MindSpeed-MM 官方文档：`/data/sejin/third_party/mindspeed-mm-26.0.0/README.md`
- Qwen3.5 音频示例：`/data/sejin/third_party/mindspeed-mm-26.0.0/examples/qwen3_5_audio/`
- 历史训练总结：`/data/sejin/baseline_26/reports/qwen35_35B_lora_60step_perf_report.md`

---

**当前状态**：配置已就绪，等待音频数据准备完成后即可启动训练。
