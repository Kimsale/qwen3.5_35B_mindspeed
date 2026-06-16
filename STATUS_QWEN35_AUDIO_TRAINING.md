# Qwen3.5-35B 音频训练 - 当前状态总结

**更新时间**: 2026-06-13 21:45
**任务**: Qwen3.5-35B-A3B + Whisper Large v3 语音多模态 LoRA 微调

---

## 一、之前的工作回顾

### ✅ 已完成（2026-06-01 至 06-12）

1. **Qwen3-30B-A3B 多模态训练**（6月4日）
   - 训练了 94/500 步后中断（未完成）
   - 模型：Qwen3-30B-A3B MoE
   - Loss: 0.5 → 0.09（正常下降）
   - 单步：4.8-11s（不稳定）
   - 输出：无 checkpoint 保存

2. **Qwen3.5-35B 纯文本 LoRA 训练**（6月12日）
   - ✅ **训练成功完成 60 步**
   - 模型：Qwen3.5-35B-A3B（40层，256专家/层）
   - 配置：LoRA r=16/α=32, FSDP2, 单机8卡
   - 性能：单步 1.9s, HBM 59.36GB/64GB (97.4%)
   - 产出：LoRA adapter checkpoint (44MB)
   - 位置：`/mnt/shared_data_196/sejin/models/Qwen3.5-35B-lora-ckpt/`

3. **Hulk 对标工作**（6月3-5日）
   - 完成了配置对齐（TP1/EP8/CP2）
   - 权重转换、数据准备
   - 性能对比分析

---

## 二、新任务：Qwen3.5-35B 音频训练

### 目标

训练 **Qwen3.5-35B-A3B + Whisper Large v3** 的语音理解能力：
- 音频 encoder：Whisper Large v3（冻结，提取特征）
- LLM：Qwen3.5-35B-A3B（LoRA 微调）
- Projector：全量训练（适配 whisper → LLM）

### 任务类型

可以训练多种语音任务：
1. **ASR（语音识别）**：<|AUDIO|> → 文字转写
2. **语音问答**：<|AUDIO|> + 问题 → 答案
3. **语音对话**：<|AUDIO|> → 自然对话回复
4. **语音指令遵循**：<|AUDIO|> + 指令 → 执行结果

---

## 三、当前准备情况

### ✅ 已完成的准备工作

| 项 | 状态 | 位置 |
|---|---|---|
| 模型权重（LLM） | ✅ | `/mnt/shared_data_196/sejin/models/Qwen3.5-35B-A3B-audio-dcp` (69GB) |
| 模型权重（Whisper） | ✅ | `/mnt/shared_data_196/sejin/models/whisper-large-v3` (2.9-5.8GB) |
| 训练配置 | ✅ | `/data/sejin/baseline_26/scripts/train_qwen35_audio.yaml` |
| 训练脚本 | ✅ | `/data/sejin/baseline_26/scripts/train_qwen35_audio.sh` |
| 数据准备工具 | ✅ | `/data/sejin/baseline_26/scripts/prepare_audio_data.py` |
| 快速开始指南 | ✅ | `/data/sejin/baseline_26/QWEN35_AUDIO_TRAINING_GUIDE.md` |
| 训练环境 | ✅ | `/data/sejin/env/venv_qwen35` (已验证) |
| 硬件 | ✅ | 单机 8×昇腾910B3 (64GB HBM/卡) |

### ⏳ 需要你完成的

**唯一缺失项：音频训练数据**

需要准备 JSONL 格式的音频+文本配对数据。

---

## 四、下一步行动（按优先级）

### 🎯 Step 1：准备音频数据（必须）

**选择一个数据来源**：

**选项 A：使用演示数据（快速测试）**
```bash
cd /data/sejin/baseline_26/scripts
python3 prepare_audio_data.py --mode demo --output /data/sejin/baseline_26/data_audio --num-samples 10
```
然后手动准备 10 个 .wav 文件到 `data_audio/audio/` 目录，或修改 JSONL 中的路径指向你的真实音频文件。

**选项 B：转换现有数据集**
```bash
# 如果你有 CSV 格式数据（audio_path, transcription）
python3 prepare_audio_data.py --mode csv \
    --input your_data.csv \
    --output /data/sejin/baseline_26/data_audio/train.jsonl \
    --audio-dir /path/to/audio/files

# 如果你有目录格式（音频在一个目录，转写在文本文件）
python3 prepare_audio_data.py --mode dir \
    --audio-dir /path/to/audio/files \
    --transcription-file transcriptions.txt \
    --output /data/sejin/baseline_26/data_audio/train.jsonl
```

**选项 C：直接编写 JSONL**

按照以下格式手动创建 `/data/sejin/baseline_26/data_audio/train.jsonl`：
```jsonl
{"id": "sample_001", "audios": ["/path/to/audio1.wav"], "messages": [{"role": "user", "content": "<|AUDIO|>\n请转写这段语音。"}, {"role": "assistant", "content": "今天天气很好。"}]}
{"id": "sample_002", "audios": ["/path/to/audio2.wav"], "messages": [{"role": "user", "content": "<|AUDIO|>\n这段音频说了什么？"}, {"role": "assistant", "content": "会议定于下周三召开。"}]}
```

**验证数据**：
```bash
python3 prepare_audio_data.py --mode validate --input /data/sejin/baseline_26/data_audio/train.jsonl
```

---

### 🚀 Step 2：小规模测试（10步）

数据准备完成后，先跑 10 步验证配置：

```bash
# 1. 修改配置为小规模测试
cd /data/sejin/baseline_26/scripts
# 编辑 train_qwen35_audio.yaml，改 max_steps: 10, save_interval: 5

# 2. 启动训练
chmod +x train_qwen35_audio.sh
./train_qwen35_audio.sh

# 3. 监控日志
tail -f /data/sejin/baseline_26/logs/qwen35_audio_*.log
```

**关键检查点**：
- [ ] 训练成功启动（无模型加载错误）
- [ ] 音频数据正常加载（无路径错误）
- [ ] 首步完成（无 OOM、无崩溃）
- [ ] Loss 正常（首步约 2-5，后续下降）
- [ ] HBM 占用合理（55-62GB，与之前接近）
- [ ] 单步耗时合理（2-5s）

---

### 📊 Step 3：完整训练（500步）

测试通过后，恢复完整配置并重新训练：

```bash
# 1. 恢复配置
# 编辑 train_qwen35_audio.yaml，改回 max_steps: 500, save_interval: 100

# 2. 重新启动训练
./train_qwen35_audio.sh

# 3. 等待训练完成（预计 20-40 分钟）
```

**预期产出**：
- Checkpoint: `/data/sejin/baseline_26/output/qwen35_audio_ckpt/checkpoint-{100,200,300,400,500}/`
- LoRA adapter: `checkpoint-500/lora_adapter_model.safetensors` (~44MB)
- 日志: `/data/sejin/baseline_26/logs/qwen35_audio_*.log`

---

### ✅ Step 4：验证效果

训练完成后，用 LoRA adapter 进行推理验证效果。

---

## 五、关键文件速查

```bash
# 配置文件
/data/sejin/baseline_26/scripts/train_qwen35_audio.yaml      # 训练配置
/data/sejin/baseline_26/scripts/train_qwen35_audio.sh        # 启动脚本
/data/sejin/baseline_26/scripts/prepare_audio_data.py        # 数据准备工具

# 文档
/data/sejin/baseline_26/QWEN35_AUDIO_TRAINING_GUIDE.md       # 详细指南

# 模型
/mnt/shared_data_196/sejin/models/Qwen3.5-35B-A3B-audio-dcp  # LLM 权重
/mnt/shared_data_196/sejin/models/whisper-large-v3           # Whisper 权重

# 数据（需你准备）
/data/sejin/baseline_26/data_audio/train.jsonl               # 训练数据
/data/sejin/baseline_26/data_audio/val.jsonl                 # 验证数据（可选）

# 输出
/data/sejin/baseline_26/output/qwen35_audio_ckpt/            # Checkpoint
/data/sejin/baseline_26/logs/qwen35_audio_*.log              # 训练日志

# 环境
/data/sejin/env/venv_qwen35                                   # Python 环境
```

---

## 六、常见问题速查

| 问题 | 解决方案 |
|---|---|
| 数据路径错误 | 检查 JSONL 中 `audios` 字段的路径是否正确 |
| OOM（显存不足） | 降低 `max_seq_length` (2048→1024) 或启用 `activation_offload` |
| 音频格式不支持 | 转换为 16kHz .wav: `ffmpeg -i input.mp3 -ar 16000 output.wav` |
| 模型加载失败 | 确认 `model_path` 指向 **audio-dcp** 版本 |
| 单步过慢 | 正常现象（Triton 编译），后续会加速 |

---

## 七、性能预期（基于历史经验）

| 指标 | 预期值 | 备注 |
|---|---|---|
| 单步耗时 | 2-5s | 音频预处理比纯文本略慢 |
| HBM 占用 | 55-62GB/64GB | 与纯文本训练接近（audio_tower 冻结） |
| Loss 初始值 | 2-5 | 取决于任务类型 |
| Loss 收敛 | 0.5-1.5 | 500步后 |
| 训练总时长 | 20-40 分钟 | 500步，单步约 2-5s |

---

## 八、联系与参考

- **详细指南**：`/data/sejin/baseline_26/QWEN35_AUDIO_TRAINING_GUIDE.md`
- **官方示例**：`/data/sejin/third_party/mindspeed-mm-26.0.0/examples/qwen3_5_audio/`
- **历史报告**：`/data/sejin/baseline_26/reports/qwen35_35B_lora_60step_perf_report.md`

---

**总结**：所有配置和脚本已就绪，只需准备音频数据即可立即开始训练。数据准备是唯一的阻塞项，完成后即可启动。
