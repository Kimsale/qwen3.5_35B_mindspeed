# Qwen3.5-35B + Whisper Large v3 LoRA Audio Training Performance Audit

**生成时间**: 2026-06-15  
**审计对象**: `/data/sejin/baseline_26/scripts/run_audio_dist100.sh` 触发的 100 step 训练  
**主日志**: `/data/sejin/baseline_26/logs/audio_dist100_20260615_122021.log`  
**实际配置**: `/data/sejin/third_party/mindspeed-mm-26.0.0/examples/qwen3_5_audio/dist100_config.yaml`  
**已有最终报告**: `/data/sejin/baseline_26/reports/qwen35_audio_dist100_final_report.md`

---

## 1. 结论摘要

| 项 | 结论 |
|---|---|
| 训练是否跑通 | 是。100/100 step 完成，退出码 0，保存 step 50/100 LoRA checkpoint。 |
| 是否是真实音频路径 | 是。样本含 `audios` 字段，数据侧展开 `<|audio_pad|>`，模型侧 Whisper encoder + audio projector 产出 embedding 后 scatter 到 LLM embedding。 |
| 是否能代表真实多模态训练性能 | 只能代表 **合成音频、冻结 Whisper/projector、LLM self-attn LoRA** 这一路径的吞吐下限/冒烟性能，不能代表真实业务音频语义训练效果，也不能代表 projector 联训性能。 |
| 最大流程问题 | 配置注释写 `projector 全量可训`，但实际 trainable 参数只有 LoRA 3,440,640，`audio_projector` 未参与训练。 |
| 硬件利用率报告是否完整 | 不完整。当前训练未启用 profiler，也没有音频训练期间的 NPU 监控 JSON；AICORE/HBM/功耗不能从现有产物中实测还原。 |
| 当前可用性能指标 | 稳定段 step time 6.830s，吞吐 4.685 samples/s，实际 input-token WPS 约 821 token/s，audio-pad token WPS 约 648 token/s。 |

---

## 2. 实际训练流程核对

### 2.1 启动链路

| 环节 | 实际值 |
|---|---|
| 启动脚本 | `/data/sejin/baseline_26/scripts/run_audio_dist100.sh` |
| 工作目录 | `/data/sejin/third_party/mindspeed-mm-26.0.0` |
| 训练入口 | `mindspeed_mm/fsdp/train/trainer.py` |
| 配置文件 | `examples/qwen3_5_audio/dist100_config.yaml` |
| 并行方式 | 单机 8 卡，`torchrun --nproc_per_node 8`，FSDP2，TP=1，EP=1 |
| CANN/ATB | `cann-8.5.0` + `nnal/atb` |
| venv | `/data/sejin/env/venv_qwen35` |
| AUDIO placeholder | 环境变量 `AUDIO_PLACEHOLDER="<\|AUDIO\|>"` |

注意：`/data/sejin/baseline_26/scripts/train_qwen35_audio.yaml` 不是本次日志对应的实际训练配置；性能口径应以 `dist100_config.yaml` 为准。

### 2.2 模型与可训练参数

| 项 | 实际值 |
|---|---|
| model_id | `qwen3_5_audio` |
| 基座 | `/mnt/shared_data_196/sejin/models/Qwen3.5-35B-A3B` |
| DCP load | `/mnt/shared_data_196/sejin/models/Qwen3.5-35B-A3B-audio-dcp` |
| Whisper | `/mnt/shared_data_196/sejin/models/whisper-large-v3` |
| audio token id | `248076` (`<\|audio_pad\|>`) |
| LoRA target | `self_attn.{q,k,v,o}_proj`，共 40 个模块 |
| LoRA 参数 | 80 个 tensor，3,440,640 elements |
| trainable ratio | 约 0.01% |
| projector | 代码路径存在，但本次未作为 trainable base 参数进入 optimizer |

关键证据：日志中各 rank 打印 `Trainable parameters: 3,440,640`，等于 80 个 LoRA tensor；如果 `audio_projector` 联训，trainable 参数量应显著高于该值。

### 2.3 数据路径

| 项 | 实际值 |
|---|---|
| JSONL | `/data/sejin/baseline_26/data_audio/train.jsonl` |
| 样本数 | 3,200 |
| 音频样本 | 3,050 |
| 纯文本样本 | 150 |
| 总音频时长 | 4.914 小时 |
| 音频格式 | 16 kHz mono wav |
| 数据性质 | 合成波形 + 合成中文文本，不是真实录音/真实转写 |

### 2.4 音频长度分布复算

| 分位 | 实际值 |
|---|---:|
| p5 | 0.777 s |
| p25 | 2.326 s |
| p50 | 4.692 s |
| mean | 5.800 s |
| p75 | 9.291 s |
| p90 | 11.364 s |
| p95 | 14.502 s |
| max | 20.000 s |
| `<3s` 占比 | 30.69% |

旧报告中 `<3s` 约 40% 的说法与当前文件复算不一致，应以本报告复算值为准。

---

## 3. 性能指标

### 3.1 训练日志指标

| 指标 | 值 |
|---|---:|
| 训练步数 | 100 |
| Global batch size | 32 |
| 消费样本数 | 3,200 |
| step 1 loss | 11.40394 |
| step 100 loss | 2.770513 |
| loss 降幅 | 75.71% |
| step 1 grad norm | 0.648 |
| step 100 grad norm | 5.881 |
| NaN/skipped | 日志未见 NaN/skipped |

### 3.2 吞吐

| 口径 | 均值 step time | 中位 step time | p95 step time | 吞吐 |
|---|---:|---:|---:|---:|
| 全 100 step | 6.941 s | 6.669 s | 7.571 s | 4.610 samples/s |
| 稳定段 step 10-100 | 6.830 s | 6.661 s | 7.478 s | 4.685 samples/s |
| 稳定段 step 20-100 | 6.822 s | 6.659 s | 7.478 s | 4.691 samples/s |

推荐使用 `step 10-100` 作为报告主口径，避开首步编译/缓存/数据预热。

### 3.3 WPS / token/s

本报告按训练后 cache 中的实际 `input_ids` 统计 WPS。这里的 WPS 表示 **processed input tokens per second**，不是自然语言“词/秒”。

| 口径 | 数据统计 | 稳定段 WPS |
|---|---:|---:|
| 实际 input token | mean 175.336 tokens/sample，sum 561,075 | 821.4 token/s |
| 实际 label token | mean 19.566 tokens/sample，sum 62,612 | 91.7 token/s |
| 实际 audio-pad token | mean 138.211 tokens/sample，sum 442,276 | 647.5 token/s |
| cutoff_len 上界 | 4096 tokens/sample | 19,189.5 token/s |

建议对外只报“实际 input-token WPS”和“audio-pad token WPS”。`cutoff_len` 上界只适合做容量上限说明，不能当真实吞吐。

---

## 4. AICORE / HBM / 功耗

### 4.1 当前实测状态

| 指标 | 当前值 | 原因 |
|---|---:|---|
| AICORE 利用率 | N/A | 本次训练 `tools.profile.enable=false`，且没有音频训练期间的 NPU 监控 JSON。 |
| HBM 使用 | N/A | 同上，日志未打印每卡 HBM 峰值。 |
| 功耗 | N/A | `scripts/npu_monitor.py` 未随本次训练启动；本次审计已修复 power 覆盖问题，但历史训练无法回溯功耗。 |
| HBM 带宽 | N/A | `npu-smi info` 通常不提供可靠带宽；需要 profiler 或厂商工具采集。 |

结论：当前产物不能严谨回答 AICORE/HBM/功耗。任何具体百分比或瓦数都必须通过复跑采集获得。

### 4.2 推荐复跑采集方式

最小侵入方案：

```bash
cd /data/sejin/baseline_26
mkdir -p metrics
/data/sejin/env/venv_qwen35/bin/python scripts/npu_monitor.py \
  metrics/audio_dist100_npu_$(date +%Y%m%d_%H%M%S).json 900 &
MON_PID=$!

bash scripts/run_audio_dist100.sh
kill $MON_PID
```

更严谨方案：

1. 在 `examples/qwen3_5_audio/dist100_config.yaml` 中开启 profiler，只采 step 20-22，避免全程 profile 扰动。
2. `tools.profile.static_param.aic_metrics_type` 使用 `PipeUtilization`。
3. `with_memory=true`，用于补充 HBM 峰值与算子内存视图。
4. 对性能主报告仍使用非 profiler 训练日志，因为 profiler 会降低吞吐。

---

## 5. 是否代表真实多模态训练性能

### 5.1 可以代表的部分

| 能代表 | 说明 |
|---|---|
| MindSpeed-MM FSDP2 路线能跑通 Qwen3.5-35B + Whisper audio 输入 | 训练完成且 checkpoint 保存。 |
| 音频 token 展开与模型 scatter 路径自洽 | 未出现 audio token 数不匹配，cache 中 audio-pad token 总数 442,276。 |
| LoRA 梯度链已恢复 | grad norm 非 0，loss 明显下降，step 50/100 safetensors 不同。 |
| 当前 batch/seq/audio 分布下的端到端 step time | 日志逐 step 记录，可作为当前合成 workload 的吞吐基线。 |

### 5.2 不能代表的部分

| 不能代表 | 原因 |
|---|---|
| 真实业务 ASR/语音理解收敛效果 | 音频波形和转写文本都是合成的，语音内容与标签没有真实语义对应。 |
| projector 联训性能 | 实际 trainable 参数只有 LoRA，projector 未训练。 |
| 真实长尾音频性能 | 当前 max 20s；团队分布 max 226.7s 虽会被 Whisper 截断，但长音频 I/O、预处理和截断行为仍未覆盖。 |
| AICORE/HBM/功耗效率 | 本次没有采集硬件计数器。 |
| 多轮/多音频/视频混合训练 | 当前样本是单音频为主，150 条 text-only。 |

---

## 6. 与业内最佳实践的差距

| 最佳实践项 | 当前状态 | 建议 |
|---|---|---|
| 报告同时给出质量、吞吐、硬件利用率、稳定性 | 部分满足 | 补采 AICORE/HBM/功耗；加入 eval 指标。 |
| 区分 warmup、steady-state、checkpoint step | 已可区分 | 主口径使用 step 10-100 或去除 checkpoint step。 |
| 明确数据真实性与可代表性 | 旧报告不足 | 本报告已标注合成数据限制。 |
| 记录实际 git diff / 框架 patch | 部分满足 | 固化 `_reset_lora_params` 修复说明，并独立提交。 |
| 训练目标与可训练参数一致 | 不满足 | 若目标是音频适配，应显式解冻 `audio_projector` 或给 projector 加 LoRA。 |
| 性能采集低扰动 | 未完成 | npu-smi 低频采样用于功耗/HBM，profiler 只采少量 step。 |
| 报告口径可复现 | 部分满足 | 保存 metrics JSON、profile 目录、日志解析脚本。 |

---

## 7. 建议修正项

1. 若目标是“语音 encoder + LLM 适配”的真实多模态训练，应让 `audio_projector` 参与训练；否则 Whisper 输出到 LLM embedding 的映射无法学习。
2. 将 `dist100_config.yaml` 注释改为实际语义：当前是 `Whisper frozen + projector frozen + LLM self-attn LoRA`。
3. 使用真实录音/真实转写或至少 TTS 语义一致数据复跑，否则 loss 下降更多反映标签模板/短文本记忆，不代表语音理解能力。
4. 启用已修正的 `scripts/npu_monitor.py`，采集 AICORE/HBM/功耗 JSON。
5. 开启短窗口 profiler 采样 step 20-22，补充 AICORE PipeUtilization 和 HBM 峰值。
6. 对外报告统一使用实际配置 `dist100_config.yaml`，不要混用 `scripts/train_qwen35_audio.yaml`。

---

## 8. 标准报告模板映射

| 模板章节 | 本报告位置 |
|---|---|
| Executive Summary | 第 1 节 |
| Environment / Hardware / Software | 第 2.1 节 |
| Model / Parallelism / Trainable Params | 第 2.2 节 |
| Dataset / Workload | 第 2.3-2.4 节 |
| Training Stability | 第 3.1 节 |
| Throughput / WPS | 第 3.2-3.3 节 |
| Hardware Utilization | 第 4 节 |
| Representativeness | 第 5 节 |
| Best-practice Gap Analysis | 第 6 节 |
| Action Items | 第 7 节 |
