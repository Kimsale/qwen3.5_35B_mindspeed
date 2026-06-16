# Qwen3.5-35B Audio LoRA EP Sweep Note

**生成时间**: 2026-06-15  
**目标**: 验证 Whisper-large-v3 encoder 冻结 + Qwen3.5-35B-A3B audio LoRA 训练在较大 EP 下能否跑通，并评估性能可用性。  
**基线**: `/data/sejin/baseline_26/reports/qwen35_audio_distfix60_performance_report_20260615.md`

---

## 1. Runs

| EP | 配置 | 启动脚本 | 日志 | 结果 |
|---:|---|---|---|---|
| 1 | `/data/sejin/third_party/mindspeed-mm-26.0.0/examples/qwen3_5_audio/distfix60_config.yaml` | `/data/sejin/baseline_26/scripts/run_audio_distfix60.sh` | `/data/sejin/baseline_26/logs/audio_distfix60_20260615_161439.log` | 成功完成 60/60 step |
| 8 | `/data/sejin/third_party/mindspeed-mm-26.0.0/examples/qwen3_5_audio/distfix60_ep8_config.yaml` | `/data/sejin/baseline_26/scripts/run_audio_distfix60_ep8.sh` | `/data/sejin/baseline_26/logs/audio_distfix60_ep8_20260615_172803.log` | OOM，未完成首个 iteration |
| 4 | `/data/sejin/third_party/mindspeed-mm-26.0.0/examples/qwen3_5_audio/distfix60_ep4_config.yaml` | `/data/sejin/baseline_26/scripts/run_audio_distfix60_ep4.sh` | `/data/sejin/baseline_26/logs/audio_distfix60_ep4_20260615_174133.log` | OOM，未完成首个 iteration |

---

## 2. Parallel Mesh

| EP | EP mesh | eFSDP mesh | 判断 |
|---:|---|---|---|
| 1 | `[ep] = False, Group size: 1` | `[efsdp] = True, Group size: 8` | 专家参数仍可跨 8 卡分片，显存健康 |
| 8 | `[ep] = True, Group size: 8` | `[efsdp] = False, Group size: 1` | 单机 8 卡 EP=8 后没有专家再分片维度，显存接近满卡 |
| 4 | `[ep] = True, Group size: 4` | `[efsdp] = True, Group size: 2` | 保留 2 卡专家再分片，但仍不足以跑通当前 audio LoRA 配置 |

EP=8 的关键问题不是 EP mesh 建不起来，而是 8 卡单机上 `expert_fully_shard_parallel_size = world_size / EP = 1`，导致本地专家没有 eFSDP 再分片。当前 35B-A3B audio wrapper + LoRA + FSDP2 路径会把每卡显存推到 64GB 附近。

---

## 3. Failure Point

EP=8 和 EP=4 都完成了以下阶段：

| 阶段 | EP=8 | EP=4 |
|---|---|---|
| 配置解析 | 通过 | 通过 |
| EP mesh 初始化 | 通过 | 通过 |
| 数据/tokenizer | 通过 | 通过 |
| LoRA 注入 | 通过，trainable params 3,440,640 | 通过，trainable params 3,440,640 |
| DCP checkpoint load | 通过 | 通过 |
| 首个 optimizer step | OOM | OOM |

OOM 位置均在 `optimizer.step()` 初始化 Adam 状态时：

| EP | 典型错误 |
|---:|---|
| 8 | `NPU out of memory. Tried to allocate 130/258 MiB ... 59.15-59.40 GiB already allocated ... 101-232 MiB free` |
| 4 | `NPU out of memory. Tried to allocate 258 MiB ... 58.65-58.90 GiB already allocated ... 200-234 MiB free` |

---

## 4. Hardware Snapshot

`npu-smi` 是失败 run 的端到端采样，包含 init 和 OOM，不代表训练吞吐。

| EP | 状态 | AICORE mean/peak | HBM mean/peak | Power mean/peak |
|---:|---|---:|---:|---:|
| 1 | 60 step 成功 | 9.5% / 33.0% | 32,513 / 33,366 MB | 158.3 / 192.2 W |
| 8 | OOM | 1.0% / 46.0% | 37,849 / 65,438 MB | 97.9 / 164.5 W |
| 4 | OOM | 0.2% / 7.0% | 39,098 / 65,483 MB | 98.5 / 143.6 W |

EP=8/EP=4 没有完成有效训练 step，因此不能报告 samples/s、WPS 或 post-warmup AICORE。可报告的只有“初始化阶段显存峰值已接近或达到 HBM 上限”。

---

## 5. Conclusion

当前单机 8 x Ascend 910B3 64GB 环境下，`Qwen3.5-35B-A3B + Whisper-large-v3 audio + LoRA + FSDP2`：

| 配置 | 是否建议作为性能口径 |
|---|---|
| EP=1 | 建议，已成功完成 60 step，有 post-warmup 性能指标 |
| EP=4 | 不建议，首个 optimizer step OOM |
| EP=8 | 不建议，首个 optimizer step OOM，且没有专家再分片维度 |

如果必须验证 EP=8 的真实性能，建议换成以下条件之一：

1. 使用至少 16 卡，让 `EP=8` 同时保留 `expert_fully_shard_parallel_size >= 2`。
2. 启用 FSDP2 CPU offload，把参数/梯度/optimizer state 卸载到 CPU；这可能能跑通，但性能会显著下降，不能和当前 EP=1 纯 NPU 口径直接比较。
3. 降低模型/训练显存压力，例如更小 MoE、关闭 audio wrapper、缩短 cutoff、降低 LoRA target 数，先做功能验证；这不再代表当前真实 audio LoRA 训练性能。
4. 若目标是单机 8 卡性能，保持 EP=1 是当前唯一有效基线。
