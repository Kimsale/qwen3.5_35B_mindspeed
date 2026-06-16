# Qwen3.5-35B + Whisper Large v3 Audio LoRA 60-Step Performance Report

**生成时间**: 2026-06-15  
**任务**: 修正音频时长分布后，使用修正后的监控脚本复跑 60 step，并补充 profiler step 20-22 纯训练窗口指标  
**框架**: MindSpeed-MM 26.0.0 / FSDP2  
**硬件**: 单机 8 x Ascend 910B3, 64GB HBM/卡  
**训练日志**: `/data/sejin/baseline_26/logs/audio_distfix60_20260615_161439.log`  
**NPU 监控**: `/data/sejin/baseline_26/metrics/audio_distfix60_20260615_161439_npu.json`  
**Profiler 日志**: `/data/sejin/baseline_26/logs/audio_distfix_profile_step20_22_20260615_164823.log`  
**Profiler 输出**: `/data/sejin/baseline_26/profiling/audio_distfix_step20_22`  
**训练配置**: `/data/sejin/third_party/mindspeed-mm-26.0.0/examples/qwen3_5_audio/distfix60_config.yaml`  
**Profiler 配置**: `/data/sejin/third_party/mindspeed-mm-26.0.0/examples/qwen3_5_audio/distfix_profile_step20_22_config.yaml`  
**启动脚本**: `/data/sejin/baseline_26/scripts/run_audio_distfix60.sh`  
**Profiler 启动脚本**: `/data/sejin/baseline_26/scripts/run_audio_distfix_profile_step20_22.sh`

---

## 1. Executive Summary

本报告的吞吐、WPS、step time 主口径均采用预热后的训练窗口，排除进程启动、模型初始化、数据准备/tokenizer、DCP load、首步编译/缓存和 checkpoint 保存。端到端硬件指标保留 `npu-smi` 全程采样，同时并列给出 profiler step 20-22 纯训练窗口，用于正式提交 AICORE/算子侧指标。

| 指标 | 结果 |
|---|---:|
| 训练状态 | 成功完成 60/60 step，退出码 0 |
| 消费样本 | 1,920 samples |
| Global batch size | 32 |
| 首步 loss | 11.5393 |
| 末步 loss | 5.2165 |
| loss 降幅 | 54.79% |
| grad norm 范围 | 0.668 - 25.010 |
| 主口径 step time, step10-60 | 6.910 s |
| 主口径吞吐, step10-60 | 4.631 samples/s |
| 主口径 input-token WPS, step10-60 | 804.2 token/s |
| 主口径 audio-pad token WPS, step10-60 | 632.6 token/s |
| Profiler 纯训练 stage, step20-22 | 6.937 s/step |
| Profiler 纯训练吞吐, step20-22 | 4.613 samples/s |
| Profiler AIC/MIX cube utilization | 69.63% duration-weighted |
| Profiler HBM bandwidth | read 72,669 MB/s, write 54,545 MB/s |
| npu-smi AICORE, 全程端到端 | mean 9.5%, peak 33.0% |
| npu-smi HBM, 全程端到端 | mean 32,513 MB/卡, peak 33,366 MB/卡 |
| npu-smi 功耗, 全程端到端 | mean 158.3 W/卡, peak 192.2 W/卡 |
| Checkpoint | `lora_adapter_iteration_60.safetensors`, 14 MB |

结论：修正后的数据分布解决了 `<3s` 占比偏低问题，训练链路稳定，LoRA 梯度正常。预热后训练窗口吞吐与 profiler 纯训练窗口一致，说明报告的 WPS/throughput 未被 init 阶段污染。端到端 `npu-smi` AICORE 均值偏低，反映短序列 LoRA 多模态 SFT 的整体系统利用率；profiler 显示 AIC/MIX 矩阵类 kernel 的 cube utilization 为 69.63%，但整体 step 仍受通信和大量 vector/slice 类算子影响。

---

## 2. Data Distribution Fix

### 2.1 修复内容

原始数据的 `<3s` 占比约 30.7%，低于团队目标约 40%。根因是两个 AED 短音频子集虽然配比合计 40%，但旧采样器允许 AED_event_2 产生较多 3s 以上样本。

本次修复保留 7 子集比例不变，调整采样模型：

| 子集 | 修正 |
|---|---|
| AED_event_2 | 截断到 2.95s，median=1.9, sigma=0.65 |
| AED_event_0 | 截断到 2.95s，median=0.8, sigma=0.65 |
| aishell1 | median 从 4.3 上调到 6.0，避免全局 p50/mean 被短音频拉低 |
| mulv18/pretrain_cap | 上限从 20s 放宽到 22s，贴近 p90/p95 |

修复脚本：`/data/sejin/baseline_26/scripts/gen_audio_dist_data.py`

### 2.2 修正后数据

| 指标 | 实际值 | 团队目标 |
|---|---:|---:|
| 总样本 | 3,200 | - |
| 音频样本 | 3,026 | - |
| 纯文本样本 | 174 | - |
| 总音频时长 | 4.856 h | - |
| p5 | 0.691s | 0.5s |
| p25 | 2.044s | 2.2s |
| p50 | 5.000s | 5.0s |
| mean | 5.777s | 6.0s |
| p75 | 8.766s | 9.9s |
| p90 | 11.239s | 11.3s |
| p95 | 14.170s | 14.2s |
| max | 22.000s | 226.7s, Whisper 截断到 30s |
| `<3s` 占比 | 41.14% | 约 40% |

数据路径：`/data/sejin/baseline_26/data_audio_distfix_3200/`

---

## 3. Training Configuration

| 项 | 配置 |
|---|---|
| model_id | `qwen3_5_audio` |
| 基座模型 | `/mnt/shared_data_196/sejin/models/Qwen3.5-35B-A3B` |
| DCP load | `/mnt/shared_data_196/sejin/models/Qwen3.5-35B-A3B-audio-dcp` |
| Whisper encoder | `/mnt/shared_data_196/sejin/models/whisper-large-v3` |
| 并行 | FSDP2, TP=1, EP=1, 8 卡全分片 |
| recompute | `model.language_model.layers.{*}` |
| precision | bf16 param, fp32 reduce |
| micro batch | 1 |
| grad accumulation | 4 |
| global batch | 32 |
| train_iters | 60 |
| learning rate | 1e-4 cosine |
| LoRA target | `self_attn.q_proj/k_proj/v_proj/o_proj` |
| LoRA rank/alpha/dropout | 16 / 32 / 0.05 |
| Trainable params | 3,440,640, trainable ratio 0.01% |

注意：本次保持与前一轮 LoRA 实验一致，仅修复数据分布和监控采集；`audio_projector` 仍未作为 trainable base 参数进入 optimizer。

---

## 4. Phase Timing

阶段耗时来自训练日志时间戳和训练引擎 step time。统计性能时只采用预热后的训练窗口，不把 init、数据处理、DCP load、首步编译/缓存和 checkpoint 写入纳入 WPS/throughput。

| 阶段 | 起止依据 | 耗时 |
|---|---|---:|
| 进程启动到数据准备入口 | first log 16:14:43 -> `Prepare data` 16:14:57 | 14.0s |
| 数据准备/tokenizer + DCP load 前准备 | `Prepare data` 16:14:57 -> DCP loaded 16:16:13 | 76.0s |
| DCP loaded 到首个 iteration 日志 | DCP loaded 16:16:13 -> iter1 log 16:16:30 | 17.0s |
| 首步预热/编译/缓存 | iter1 measured step time | 17.018s |
| 预热后主统计窗口 | step10-60 measured sum | 352.388s |
| 预热后 profiler 对齐窗口 | step20-60 measured sum | 283.093s |
| checkpoint 保存 | iter60 log 16:23:18 -> save log 16:23:19 | 1.0s |
| 训练日志首行到 checkpoint 保存 | first log 16:14:43 -> save log 16:23:19 | 516.0s |
| npu-smi 监控覆盖 | monitor JSON `duration_s` | 572.8s |

说明：`npu-smi` 监控覆盖时间长于训练日志首行到保存时间，包含脚本 sleep、启动、收尾和监控进程采样尾部；因此硬件端到端均值不能直接等同纯训练利用率。

---

## 5. Training Stability

| Step | Loss | Grad Norm | Step Time |
|---:|---:|---:|---:|
| 1 | 11.5393 | 0.916 | 17.018s |
| 10 | 10.5501 | 5.645 | 6.627s |
| 20 | 7.7913 | 11.302 | 6.702s |
| 30 | 6.4165 | 10.977 | 7.550s |
| 40 | 5.6277 | 7.193 | 7.620s |
| 50 | 5.2816 | 6.820 | 6.664s |
| 60 | 5.2165 | 6.381 | 6.652s |

稳定性判断：

| 项 | 判断 |
|---|---|
| loss | 单调趋势明显下降，60 step 降幅 54.79% |
| grad | 非零且整体稳定，step 55 有一次 25.010 峰值但未造成 NaN/失败 |
| NaN / skipped | 日志未见 NaN/skipped |
| checkpoint | step 60 成功保存 LoRA adapter |

Checkpoint:

`/data/sejin/baseline_26/output/ckpt_audio_distfix60/lora_adapter_iteration_60.safetensors`

MD5: `9f5bee12917349d4a1887874230019a7`

---

## 6. Throughput / WPS

### 6.1 Step Time

| 口径 | 样本数 | Mean | Median | P90 | P95 | Samples/s | 是否用于主报告 |
|---|---:|---:|---:|---:|---:|---:|---|
| all step 1-60 | 60 | 7.083s | 6.718s | 7.566s | 7.629s | 4.518 | 否，包含首步预热 |
| post-warmup step 2-60 | 59 | 6.915s | 6.712s | 7.562s | 7.621s | 4.628 | 参考 |
| post-warmup step 5-60 | 56 | 6.910s | 6.718s | 7.556s | 7.593s | 4.631 | 参考 |
| post-warmup step 10-60 | 51 | 6.910s | 6.711s | 7.562s | 7.602s | 4.631 | 主口径 |
| post-warmup step 20-60 | 41 | 6.905s | 6.705s | 7.550s | 7.585s | 4.635 | profiler 对齐参考 |

推荐对外主口径：`post-warmup step 10-60`。原因是 step1 明显包含编译/缓存预热，step2 之后已稳定；选择 step10-60 更符合业内性能报告中跳过 warmup 的保守口径。

### 6.2 Token Statistics

来自训练 cache 的实际 token 统计：

| 指标 | Mean/sample | P50 | P95 | Max | Total |
|---|---:|---:|---:|---:|---:|
| input tokens | 173.639 | 154.0 | 384.1 | 601 | 555,644 |
| label tokens | 19.502 | - | - | - | 62,407 |
| audio-pad tokens | 136.597 | - | - | - | 437,110 |

### 6.3 WPS

按 `post-warmup step 10-60` 计算：

| 口径 | WPS |
|---|---:|
| 实际 input-token WPS | 804.2 token/s |
| label-token WPS | 90.3 token/s |
| audio-pad token WPS | 632.6 token/s |

说明：这里的 WPS 是 processed tokens/s，不是自然语言“词/秒”。多模态训练建议同时报告 input-token WPS 与 audio-pad token WPS。上述 WPS 已排除 init、tokenizer、DCP load、首步预热和 checkpoint。

---

## 7. Hardware Metrics

### 7.1 npu-smi 端到端指标

监控方式：`/data/sejin/baseline_26/scripts/npu_monitor.py` 调用 `npu-smi info` 低频采样，覆盖脚本启动、预处理、训练、checkpoint 和收尾，因此它是端到端系统口径，不是纯算子口径。

| 指标 | Mean | Peak | 备注 |
|---|---:|---:|---|
| AICORE | 9.5% | 33.0% | npu-smi 全程低频采样，含非训练空隙 |
| HBM used | 32,513 MB/卡 | 33,366 MB/卡 | 约 49.6% mean, 50.9% peak |
| Power | 158.3 W/卡 | 192.2 W/卡 | 约 1.27 kW mean for 8 cards, per-card mean x8 |

### 7.2 Profiler step20-22 纯训练窗口

Profiler 运行配置为 24 step，在已过预热的 step20-22 窗口采集。`step_trace_time.csv` 中记录为 Step 19/20/21，为 profiler 内部 step index 偏移；对应训练日志 step20-22 附近的纯训练窗口。训练日志中 step22 的 `162683.5 ms` 包含 profiler 在线解析开销，因此不用于吞吐/WPS。

| 指标 | 结果 |
|---|---:|
| captured steps | profiler Step 19/20/21 |
| avg stage time | 6.9365s |
| avg pure-window throughput | 4.613 samples/s |
| avg computing time | 3.1877s |
| computing / stage | 45.96% |
| communication not overlapped | 3.5843s |
| communication not overlapped / stage | 51.67% |
| total communication | 4.5128s |
| total communication / stage | 65.06% |
| free / stage | 2.37% |
| preparing avg | 0.0032s |
| HBM bandwidth | read 72,669 MB/s, write 54,545 MB/s |
| profiler NPU memory | mean 41,296 MB, peak 52,858 MB on device 0 |

Profiler AICORE/kernel 侧指标：

| 口径 | Duration share | Weighted utilization |
|---|---:|---:|
| AIC/MIX kernel duration share in all kernel rows | 13.38% | cube utilization 69.63% |
| MIX_AIC only | 10.93% | cube utilization 66.45% |
| AI_CORE only | 2.45% | cube utilization 83.84% |
| AI_VECTOR_CORE duration share | 29.50% | vector ratio 4.88% |
| AI_CORE + MIX_AIC + AI_VECTOR_CORE | 42.88% | cube utilization 21.72% |

Top operators by profiler total time:

| Rank | Op | Core Type | Ratio |
|---:|---|---|---:|
| 1 | Slice | AI_VECTOR_CORE | 36.53% |
| 2 | GroupedMatmul | MIX_AIC | 14.87% |
| 3 | Cast | AI_VECTOR_CORE | 9.10% |
| 4 | ForeachCopy | AI_VECTOR_CORE | 4.51% |
| 5 | MatMulV2 | AI_CORE | 4.51% |
| 6 | TransData | AI_VECTOR_CORE | 4.24% |
| 7 | prepare_wy_repr_bwd_kernel | MIX_AIC | 2.51% |

解读：

| 现象 | 判断 |
|---|---|
| npu-smi AICORE 低、profiler AIC/MIX cube 高 | 两者口径不同。npu-smi 是端到端低频采样，profiler 是训练窗口内算子加权。正式报告应并列展示，不应混用。 |
| profiler 计算占比约 46%、非重叠通信约 52% | 当前 FSDP2 + LoRA + 短序列 workload 中通信/调度占比较高，纯矩阵核利用率不能代表整个 step 利用率。 |
| Slice/Cast/TransData 占比高 | 短序列、多模态 padding/格式转换、FSDP/recompute 会放大小算子和数据搬运占比。 |
| HBM 约半卡到 52.9GB 峰值 | 35B 通过 FSDP2 分片 + recompute 可稳定放入 8 卡，但 profiler 峰值高于 npu-smi 均值，符合短窗口精细采样预期。 |

---

## 8. Best-Practice Assessment

| 维度 | 当前状态 | 评价 |
|---|---|---|
| 数据分布 | `<3s` 修到 41.14%，p50/p90/p95 贴近目标 | 达标 |
| 数据真实性 | 合成波形 + 合成文本 | 只能代表性能冒烟，不代表真实 ASR 质量 |
| 训练稳定性 | loss 下降，grad 非零，checkpoint 正常 | 达标 |
| 吞吐口径 | 明确 post-warmup step10-60 主口径，排除 init/checkpoint | 达标 |
| WPS 口径 | input-token、label-token、audio-pad token 分开报告 | 达标 |
| 硬件指标 | npu-smi 端到端 + profiler step20-22 纯训练窗口并列 | 达标 |
| AICORE 报告 | 提供 AIC/MIX duration-weighted cube utilization 和端到端 AICORE | 达标 |
| 可复现性 | 数据脚本、配置、启动脚本、日志、JSON、profiler CSV 均保存 | 达标 |
| 多模态真实性 | Whisper 路径真实，但 projector 未训练 | 部分达标 |

业内标准报告口径建议：

| 报告项 | 当前采用方式 |
|---|---|
| Warmup exclusion | step1 排除，主口径 step10-60 |
| Pure training window | profiler step20-22 |
| E2E vs profiler | `npu-smi` 与 profiler 分表并列，不混算 |
| Throughput | samples/s + token WPS 同时报告 |
| Memory | npu-smi HBM mean/peak + profiler memory peak |
| Power | npu-smi per-card mean/peak，注明端到端口径 |
| Traceability | 所有原始日志、JSON、CSV 路径列入 artifacts |

---

## 9. Artifacts

| 类型 | 路径 |
|---|---|
| 修正数据 | `/data/sejin/baseline_26/data_audio_distfix_3200/` |
| 数据生成脚本 | `/data/sejin/baseline_26/scripts/gen_audio_dist_data.py` |
| 60-step 启动脚本 | `/data/sejin/baseline_26/scripts/run_audio_distfix60.sh` |
| 60-step 配置 | `/data/sejin/third_party/mindspeed-mm-26.0.0/examples/qwen3_5_audio/distfix60_config.yaml` |
| 训练日志 | `/data/sejin/baseline_26/logs/audio_distfix60_20260615_161439.log` |
| 监控 JSON | `/data/sejin/baseline_26/metrics/audio_distfix60_20260615_161439_npu.json` |
| 监控摘要日志 | `/data/sejin/baseline_26/logs/audio_distfix60_20260615_161439_monitor.log` |
| Profiler 启动脚本 | `/data/sejin/baseline_26/scripts/run_audio_distfix_profile_step20_22.sh` |
| Profiler 配置 | `/data/sejin/third_party/mindspeed-mm-26.0.0/examples/qwen3_5_audio/distfix_profile_step20_22_config.yaml` |
| Profiler 日志 | `/data/sejin/baseline_26/logs/audio_distfix_profile_step20_22_20260615_164823.log` |
| Profiler 输出目录 | `/data/sejin/baseline_26/profiling/audio_distfix_step20_22` |
| Profiler step trace | `/data/sejin/baseline_26/profiling/audio_distfix_step20_22/task1-910B-155_3373789_20260615165200144_ascend_pt/ASCEND_PROFILER_OUTPUT/step_trace_time.csv` |
| Profiler kernel details | `/data/sejin/baseline_26/profiling/audio_distfix_step20_22/task1-910B-155_3373789_20260615165200144_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv` |
| Profiler op statistic | `/data/sejin/baseline_26/profiling/audio_distfix_step20_22/task1-910B-155_3373789_20260615165200144_ascend_pt/ASCEND_PROFILER_OUTPUT/op_statistic.csv` |
| Profiler HBM | `/data/sejin/baseline_26/profiling/audio_distfix_step20_22/task1-910B-155_3373789_20260615165200144_ascend_pt/PROF_000001_20260615165200145_FEQDGLBKHMJDDEAC/mindstudio_profiler_output/hbm_20260615165407.csv` |
| LoRA checkpoint | `/data/sejin/baseline_26/output/ckpt_audio_distfix60/lora_adapter_iteration_60.safetensors` |

---

## 10. Final Recommendations

1. 若要代表真实多模态训练质量，下一轮应使用真实音频/真实转写，或至少 TTS 语义一致数据。
2. 若目标是音频适配能力，应显式让 `audio_projector` 参与训练，或给 projector 添加 LoRA/解冻策略；当前只有 LLM self-attn LoRA。
3. 若目标是硬件峰值性能，应单独跑长序列/更高 packed token 数 workload，当前平均 174 input tokens/sample 太短。
4. 正式提交性能时，建议采用本报告的两套并列硬件口径：`npu-smi` 端到端 AICORE/HBM/功耗 + profiler step20-22 纯训练窗口 AIC/MIX cube utilization、step trace、HBM bandwidth。
