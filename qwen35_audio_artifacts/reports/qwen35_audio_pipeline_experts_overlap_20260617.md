# Qwen3.5-35B Audio Pipeline Experts Overlap Performance Report

**生成时间**: 2026-06-17
**运行机器**: `172.29.226.188` / `task3-910B-188`
**worktree**: `/data/sejin/third_party/mindspeed-mm-pipeline-experts-overlap`
**分支**: `feat/qwen35-audio-pipeline-experts-overlap`
**提交**: `8ada09f36 Stabilize Qwen audio pipeline experts run config`
**模型/任务**: Qwen3.5-35B-A3B + Whisper-large-v3 audio encoder + LoRA 语音训练
**约束**: 不修改模型结构、专家数、MoE 路由数学语义和 Whisper encoder；只调整训练分布式策略与运行配置。

本报告只统计 warmup 后训练窗口，跳过前 10 step；init、safe_open 加载、Whisper 加载、首步编译和 LoRA 保存不计入主性能均值。

---

## 1. 代码状态

当前分支已提交，worktree clean。关键提交如下：

| Commit | 内容 |
|---|---|
| `c90023bc` | 实现 Qwen audio Pipeline Experts overlap：新增 pipeline dispatcher、chunk/multi-stream A2A dispatch/combine、相关参数和 Qwen MoE 接入。 |
| `f40cb3c6` | 修复自定义模型配置 dtype 兼容。 |
| `8ada09f36` | 稳定 mbs1/pad1536 运行配置，并修复 `_AsyncA2AWork.wait()` 等待共享通信 stream 导致后续 chunk 被串行阻塞的问题。 |

主要文件：

| 文件 | 作用 |
|---|---|
| `/data/sejin/third_party/mindspeed-mm-pipeline-experts-overlap/mindspeed_mm/fsdp/distributed/expert_parallel/ep_dispatcher.py` | pipeline dispatcher、chunk dispatch/combine、多 stream overlap 和 MoE phase timing。 |
| `/data/sejin/third_party/mindspeed-mm-pipeline-experts-overlap/mindspeed_mm/fsdp/distributed/expert_parallel/expert_parallel.py` | dispatcher 选择与 forward 接入。 |
| `/data/sejin/third_party/mindspeed-mm-pipeline-experts-overlap/mindspeed_mm/fsdp/models/qwen3_5_moe/modeling_qwen3_5_moe.py` | Qwen3.5 MoE forward 接入 pipeline dispatcher。 |
| `/data/sejin/third_party/mindspeed-mm-pipeline-experts-overlap/mindspeed_mm/fsdp/params/parallel_args.py` | 新增 pipeline dispatcher 参数。 |
| `/data/sejin/third_party/mindspeed-mm-pipeline-experts-overlap/examples/qwen3_5_audio/perf_tuning/mbs1_pipeline_pad1536_nosync_80.yaml` | 本次稳定训练配置。 |

---

## 2. 训练运行

| 项 | 值 |
|---|---:|
| Run tag | `qwen35_audio_pipeline_mbs1_pad1536_ep8_8ada09f36_20260617_110652` |
| 状态 | 成功完成 `80/80`，`TRAIN_RC=0`，`ANALYSIS_RC=0` |
| 训练配置 | `/data/sejin/third_party/mindspeed-mm-pipeline-experts-overlap/examples/qwen3_5_audio/perf_tuning/mbs1_pipeline_pad1536_nosync_80.yaml` |
| 训练日志 | `/data/sejin/baseline_26/logs/qwen35_audio_pipeline_mbs1_pad1536_ep8_8ada09f36_20260617_110652.log` |
| NPU monitor | `/data/sejin/baseline_26/metrics/qwen35_audio_pipeline_mbs1_pad1536_ep8_8ada09f36_20260617_110652_npu.json` |
| 自动分析 JSON | `/data/sejin/baseline_26/reports/perf_runs/qwen35_audio_pipeline_mbs1_pad1536_ep8_8ada09f36_20260617_110652.json` |
| 自动分析 MD | `/data/sejin/baseline_26/reports/perf_runs/qwen35_audio_pipeline_mbs1_pad1536_ep8_8ada09f36_20260617_110652.md` |

本次结束后检查 `npu-smi info`，8 张 910B3 均无训练进程残留，AICORE 为 0%，机器已回到空闲状态。

---

## 3. 配置快照

| 项 | 值 |
|---|---:|
| EP | `8` |
| TP / Ulysses | `1 / 1` |
| FSDP | `fully_shard_parallel_size: auto` |
| dispatcher | `pipeline` |
| pipeline chunks | `4` |
| pipeline multi stream | `true` |
| pipeline min tokens per chunk | `1024` |
| micro batch / GA | `1 / 4` |
| global batch | `32` |
| pad_to_multiple_of | `1536` |
| chunk_loss_size | `1024` |
| recompute | `false` |
| gradient_accumulation_no_sync | `true` |
| grouped expert matmul | `true` |
| dtype | `param_dtype=bf16`, `reduce_dtype=fp32` |
| LoRA | rank `16`, alpha `32`, dropout `0.05` |
| train_iters | `80` |

关键运行环境：

| 环境变量 | 值 |
|---|---:|
| `NON_MEGATRON` | `true` |
| `MULTI_STREAM_MEMORY_REUSE` | `2` |
| `TASK_QUEUE_ENABLE` | `2` |
| `PYTORCH_NPU_ALLOC_CONF` | `expandable_segments:True` |
| `TORCH_DEVICE_BACKEND_AUTOLOAD` | `0` |
| `AUDIO_PLACEHOLDER` | `<\|AUDIO\|>` |
| `MOE_PHASE_TIMING` | `1` |

---

## 4. Warmup 后性能

统计窗口：step 11-80，共 70 step。

| 指标 | 本次 Pipeline Experts |
|---|---:|
| step time mean | 8.589 s |
| step time p50 / p90 / p95 | 8.486 / 9.092 / 9.279 s |
| samples/s | 3.726 |
| input WPS | 645.5 |
| label WPS | 72.6 |
| audio-pad WPS | 508.0 |
| loss first -> last measured | 11.2700 -> 4.7347 |
| AICORE mean / peak | 12.04% / 42.0% |
| HBM mean / peak | 56,581.7 / 58,286 MB |
| Power mean / peak | 135.28 / 182.9 W |
| error category | `none` |

阶段耗时：

| Phase | Mean ms | Share |
|---|---:|---:|
| forward | 5727.8 | 66.7% |
| backward | 2624.9 | 30.6% |
| move | 152.3 | 1.8% |
| clip | 31.3 | 0.4% |
| get_batch | 2.0 | 0.0% |
| optimizer | 3.9 | 0.0% |

MoE EP phase timing：

| MoE phase | Mean ms | P90 ms |
|---|---:|---:|
| pipeline_wait_dispatch | 10.418 | 10.478 |
| pipeline_compute | 7.004 | 7.402 |
| permute_post_a2a | 6.094 | 6.485 |
| dispatch_preprocess | 3.674 | 4.268 |
| pipeline_start_combine | 1.216 | 1.237 |
| gmm_fc1 | 0.389 | 0.395 |
| gmm_fc2 | 0.302 | 0.306 |
| unpermute_final | 0.285 | 0.287 |
| pipeline_wait_combine | 0.068 | 0.068 |

Run phase timing，以下不计入主性能均值：

| Phase | Seconds |
|---|---:|
| startup_to_manual_load | 3.24 |
| qwen_safe_open_load | 31.85 |
| whisper_load | 1.27 |
| projector_and_final_barrier | 6.48 |
| post_load_to_iter1_end | 44.07 |
| step1_time | 27.81 |
| all_logged_steps | 705.51 |
| warmup_logged_steps | 104.29 |
| measured_logged_steps | 601.22 |
| last_step_to_lora_save | 0.10 |

---

## 5. 与 EP8 LLM Pack 最优基线对比

本轮主目标按 WPS 排序，不再以严格 HBM 55-60GB 作为主比较口径。对比基线改为 EP8 LLM 改为 pack 后的最优配置，数据来自 GitHub 报告 `pack_format_validation_report.md`：<https://github.com/Kimsale/qwen3.5_35B_mindspeed/blob/feat/llm-pad-to-pack-recompute/pack_format_validation_report.md>。该报告显示 `pack rc_off, mbs=1` 稳定完成 80 step，last 40 steps 平均 `2111.4 WPS`、约 `3.6s/iter`、约 `40GB/card HBM`。同报告中的 `pack rc_on` 为 `1475.3 WPS`、约 `33GB/card HBM`，以约 30% WPS 换 7GB 显存。

| 指标 | EP8 LLM pack 最优基线 | Pipeline Experts `mbs1/pad1536` | 变化 |
|---|---:|---:|---:|
| 目标口径 | WPS 优先 | WPS 优先 | 一致 |
| input WPS | 2111.4 | 645.5 | -69.4%，仅为基线 30.6% |
| 相对吞吐 | 1.00x | 0.31x | pack 基线约 3.27x 更快 |
| step time mean | ~3.6 s | 8.589 s | Pipeline 慢约 2.39x |
| AICORE mean / peak | 未记录 | 12.04% / 42.0% | Pipeline 实测偏低 |
| HBM mean / peak | ~40GB/card | 56,581.7 / 58,286 MB | Pipeline 多占约 16.6GB/card |
| Power mean / peak | 未记录 | 135.28 / 182.9 W | Pipeline 实测偏低 |

补充参考：如果只在 manual EP8 非 pack 配置内部比较，之前最高完成 run 是 `ep8_mbs1_ga4_rc_off_pad1024_pregather_nosync`，input WPS 为 1414.6；`pad1280_current` 为 1295.8；严格 HBM 55-60GB 的 `pad1536_nosync_rerun05` 为 1132.5。这些都低于 EP8 LLM pack 最优基线 2111.4，因此 `pad1536` 不再作为主基线。

结论：本次 Pipeline Experts overlap 在 `mbs1/ga4/pad1536` 下稳定完成训练，但 WPS 只有 645.5，明显低于 EP8 LLM pack 最优基线 2111.4。按当前主目标“提高 WPS”，该实现是性能回退，不建议作为优化方向继续放大。

---

## 6. 现象解释

1. **chunk 后专家计算太小，overlap 收益不足。**
   本次 `pipeline_chunks=4`，但 MoE phase timing 中 `pipeline_compute` 只有约 7.0 ms，`pipeline_wait_dispatch` 约 10.4 ms。通信等待没有被专家计算覆盖，反而暴露在 critical path。

2. **forward 成为主要回归来源。**
   Pipeline run 的 forward 平均 5727.8 ms；它不仅低于 EP8 LLM pack 最优 WPS 基线，也低于 manual EP8 非 pack 的高 WPS 配置。新增 pipeline 调度、chunk 化、A2A work 管理和额外同步开销超过了 overlap 收益。

3. **HBM 不是主优化目标，WPS 才是主目标。**
   Pipeline run 的 HBM 在 56-58GB，但 WPS 只有 645.5。当前结果说明“占满更多 HBM”不能直接转化为吞吐，后续比较应优先看 pack 基线 2111.4 以及 post-warmup WPS。

4. **mbs2 方向仍不稳定。**
   之前 `mbs2_fa2_pipeline_bucket64_chunk512_80.yaml` 在修复 `<|AUDIO|>` placeholder 后能进入训练，但 step 24 后挂起。pack validation 报告也指出 `mbs=2` 的 pack 路径会在 FSDP2 lazy initialization 阶段挂住，根因是 pack 后各 rank 变长序列不对齐，需要 collator 做 rank 对齐。因此本次最终采用 mbs1 稳定口径记录性能。

---

## 7. 结论和建议

本次任务已完成：新 worktree 分支已提交，代码已在 172.29.226.188 上启动训练并成功完成 80 step，warmup 后性能、训练日志、NPU monitor、MoE phase timing 和 run phase timing 均已记录。

当前 Pipeline Experts overlap 实现不建议替换生产默认配置。若目标是 WPS，应以 EP8 LLM pack 最优配置 `ep8_pack_188.yaml` / `pack rc_off, mbs=1` 作为主基线继续优化；manual EP8 非 pack 内部的可参考上限是 `pad1024_pregather_nosync` 的 1414.6 WPS，严格 HBM 配置 `pad1536_nosync` 只保留为显存约束参考，不作为主目标。

后续若继续沿 Pipeline Experts 方向优化，优先做更小风险的 ablation：

| 优先级 | 方向 | 判断标准 |
|---|---|---|
| P0 | `pipeline_chunks=2` 对比当前 `4` | 如果 WPS 明显上升且 forward 开销下降，再继续。 |
| P0 | `pipeline_chunks=1` 或 pipeline dispatcher 但不切 chunk | 分离 dispatcher 框架开销与 chunk/multi-stream 开销。 |
| P1 | 提高 `pipeline_min_tokens_per_chunk` | 避免 GMM 被切得过碎。 |
| P1 | 只在 token 数超过阈值时启用 pipeline dispatcher | 小 batch/短 token 直接走原 eager dispatcher。 |
| P2 | mbs2 稳定化专项 | 需要单独定位 step24 挂起/OOM，不能作为当前稳定性能口径。 |

在当前数据和单机 EP8 训练口径下，Pipeline Experts 的实测结果是稳定但 WPS 回退；相对 EP8 LLM pack 最优基线 2111.4，本次 645.5 WPS 只有 30.6%。后续优化应优先围绕 pack/有效 token 密度和更高 WPS 展开，而不是继续以 HBM 55-60GB 为主目标调 padding。
