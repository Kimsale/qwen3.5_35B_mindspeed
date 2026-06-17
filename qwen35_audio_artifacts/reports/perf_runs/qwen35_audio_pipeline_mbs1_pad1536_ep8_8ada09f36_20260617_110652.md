# qwen35_audio_pipeline_mbs1_pad1536_ep8_8ada09f36_20260617_110652 Performance Analysis

- Config: `/data/sejin/third_party/mindspeed-mm-pipeline-experts-overlap/examples/qwen3_5_audio/perf_tuning/mbs1_pipeline_pad1536_nosync_80.yaml`
- Train log: `/data/sejin/baseline_26/logs/qwen35_audio_pipeline_mbs1_pad1536_ep8_8ada09f36_20260617_110652.log`
- Monitor JSON: `/data/sejin/baseline_26/metrics/qwen35_audio_pipeline_mbs1_pad1536_ep8_8ada09f36_20260617_110652_npu.json`
- Warmup skipped steps: `10`

## Config Snapshot

| Item | Value |
|---|---:|
| model_id | `qwen3_5_audio_manual_ep` |
| attn_implementation | `sdpa` |
| expert_parallel_size | `8` |
| ep_dispatcher | `pipeline` |
| fully_shard_parallel_size | `auto` |
| tensor_parallel_size | `1` |
| ulysses_parallel_size | `1` |
| recompute | `False` |
| param_dtype | `bf16` |
| reduce_dtype | `fp32` |
| micro_batch_size | `1` |
| gradient_accumulation_steps | `4` |
| gradient_accumulation_no_sync | `True` |
| empty_cache_interval | `0` |
| train_iters | `80` |
| lr | `0.0001` |
| clip_grad | `1.0` |
| cutoff_len | `4096` |
| sampler_type | `BaseRandomBatchSampler` |
| length_bucket_size_multiplier | `None` |
| pad_to_multiple_of | `1536` |
| chunk_loss_size | `1024` |
| use_grouped_expert_matmul | `True` |
| num_workers | `8` |
| prefetch_factor | `None` |
| persistent_workers | `None` |
| dataloader_timeout | `None` |
| perf_timing_sync | `False` |
| perf_timing_log_micro_steps | `False` |
| perf_timing_diagnostic_sync_phases | `[]` |
| lora_rank | `16` |
| lora_alpha | `32` |
| lora_dropout | `0.05` |

## Runtime Env

| Item | Value |
|---|---:|
| MULTI_STREAM_MEMORY_REUSE | `2` |
| TASK_QUEUE_ENABLE | `2` |
| PYTORCH_NPU_ALLOC_CONF | `expandable_segments:True` |
| ACLNN_CACHE_LIMIT | `100000` |
| CPU_AFFINITY_CONF | `1` |
| HCCL_CONNECT_TIMEOUT | `1800` |

## Post-Warmup Metrics

| Metric | Value |
|---|---:|
| measured steps | 70 |
| step time mean | 8.589 s |
| step time p50 / p90 / p95 | 8.486 / 9.092 / 9.279 s |
| samples/s | 3.726 |
| input WPS | 645.5 |
| label WPS | 72.6 |
| audio-pad WPS | 508.0 |
| loss first -> last measured | 11.2700 -> 4.7347 |
| AICORE mean / peak | 12.04% / 42.0% |
| HBM mean / peak | 56581.7 / 58286 MB |
| Power mean / peak | 135.28 / 182.9 W |
| active-chip AICORE mean / p90 / peak | 12.04% / 25.0% / 42.0% |
| active-chip HBM mean / p90 / peak | 56581.7 / 58269 / 58286 MB |
| active-chip Power mean / p90 / peak | 135.28 / 158.2 / 182.9 W |
| error category | `none` |

## WPS Baseline Comparison

Primary optimization target is WPS, not strict HBM 55-60GB. The comparison baseline is the best EP8 LLM pack configuration from `pack_format_validation_report.md`: <https://github.com/Kimsale/qwen3.5_35B_mindspeed/blob/feat/llm-pad-to-pack-recompute/pack_format_validation_report.md>. The baseline is `pack rc_off, mbs=1`, stable 80 steps, last-40-step average `2111.4 WPS`, `~3.6s/iter`, `~40GB/card HBM`.

| Metric | EP8 LLM pack best | This pipeline run | Delta |
|---|---:|---:|---:|
| input WPS | 2111.4 | 645.5 | -69.4%, 0.31x baseline |
| step time mean | ~3.6 s | 8.589 s | pipeline 2.39x slower |
| HBM mean / peak | ~40GB/card | 56581.7 / 58286 MB | pipeline +16.6GB/card |
| AICORE mean / peak | N/A | 12.04% / 42.0% | pipeline measured only |

The pipeline run is stable, but it does not improve the WPS target relative to the EP8 LLM pack baseline.

## Average Step Phase Timing

| Phase | Mean ms | Share |
|---|---:|---:|
| backward | 2624.9 | 30.6% |
| clip | 31.3 | 0.4% |
| empty_cache | 0.0 | 0.0% |
| forward | 5727.8 | 66.7% |
| get_batch | 2.0 | 0.0% |
| loss_setup | 1.1 | 0.0% |
| lr_scheduler | 0.0 | 0.0% |
| move | 152.3 | 1.8% |
| optimizer | 3.9 | 0.0% |
| pregather | 0.0 | 0.0% |
| profiler | 0.0 | 0.0% |
| zero_grad | 0.3 | 0.0% |

## MoE EP Phase Timing

- MoE timing windows: 24
- Last input_splits: `[1052, 1523, 1934, 439, 2684, 554, 2763, 1339]`
- Last output_splits: `[1052, 432, 1293, 2676, 2240, 453, 888, 1127]`

| MoE phase | Mean ms | P90 ms |
|---|---:|---:|
| dispatch_preprocess | 3.674 | 4.268 |
| gmm_fc1 | 0.389 | 0.395 |
| gmm_fc2 | 0.302 | 0.306 |
| permute_post_a2a | 6.094 | 6.485 |
| permute_pre_a2a | 0.070 | 0.071 |
| pipeline_compute | 7.004 | 7.402 |
| pipeline_start_combine | 1.216 | 1.237 |
| pipeline_wait_combine | 0.068 | 0.068 |
| pipeline_wait_dispatch | 10.418 | 10.478 |
| swiglu | 0.103 | 0.104 |
| unpermute_final | 0.285 | 0.287 |
| unpermute_pre_combine | 0.116 | 0.117 |

| Expert load metric | Mean | P90 |
|---|---:|---:|
| expert_counts_max | 5651.888 | 5766.870 |
| expert_counts_mean | 371.468 | 387.160 |
| expert_counts_nonzero | 27.605 | 28.704 |
| expert_counts_std | 1165.744 | 1203.391 |

## Run Phase Times

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

## Per-Chip Hardware Window

| Chip | AICORE mean / peak | HBM mean / peak MB | Power mean / peak W |
|---:|---:|---:|---:|
| 0 | 12.22 / 40.0 | 56125.1 / 56128 | 137.26 / 181.0 |
| 1 | 11.65 / 35.0 | 56478.0 / 56491 | 134.31 / 176.5 |
| 2 | 11.73 / 41.0 | 58281.4 / 58286 | 130.08 / 171.4 |
| 3 | 11.69 / 37.0 | 56563.9 / 56567 | 131.61 / 179.4 |
| 4 | 11.45 / 41.0 | 56359.2 / 56494 | 134.00 / 177.3 |
| 5 | 12.53 / 42.0 | 56102.1 / 56128 | 140.56 / 181.3 |
| 6 | 12.11 / 42.0 | 55902.1 / 56031 | 138.92 / 182.9 |
| 7 | 12.90 / 41.0 | 56842.2 / 56888 | 135.51 / 176.5 |
