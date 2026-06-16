# ep8_mbs1_ga4_rc_off_pad1536_nosync_rerun05 Performance Analysis

- Config: `/data/sejin/third_party/mindspeed-mm-26.0.0/examples/qwen3_5_audio/perf_tuning/ep8_mbs1_ga4_rc_off_pad1536_nosync.yaml`
- Train log: `/data/sejin/baseline_26/logs/ep8_mbs1_ga4_rc_off_pad1536_nosync_rerun05_20260616_010537.log`
- Monitor JSON: `/data/sejin/baseline_26/metrics/ep8_mbs1_ga4_rc_off_pad1536_nosync_rerun05_20260616_010537_npu_full.json`
- Warmup skipped steps: `10`

## Config Snapshot

| Item | Value |
|---|---:|
| model_id | `qwen3_5_audio_manual_ep` |
| attn_implementation | `sdpa` |
| expert_parallel_size | `8` |
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
| num_workers | `8` |
| prefetch_factor | `None` |
| persistent_workers | `None` |
| dataloader_timeout | `None` |
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
| step time mean | 4.895 s |
| step time p50 / p90 / p95 | 4.855 / 4.888 / 5.170 s |
| samples/s | 6.537 |
| input WPS | 1132.5 |
| label WPS | 127.3 |
| audio-pad WPS | 891.3 |
| loss first -> last measured | 11.2700 -> 4.6936 |
| AICORE mean / peak | 23.43% / 43.0% |
| HBM mean / peak | 56400.7 / 58059 MB |
| Power mean / peak | 164.54 / 205.6 W |
| active-chip AICORE mean / p90 / peak | 23.43% / 32.0% / 43.0% |
| active-chip HBM mean / p90 / peak | 56400.7 / 57996 / 58059 MB |
| active-chip Power mean / p90 / peak | 164.54 / 180.4 / 205.6 W |
| error category | `none` |

## Average Step Phase Timing

| Phase | Mean ms | Share |
|---|---:|---:|
| backward | 1616.2 | 33.0% |
| clip | 139.2 | 2.8% |
| empty_cache | 0.0 | 0.0% |
| forward | 2662.7 | 54.4% |
| get_batch | 1.8 | 0.0% |
| loss_setup | 0.9 | 0.0% |
| lr_scheduler | 0.0 | 0.0% |
| move | 441.9 | 9.0% |
| optimizer | 3.9 | 0.1% |
| pregather | 0.0 | 0.0% |
| profiler | 0.0 | 0.0% |
| zero_grad | 0.3 | 0.0% |

## Run Phase Times

| Phase | Seconds |
|---|---:|
| startup_to_manual_load | 3.45 |
| qwen_safe_open_load | 29.81 |
| whisper_load | 1.27 |
| projector_and_final_barrier | 13.50 |
| post_load_to_iter1_end | 39.08 |
| step1_time | 24.96 |
| all_logged_steps | 411.55 |
| warmup_logged_steps | 68.87 |
| measured_logged_steps | 342.68 |
| last_step_to_lora_save | 0.10 |

## Per-Chip Hardware Window

| Chip | AICORE mean / peak | HBM mean / peak MB | Power mean / peak W |
|---:|---:|---:|---:|
| 0 | 23.90 / 43.0 | 56202.2 / 56207 | 166.11 / 201.7 |
| 1 | 23.13 / 42.0 | 56275.6 / 56298 | 168.93 / 201.2 |
| 2 | 24.66 / 41.0 | 58039.2 / 58059 | 163.66 / 194.5 |
| 3 | 22.98 / 39.0 | 56352.7 / 56353 | 163.03 / 200.4 |
| 4 | 23.15 / 41.0 | 56186.0 / 56335 | 160.06 / 195.6 |
| 5 | 23.53 / 41.0 | 55871.3 / 55872 | 168.36 / 205.6 |
| 6 | 22.60 / 38.0 | 55649.7 / 55853 | 163.77 / 198.1 |
| 7 | 23.50 / 43.0 | 56629.0 / 56673 | 162.40 / 199.3 |
