# ep8_mbs1_ga4_rc_off_pad1280_current Performance Analysis

- Config: `/data/sejin/third_party/mindspeed-mm-26.0.0/examples/qwen3_5_audio/perf_tuning/ep8_mbs1_ga4_rc_off_pad1280.yaml`
- Train log: `/data/sejin/baseline_26/logs/ep8_mbs1_ga4_rc_off_pad1280_current_20260616_021501.log`
- Monitor JSON: `/data/sejin/baseline_26/metrics/ep8_mbs1_ga4_rc_off_pad1280_current_20260616_021501_npu_full.json`
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
| gradient_accumulation_no_sync | `False` |
| empty_cache_interval | `0` |
| train_iters | `80` |
| lr | `0.0001` |
| clip_grad | `1.0` |
| cutoff_len | `4096` |
| sampler_type | `BaseRandomBatchSampler` |
| length_bucket_size_multiplier | `None` |
| pad_to_multiple_of | `1280` |
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
| step time mean | 4.279 s |
| step time p50 / p90 / p95 | 4.185 / 4.479 / 4.980 s |
| samples/s | 7.479 |
| input WPS | 1295.8 |
| label WPS | 145.7 |
| audio-pad WPS | 1019.7 |
| loss first -> last measured | 11.3314 -> 4.9514 |
| AICORE mean / peak | 23.58% / 38.0% |
| HBM mean / peak | 51925.4 / 53244 MB |
| Power mean / peak | 166.53 / 195.9 W |
| active-chip AICORE mean / p90 / peak | 23.58% / 33.0% / 38.0% |
| active-chip HBM mean / p90 / peak | 51925.4 / 53243 / 53244 MB |
| active-chip Power mean / p90 / peak | 166.53 / 182.1 / 195.9 W |
| error category | `none` |

## Average Step Phase Timing

| Phase | Mean ms | Share |
|---|---:|---:|
| backward | 1608.0 | 37.6% |
| clip | 49.3 | 1.2% |
| empty_cache | 0.0 | 0.0% |
| forward | 2405.8 | 56.2% |
| get_batch | 1.9 | 0.0% |
| loss_setup | 0.9 | 0.0% |
| lr_scheduler | 0.0 | 0.0% |
| move | 179.0 | 4.2% |
| optimizer | 3.9 | 0.1% |
| pregather | 0.0 | 0.0% |
| profiler | 0.0 | 0.0% |
| zero_grad | 0.3 | 0.0% |

## Run Phase Times

| Phase | Seconds |
|---|---:|
| startup_to_manual_load | 3.63 |
| qwen_safe_open_load | 38.13 |
| whisper_load | 1.30 |
| projector_and_final_barrier | 2.64 |
| post_load_to_iter1_end | 36.63 |
| step1_time | 23.90 |
| all_logged_steps | 361.29 |
| warmup_logged_steps | 61.79 |
| measured_logged_steps | 299.50 |
| last_step_to_lora_save | 0.10 |

## Per-Chip Hardware Window

| Chip | AICORE mean / peak | HBM mean / peak MB | Power mean / peak W |
|---:|---:|---:|---:|
| 0 | 21.50 / 38.0 | 51790.2 / 51794 | 167.42 / 194.7 |
| 1 | 23.45 / 37.0 | 51877.9 / 51904 | 170.88 / 195.9 |
| 2 | 27.37 / 38.0 | 53243.7 / 53244 | 168.48 / 194.1 |
| 3 | 26.73 / 37.0 | 51882.7 / 51883 | 168.04 / 193.7 |
| 4 | 23.82 / 37.0 | 51713.6 / 51843 | 161.29 / 187.5 |
| 5 | 23.87 / 38.0 | 51463.7 / 51482 | 168.41 / 194.8 |
| 6 | 21.50 / 36.0 | 51329.3 / 51521 | 165.49 / 195.1 |
| 7 | 20.43 / 36.0 | 52101.9 / 52182 | 162.24 / 189.7 |
