# ep8_mbs1_ga4_rc_off_pad1408_lora64_nonexpert_nosync Performance Analysis

- Config: `/data/sejin/third_party/mindspeed-mm-26.0.0/examples/qwen3_5_audio/perf_tuning/ep8_mbs1_ga4_rc_off_pad1408_lora64_nonexpert_nosync.yaml`
- Train log: `/data/sejin/baseline_26/logs/ep8_mbs1_ga4_rc_off_pad1408_lora64_nonexpert_nosync_20260616_013845.log`
- Monitor JSON: `/data/sejin/baseline_26/metrics/ep8_mbs1_ga4_rc_off_pad1408_lora64_nonexpert_nosync_20260616_013845_npu_full.json`
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
| pad_to_multiple_of | `1408` |
| chunk_loss_size | `1024` |
| num_workers | `8` |
| prefetch_factor | `None` |
| persistent_workers | `None` |
| dataloader_timeout | `None` |
| lora_rank | `64` |
| lora_alpha | `128` |
| lora_dropout | `0.05` |

## Runtime Env

| Item | Value |
|---|---:|
| MULTI_STREAM_MEMORY_REUSE | `None` |
| TASK_QUEUE_ENABLE | `None` |
| PYTORCH_NPU_ALLOC_CONF | `None` |
| ACLNN_CACHE_LIMIT | `None` |
| CPU_AFFINITY_CONF | `None` |
| HCCL_CONNECT_TIMEOUT | `None` |

## Post-Warmup Metrics

| Metric | Value |
|---|---:|
| measured steps | 70 |
| step time mean | 5.308 s |
| step time p50 / p90 / p95 | 5.213 / 5.555 / 5.714 s |
| samples/s | 6.029 |
| input WPS | 1044.5 |
| label WPS | 117.4 |
| audio-pad WPS | 822.0 |
| loss first -> last measured | 4.9416 -> 0.4156 |
| AICORE mean / peak | 21.92% / 40.0% |
| HBM mean / peak | 57522.7 / 59671 MB |
| Power mean / peak | 159.94 / 200.4 W |
| active-chip AICORE mean / p90 / peak | 21.92% / 31.0% / 40.0% |
| active-chip HBM mean / p90 / peak | 57522.7 / 59530 / 59671 MB |
| active-chip Power mean / p90 / peak | 159.94 / 173.8 / 200.4 W |
| error category | `none` |

## Average Step Phase Timing

| Phase | Mean ms | Share |
|---|---:|---:|
| backward | 2166.1 | 40.8% |
| clip | 59.9 | 1.1% |
| empty_cache | 0.0 | 0.0% |
| forward | 2791.3 | 52.6% |
| get_batch | 2.0 | 0.0% |
| loss_setup | 0.9 | 0.0% |
| lr_scheduler | 0.0 | 0.0% |
| move | 210.8 | 4.0% |
| optimizer | 23.0 | 0.4% |
| pregather | 0.0 | 0.0% |
| profiler | 0.0 | 0.0% |
| zero_grad | 1.6 | 0.0% |

## Run Phase Times

| Phase | Seconds |
|---|---:|
| startup_to_manual_load | 5.97 |
| qwen_safe_open_load | 34.12 |
| whisper_load | 1.75 |
| projector_and_final_barrier | 5.96 |
| post_load_to_iter1_end | 57.34 |
| step1_time | 25.45 |
| all_logged_steps | 444.85 |
| warmup_logged_steps | 73.29 |
| measured_logged_steps | 371.55 |
| last_step_to_lora_save | 1.88 |

## Per-Chip Hardware Window

| Chip | AICORE mean / peak | HBM mean / peak MB | Power mean / peak W |
|---:|---:|---:|---:|
| 0 | 21.04 / 39.0 | 57816.0 / 57918 | 159.61 / 196.5 |
| 1 | 21.96 / 39.0 | 57509.5 / 57510 | 164.57 / 200.4 |
| 2 | 22.60 / 38.0 | 59613.1 / 59671 | 159.55 / 193.4 |
| 3 | 21.63 / 38.0 | 56893.6 / 56909 | 158.83 / 192.5 |
| 4 | 22.25 / 39.0 | 56891.1 / 56893 | 155.64 / 193.7 |
| 5 | 22.31 / 40.0 | 57052.0 / 57107 | 163.98 / 193.2 |
| 6 | 21.26 / 37.0 | 56966.1 / 56968 | 159.78 / 195.0 |
| 7 | 22.28 / 38.0 | 57440.0 / 57508 | 157.57 / 191.7 |
