# ep8_mbs1_ga4_rc_off_pad1536_nosync_fa2 Performance Analysis

- Config: `/data/sejin/third_party/mindspeed-mm-26.0.0/examples/qwen3_5_audio/perf_tuning/ep8_mbs1_ga4_rc_off_pad1536_nosync_fa2.yaml`
- Train log: `/data/sejin/baseline_26/logs/ep8_mbs1_ga4_rc_off_pad1536_nosync_fa2_20260616_012408.log`
- Monitor JSON: `/data/sejin/baseline_26/metrics/ep8_mbs1_ga4_rc_off_pad1536_nosync_fa2_20260616_012408_npu_full.json`
- Warmup skipped steps: `10`

## Config Snapshot

| Item | Value |
|---|---:|
| model_id | `qwen3_5_audio_manual_ep` |
| attn_implementation | `flash_attention_2` |
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
| step time mean | 4.971 s |
| step time p50 / p90 / p95 | 4.887 / 5.195 / 5.413 s |
| samples/s | 6.437 |
| input WPS | 1115.2 |
| label WPS | 125.4 |
| audio-pad WPS | 877.6 |
| loss first -> last measured | 11.2634 -> 4.7430 |
| AICORE mean / peak | 22.22% / 43.0% |
| HBM mean / peak | 55822.2 / 56626 MB |
| Power mean / peak | 161.49 / 202.5 W |
| active-chip AICORE mean / p90 / peak | 22.22% / 31.7% / 43.0% |
| active-chip HBM mean / p90 / peak | 55822.2 / 56504 / 56626 MB |
| active-chip Power mean / p90 / peak | 161.49 / 177.6 / 202.5 W |
| error category | `none` |

## Average Step Phase Timing

| Phase | Mean ms | Share |
|---|---:|---:|
| backward | 1630.2 | 32.8% |
| clip | 133.2 | 2.7% |
| empty_cache | 0.0 | 0.0% |
| forward | 2740.6 | 55.1% |
| get_batch | 2.1 | 0.0% |
| loss_setup | 0.9 | 0.0% |
| lr_scheduler | 0.0 | 0.0% |
| move | 431.0 | 8.7% |
| optimizer | 3.9 | 0.1% |
| pregather | 0.0 | 0.0% |
| profiler | 0.0 | 0.0% |
| zero_grad | 0.3 | 0.0% |

## Run Phase Times

| Phase | Seconds |
|---|---:|
| startup_to_manual_load | 3.38 |
| qwen_safe_open_load | 38.05 |
| whisper_load | 1.34 |
| projector_and_final_barrier | 3.60 |
| post_load_to_iter1_end | 57.15 |
| step1_time | 24.71 |
| all_logged_steps | 416.83 |
| warmup_logged_steps | 68.83 |
| measured_logged_steps | 348.00 |
| last_step_to_lora_save | 0.10 |

## Per-Chip Hardware Window

| Chip | AICORE mean / peak | HBM mean / peak MB | Power mean / peak W |
|---:|---:|---:|---:|
| 0 | 22.25 / 42.0 | 56121.1 / 56155 | 162.61 / 199.6 |
| 1 | 21.70 / 43.0 | 55610.3 / 55625 | 165.26 / 202.5 |
| 2 | 23.11 / 41.0 | 56264.5 / 56265 | 159.01 / 192.8 |
| 3 | 21.76 / 43.0 | 55572.4 / 55604 | 160.16 / 193.7 |
| 4 | 22.22 / 42.0 | 55757.4 / 55768 | 156.82 / 196.5 |
| 5 | 22.10 / 43.0 | 55266.8 / 55284 | 164.71 / 200.4 |
| 6 | 21.61 / 42.0 | 55420.5 / 55462 | 161.88 / 197.5 |
| 7 | 22.98 / 42.0 | 56564.3 / 56626 | 161.43 / 194.7 |
