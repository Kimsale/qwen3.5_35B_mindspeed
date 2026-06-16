# ep8_mbs1_ga4_rc_off_pad1024_pregather_nosync Performance Analysis

- Config: `/data/sejin/third_party/mindspeed-mm-26.0.0/examples/qwen3_5_audio/perf_tuning/ep8_mbs1_ga4_rc_off_pad1024_pregather_nosync.yaml`
- Train log: `/data/sejin/baseline_26/logs/ep8_mbs1_ga4_rc_off_pad1024_pregather_nosync_20260615_235959.log`
- Monitor JSON: `/data/sejin/baseline_26/metrics/ep8_mbs1_ga4_rc_off_pad1024_pregather_nosync_20260615_235959_npu_full.json`
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
| train_iters | `80` |
| lr | `0.0001` |
| clip_grad | `1.0` |
| cutoff_len | `4096` |
| sampler_type | `BaseRandomBatchSampler` |
| length_bucket_size_multiplier | `None` |
| pad_to_multiple_of | `1024` |
| num_workers | `8` |
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
| step time mean | 3.919 s |
| step time p50 / p90 / p95 | 3.875 / 3.971 / 4.202 s |
| samples/s | 8.165 |
| input WPS | 1414.6 |
| label WPS | 159.0 |
| audio-pad WPS | 1113.3 |
| loss first -> last measured | 11.2514 -> 4.7147 |
| AICORE mean / peak | 18.47% / 38.0% |
| HBM mean / peak | 48752.6 / 49773 MB |
| Power mean / peak | 163.06 / 197.7 W |

## Average Step Phase Timing

| Phase | Mean ms | Share |
|---|---:|---:|
| backward | 1524.8 | 38.9% |
| clip | 24.6 | 0.6% |
| forward | 2049.7 | 52.3% |
| get_batch | 1.9 | 0.0% |
| loss_setup | 0.9 | 0.0% |
| lr_scheduler | 0.0 | 0.0% |
| move | 81.9 | 2.1% |
| optimizer | 3.9 | 0.1% |
| pregather | 202.2 | 5.2% |
| profiler | 0.0 | 0.0% |
| zero_grad | 0.3 | 0.0% |

## Run Phase Times

| Phase | Seconds |
|---|---:|
| startup_to_manual_load | 3.61 |
| qwen_safe_open_load | 32.44 |
| whisper_load | 1.27 |
| projector_and_final_barrier | 4.61 |
| post_load_to_iter1_end | 54.75 |
| step1_time | 23.10 |
| all_logged_steps | 332.83 |
| last_step_to_lora_save | 0.10 |
