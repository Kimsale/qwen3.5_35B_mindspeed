# ep8_mbs2_ga2_rc_on_pad128_bucket_nosync_probe Performance Analysis

- Config: `/data/sejin/third_party/mindspeed-mm-26.0.0/examples/qwen3_5_audio/perf_tuning/ep8_mbs2_ga2_rc_on_pad128_bucket_nosync_probe.yaml`
- Train log: `/data/sejin/baseline_26/logs/ep8_mbs2_ga2_rc_on_pad128_bucket_nosync_probe_20260615_232443.log`
- Monitor JSON: `/data/sejin/baseline_26/metrics/ep8_mbs2_ga2_rc_on_pad128_bucket_nosync_probe_20260615_232443_npu_full.json`
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
| recompute | `True` |
| param_dtype | `bf16` |
| reduce_dtype | `fp32` |
| micro_batch_size | `2` |
| gradient_accumulation_steps | `2` |
| gradient_accumulation_no_sync | `True` |
| train_iters | `40` |
| lr | `0.0001` |
| clip_grad | `1.0` |
| cutoff_len | `4096` |
| sampler_type | `LengthBucketBatchSampler` |
| length_bucket_size_multiplier | `32` |
| pad_to_multiple_of | `128` |
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
| measured steps | 4 |
| step time mean | 2.957 s |
| step time p50 / p90 / p95 | 2.775 / 3.486 / 3.618 s |
| samples/s | 10.822 |
| input WPS | 2561.3 |
| label WPS | 193.4 |
| audio-pad WPS | 2181.3 |
| loss first -> last measured | 11.3474 -> 11.3769 |
| AICORE mean / peak | 11.88% / 46.0% |
| HBM mean / peak | 35697.1 / 37197 MB |
| Power mean / peak | 141.52 / 185.8 W |

## Average Step Phase Timing

| Phase | Mean ms | Share |
|---|---:|---:|
| backward | 1321.3 | 44.7% |
| clip | 6.8 | 0.2% |
| forward | 1600.6 | 54.1% |
| get_batch | 0.8 | 0.0% |
| loss_setup | 0.5 | 0.0% |
| lr_scheduler | 0.0 | 0.0% |
| move | 6.4 | 0.2% |
| optimizer | 3.9 | 0.1% |
| pregather | 0.0 | 0.0% |
| profiler | 0.0 | 0.0% |
| zero_grad | 0.3 | 0.0% |

## Run Phase Times

| Phase | Seconds |
|---|---:|
| startup_to_manual_load | 3.53 |
| qwen_safe_open_load | 33.22 |
| whisper_load | 1.24 |
| projector_and_final_barrier | 3.83 |
| post_load_to_iter1_end | 55.84 |
| step1_time | 21.00 |
| all_logged_steps | 57.66 |

## Errors
- `Traceback (most recent call last):`
