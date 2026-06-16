# ep8_mbs2_ga2_rc_off_pad128_bucket_tq1 Performance Analysis

- Config: `/data/sejin/third_party/mindspeed-mm-26.0.0/examples/qwen3_5_audio/perf_tuning/ep8_mbs2_ga2_rc_off_pad128_bucket.yaml`
- Train log: `/data/sejin/baseline_26/logs/ep8_mbs2_ga2_rc_off_pad128_bucket_tq1_20260615_220225.log`
- Monitor JSON: `/data/sejin/baseline_26/metrics/ep8_mbs2_ga2_rc_off_pad128_bucket_tq1_20260615_220225_npu_full.json`
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
| micro_batch_size | `2` |
| gradient_accumulation_steps | `2` |
| train_iters | `80` |
| lr | `0.0001` |
| clip_grad | `1.0` |
| cutoff_len | `4096` |
| sampler_type | `LengthBucketBatchSampler` |
| length_bucket_size_multiplier | `64` |
| pad_to_multiple_of | `128` |
| num_workers | `8` |
| lora_rank | `16` |
| lora_alpha | `32` |
| lora_dropout | `0.05` |

## Post-Warmup Metrics

| Metric | Value |
|---|---:|
| measured steps | 14 |
| step time mean | 2.513 s |
| step time p50 / p90 / p95 | 2.502 / 2.624 / 2.792 s |
| samples/s | 12.735 |
| input WPS | 2536.1 |
| label WPS | 250.1 |
| audio-pad WPS | 2062.1 |
| loss first -> last measured | 11.4788 -> 9.7709 |
| AICORE mean / peak | 18.07% / 61.0% |
| HBM mean / peak | 60746.5 / 65512 MB |
| Power mean / peak | 157.30 / 224.2 W |

## Average Step Phase Timing

| Phase | Mean ms | Share |
|---|---:|---:|
| backward | 973.8 | 38.8% |
| clip | 40.4 | 1.6% |
| forward | 1436.8 | 57.2% |
| get_batch | 0.8 | 0.0% |
| loss_setup | 1.0 | 0.0% |
| lr_scheduler | 0.0 | 0.0% |
| move | 26.7 | 1.1% |
| optimizer | 7.0 | 0.3% |
| pregather | 0.0 | 0.0% |
| profiler | 0.0 | 0.0% |
| zero_grad | 0.3 | 0.0% |

## Run Phase Times

| Phase | Seconds |
|---|---:|
| startup_to_manual_load | 3.38 |
| qwen_safe_open_load | 32.21 |
| whisper_load | 1.37 |
| projector_and_final_barrier | 7.83 |
| post_load_to_iter1_end | 40.95 |
| step1_time | 21.97 |
| all_logged_steps | 81.67 |

## Errors
- `Traceback (most recent call last):`
