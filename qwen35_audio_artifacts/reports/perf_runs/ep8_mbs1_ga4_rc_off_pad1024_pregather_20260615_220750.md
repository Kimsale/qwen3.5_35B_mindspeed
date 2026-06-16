# ep8_mbs1_ga4_rc_off_pad1024_pregather Performance Analysis

- Config: `/data/sejin/third_party/mindspeed-mm-26.0.0/examples/qwen3_5_audio/perf_tuning/ep8_mbs1_ga4_rc_off_pad1024_pregather.yaml`
- Train log: `/data/sejin/baseline_26/logs/ep8_mbs1_ga4_rc_off_pad1024_pregather_20260615_220750.log`
- Monitor JSON: `/data/sejin/baseline_26/metrics/ep8_mbs1_ga4_rc_off_pad1024_pregather_20260615_220750_npu_full.json`
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

## Post-Warmup Metrics

| Metric | Value |
|---|---:|
| measured steps | 70 |
| step time mean | 4.457 s |
| step time p50 / p90 / p95 | 4.402 / 4.623 / 4.957 s |
| samples/s | 7.179 |
| input WPS | 1243.8 |
| label WPS | 139.8 |
| audio-pad WPS | 978.8 |
| loss first -> last measured | 11.3172 -> 4.9067 |
| AICORE mean / peak | 20.90% / 50.0% |
| HBM mean / peak | 57029.4 / 60786 MB |
| Power mean / peak | 164.90 / 233.2 W |

## Average Step Phase Timing

| Phase | Mean ms | Share |
|---|---:|---:|
| backward | 1599.8 | 35.9% |
| clip | 63.2 | 1.4% |
| forward | 2327.3 | 52.2% |
| get_batch | 2.0 | 0.0% |
| loss_setup | 1.0 | 0.0% |
| lr_scheduler | 0.0 | 0.0% |
| move | 214.9 | 4.8% |
| optimizer | 3.9 | 0.1% |
| pregather | 215.0 | 4.8% |
| profiler | 0.0 | 0.0% |
| zero_grad | 0.3 | 0.0% |

## Run Phase Times

| Phase | Seconds |
|---|---:|
| startup_to_manual_load | 3.51 |
| qwen_safe_open_load | 31.48 |
| whisper_load | 1.33 |
| projector_and_final_barrier | 9.86 |
| post_load_to_iter1_end | 57.62 |
| step1_time | 24.16 |
| all_logged_steps | 376.41 |
| last_step_to_lora_save | 0.11 |
