# ep8_mbs2_ga2_rc_off_pad128_bucket Performance Analysis

- Config: `/data/sejin/third_party/mindspeed-mm-26.0.0/examples/qwen3_5_audio/perf_tuning/ep8_mbs2_ga2_rc_off_pad128_bucket.yaml`
- Train log: `/data/sejin/baseline_26/logs/ep8_mbs2_ga2_rc_off_pad128_bucket_20260615_215730.log`
- Monitor JSON: `/data/sejin/baseline_26/metrics/ep8_mbs2_ga2_rc_off_pad128_bucket_20260615_215730_npu_full.json`
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
| measured steps | 13 |
| step time mean | 2.132 s |
| step time p50 / p90 / p95 | 2.129 / 2.201 / 2.223 s |
| samples/s | 15.008 |
| input WPS | 2718.3 |
| label WPS | 290.8 |
| audio-pad WPS | 2163.9 |
| loss first -> last measured | 11.4686 -> 9.4605 |
| AICORE mean / peak | 23.62% / 64.0% |
| HBM mean / peak | 61235.2 / 64387 MB |
| Power mean / peak | 167.72 / 243.2 W |

## Average Step Phase Timing

| Phase | Mean ms | Share |
|---|---:|---:|
| backward | 779.3 | 36.6% |
| clip | 50.4 | 2.4% |
| forward | 1230.4 | 57.7% |
| get_batch | 0.9 | 0.0% |
| loss_setup | 0.5 | 0.0% |
| lr_scheduler | 0.0 | 0.0% |
| move | 46.6 | 2.2% |
| optimizer | 3.9 | 0.2% |
| pregather | 0.0 | 0.0% |
| profiler | 0.0 | 0.0% |
| zero_grad | 0.3 | 0.0% |

## Run Phase Times

| Phase | Seconds |
|---|---:|
| startup_to_manual_load | 3.49 |
| qwen_safe_open_load | 33.78 |
| whisper_load | 1.25 |
| projector_and_final_barrier | 7.92 |
| post_load_to_iter1_end | 56.09 |
| step1_time | 22.31 |
| all_logged_steps | 81.30 |

## Errors
- `[rank7]:[E615 22:00:29.320155898 compiler_depend.ts:444] NPU out of memory. NPUWorkspaceAllocator tried to allocate 1.89 GiB(NPU 7; 60.96 GiB total capacity; 1.82 GiB free). If you want to reduce memory usage, take a try to set the environment variable TASK_QUEUE_ENABLE=1.`
- `[rank7]: Traceback (most recent call last):`
- `[rank7]: RuntimeError: The Inner error is reported as above. The process exits for this inner error, and the current working operator name is aclnnMul.`
- `Traceback (most recent call last):`
- `raise ChildFailedError(`
- `torch.distributed.elastic.multiprocessing.errors.ChildFailedError:`
