# ep8_mbs2_ga2_rc_off_pad256_bucket Performance Analysis

- Config: `/data/sejin/third_party/mindspeed-mm-26.0.0/examples/qwen3_5_audio/perf_tuning/ep8_mbs2_ga2_rc_off_pad256_bucket.yaml`
- Train log: `/data/sejin/baseline_26/logs/ep8_mbs2_ga2_rc_off_pad256_bucket_20260615_214657.log`
- Monitor JSON: `/data/sejin/baseline_26/metrics/ep8_mbs2_ga2_rc_off_pad256_bucket_20260615_214657_npu_full.json`
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
| pad_to_multiple_of | `256` |
| num_workers | `8` |
| lora_rank | `16` |
| lora_alpha | `32` |
| lora_dropout | `0.05` |

## Post-Warmup Metrics

| Metric | Value |
|---|---:|
| measured steps | 0 |
| step time mean | N/A s |
| step time p50 / p90 / p95 | N/A / N/A / N/A s |
| samples/s | N/A |
| input WPS | N/A |
| label WPS | N/A |
| audio-pad WPS | N/A |
| loss first -> last measured | N/A -> N/A |
| AICORE mean / peak | N/A% / N/A% |
| HBM mean / peak | N/A / N/A MB |
| Power mean / peak | N/A / N/A W |

## Average Step Phase Timing

| Phase | Mean ms | Share |
|---|---:|---:|

## Run Phase Times

| Phase | Seconds |
|---|---:|
| startup_to_manual_load | 3.57 |
| qwen_safe_open_load | 31.35 |
| whisper_load | 1.29 |
| projector_and_final_barrier | 12.14 |
| post_load_to_iter1_end | 61.15 |
| step1_time | 23.10 |
| all_logged_steps | 41.76 |

## Errors
- `[rank1]:[E615 21:49:22.056172510 compiler_depend.ts:444] NPU out of memory. NPUWorkspaceAllocator tried to allocate 1.66 GiB(NPU 1; 60.96 GiB total capacity; 1006.32 MiB free). If you want to reduce memory usage, take a try to set the environment variable TASK_QUEUE_ENABLE=1.`
- `[rank1]: Traceback (most recent call last):`
- `[rank1]: RuntimeError: The Inner error is reported as above. The process exits for this inner error, and the current working operator name is aclnnMm.`
- `[rank2]:[E615 21:49:22.125325668 compiler_depend.ts:444] NPU out of memory. NPUWorkspaceAllocator tried to allocate 1.66 GiB(NPU 2; 60.96 GiB total capacity; 1.35 GiB free). If you want to reduce memory usage, take a try to set the environment variable TASK_QUEUE_ENABLE=1.`
- `[rank2]: Traceback (most recent call last):`
- `[rank2]: RuntimeError: The Inner error is reported as above. The process exits for this inner error, and the current working operator name is aclnnMm.`
- `Traceback (most recent call last):`
- `raise ChildFailedError(`
- `torch.distributed.elastic.multiprocessing.errors.ChildFailedError:`
