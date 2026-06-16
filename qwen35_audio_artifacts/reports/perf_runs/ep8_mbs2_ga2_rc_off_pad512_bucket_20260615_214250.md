# ep8_mbs2_ga2_rc_off_pad512_bucket Performance Analysis

- Config: `/data/sejin/third_party/mindspeed-mm-26.0.0/examples/qwen3_5_audio/perf_tuning/ep8_mbs2_ga2_rc_off_pad512_bucket.yaml`
- Train log: `/data/sejin/baseline_26/logs/ep8_mbs2_ga2_rc_off_pad512_bucket_20260615_214250.log`
- Monitor JSON: `/data/sejin/baseline_26/metrics/ep8_mbs2_ga2_rc_off_pad512_bucket_20260615_214250_npu_full.json`
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
| pad_to_multiple_of | `512` |
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
| startup_to_manual_load | 3.48 |
| qwen_safe_open_load | 33.51 |
| whisper_load | 1.32 |
| projector_and_final_barrier | 7.06 |
| post_load_to_iter1_end | 69.82 |
| step1_time | 32.93 |
| all_logged_steps | 35.26 |

## Errors
- `[rank7]: Traceback (most recent call last):`
- `[rank7]: RuntimeError: NPU out of memory. Tried to allocate 1.90 GiB (NPU 7; 60.96 GiB total capacity; 46.97 GiB already allocated; 46.97 GiB current active; 1.65 GiB free; 47.25 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.`
- `[ERROR] 2026-06-15-21:45:45 (PID:3734596, Device:7, RankID:-1) ERR99999 UNKNOWN applicaiton exception`
- `Traceback (most recent call last):`
- `raise ChildFailedError(`
- `torch.distributed.elastic.multiprocessing.errors.ChildFailedError:`
