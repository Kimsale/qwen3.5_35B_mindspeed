# ep8_mbs2_ga2_rc_off_pad128_bucket_nosync_chunk512_emptycache Performance Analysis

- Config: `/data/sejin/third_party/mindspeed-mm-26.0.0/examples/qwen3_5_audio/perf_tuning/ep8_mbs2_ga2_rc_off_pad128_bucket_nosync_chunk512_emptycache.yaml`
- Train log: `/data/sejin/baseline_26/logs/ep8_mbs2_ga2_rc_off_pad128_bucket_nosync_chunk512_emptycache_20260616_003945.log`
- Monitor JSON: `/data/sejin/baseline_26/metrics/ep8_mbs2_ga2_rc_off_pad128_bucket_nosync_chunk512_emptycache_20260616_003945_npu_full.json`
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
| gradient_accumulation_no_sync | `True` |
| empty_cache_interval | `1` |
| train_iters | `80` |
| lr | `0.0001` |
| clip_grad | `1.0` |
| cutoff_len | `4096` |
| sampler_type | `LengthBucketBatchSampler` |
| length_bucket_size_multiplier | `64` |
| pad_to_multiple_of | `128` |
| chunk_loss_size | `512` |
| num_workers | `8` |
| lora_rank | `16` |
| lora_alpha | `32` |
| lora_dropout | `0.05` |

## Runtime Env

| Item | Value |
|---|---:|
| MULTI_STREAM_MEMORY_REUSE | `2` |
| TASK_QUEUE_ENABLE | `1` |
| PYTORCH_NPU_ALLOC_CONF | `max_split_size_mb:512` |
| ACLNN_CACHE_LIMIT | `100000` |
| CPU_AFFINITY_CONF | `1` |
| HCCL_CONNECT_TIMEOUT | `1800` |

## Post-Warmup Metrics

| Metric | Value |
|---|---:|
| measured steps | 14 |
| step time mean | 2.458 s |
| step time p50 / p90 / p95 | 2.377 / 2.439 / 2.867 s |
| samples/s | 13.019 |
| input WPS | 2592.5 |
| label WPS | 255.7 |
| audio-pad WPS | 2108.0 |
| loss first -> last measured | 11.4660 -> 9.3553 |
| AICORE mean / peak | 12.57% / 45.0% |
| HBM mean / peak | 43658.0 / 65190 MB |
| Power mean / peak | 148.14 / 189.2 W |
| active-chip AICORE mean / p90 / peak | 12.57% / 24.0% / 45.0% |
| active-chip HBM mean / p90 / peak | 43658.0 / 52785 / 65190 MB |
| active-chip Power mean / p90 / peak | 148.14 / 167.3 / 189.2 W |
| error category | `terminated_or_hung` |

## Average Step Phase Timing

| Phase | Mean ms | Share |
|---|---:|---:|
| backward | 958.0 | 39.0% |
| clip | 24.2 | 1.0% |
| empty_cache | 124.7 | 5.1% |
| forward | 1293.7 | 52.6% |
| get_batch | 0.9 | 0.0% |
| loss_setup | 0.9 | 0.0% |
| lr_scheduler | 0.0 | 0.0% |
| move | 18.8 | 0.8% |
| optimizer | 7.1 | 0.3% |
| pregather | 0.0 | 0.0% |
| profiler | 0.0 | 0.0% |
| zero_grad | 0.3 | 0.0% |

## Run Phase Times

| Phase | Seconds |
|---|---:|
| startup_to_manual_load | 3.54 |
| qwen_safe_open_load | 32.45 |
| whisper_load | 1.30 |
| projector_and_final_barrier | 6.74 |
| post_load_to_iter1_end | 53.34 |
| step1_time | 21.38 |
| all_logged_steps | 77.90 |
| warmup_logged_steps | 43.48 |
| measured_logged_steps | 34.41 |

## Per-Chip Hardware Window

| Chip | AICORE mean / peak | HBM mean / peak MB | Power mean / peak W |
|---:|---:|---:|---:|
| 0 | 12.76 / 45.0 | 44149.2 / 61519 | 146.28 / 189.2 |
| 1 | 12.47 / 35.0 | 44480.4 / 63108 | 151.33 / 183.8 |
| 2 | 11.06 / 20.0 | 43430.3 / 62735 | 143.52 / 167.6 |
| 3 | 13.35 / 32.0 | 42801.0 / 64962 | 145.95 / 170.9 |
| 4 | 13.82 / 31.0 | 43524.1 / 65190 | 146.49 / 168.6 |
| 5 | 15.00 / 43.0 | 44240.0 / 64887 | 155.96 / 178.2 |
| 6 | 11.94 / 31.0 | 43940.2 / 65125 | 150.45 / 175.2 |
| 7 | 10.18 / 28.0 | 42698.8 / 64264 | 145.19 / 173.2 |

## Errors
- `W0616 00:44:57.731000 4026234 torch/distributed/elastic/agent/server/api.py:719] Received Signals.SIGTERM death signal, shutting down workers`
- `W0616 00:44:57.732000 4026234 torch/distributed/elastic/multiprocessing/api.py:900] Sending process 4026322 closing signal SIGTERM`
- `W0616 00:44:57.733000 4026234 torch/distributed/elastic/multiprocessing/api.py:900] Sending process 4026323 closing signal SIGTERM`
- `W0616 00:44:57.742000 4026234 torch/distributed/elastic/multiprocessing/api.py:900] Sending process 4026324 closing signal SIGTERM`
- `W0616 00:44:57.749000 4026234 torch/distributed/elastic/multiprocessing/api.py:900] Sending process 4026325 closing signal SIGTERM`
- `W0616 00:44:57.758000 4026234 torch/distributed/elastic/multiprocessing/api.py:900] Sending process 4026326 closing signal SIGTERM`
- `W0616 00:44:57.764000 4026234 torch/distributed/elastic/multiprocessing/api.py:900] Sending process 4026327 closing signal SIGTERM`
- `W0616 00:44:57.773000 4026234 torch/distributed/elastic/multiprocessing/api.py:900] Sending process 4026328 closing signal SIGTERM`
- `W0616 00:44:57.785000 4026234 torch/distributed/elastic/multiprocessing/api.py:900] Sending process 4026329 closing signal SIGTERM`
- `Traceback (most recent call last):`
- `raise SignalException(f"Process {os.getpid()} got signal: {sigval}", sigval=sigval)`
- `torch.distributed.elastic.multiprocessing.api.SignalException: Process 4026234 got signal: 15`
