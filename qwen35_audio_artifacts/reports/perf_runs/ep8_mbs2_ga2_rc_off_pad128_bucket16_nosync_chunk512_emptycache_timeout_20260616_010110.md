# ep8_mbs2_ga2_rc_off_pad128_bucket16_nosync_chunk512_emptycache_timeout Performance Analysis

- Config: `/data/sejin/third_party/mindspeed-mm-26.0.0/examples/qwen3_5_audio/perf_tuning/ep8_mbs2_ga2_rc_off_pad128_bucket16_nosync_chunk512_emptycache_timeout.yaml`
- Train log: `/data/sejin/baseline_26/logs/ep8_mbs2_ga2_rc_off_pad128_bucket16_nosync_chunk512_emptycache_timeout_20260616_010110.log`
- Monitor JSON: `/data/sejin/baseline_26/metrics/ep8_mbs2_ga2_rc_off_pad128_bucket16_nosync_chunk512_emptycache_timeout_20260616_010110_npu_full.json`
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
| length_bucket_size_multiplier | `16` |
| pad_to_multiple_of | `128` |
| chunk_loss_size | `512` |
| num_workers | `8` |
| prefetch_factor | `None` |
| persistent_workers | `None` |
| dataloader_timeout | `120` |
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
| active-chip AICORE mean / p90 / peak | N/A% / N/A% / N/A% |
| active-chip HBM mean / p90 / peak | N/A / N/A / N/A MB |
| active-chip Power mean / p90 / peak | N/A / N/A / N/A W |
| error category | `terminated_or_hung` |

## Average Step Phase Timing

| Phase | Mean ms | Share |
|---|---:|---:|

## Run Phase Times

| Phase | Seconds |
|---|---:|
| startup_to_manual_load | 3.48 |
| qwen_safe_open_load | 30.91 |
| whisper_load | 1.33 |
| projector_and_final_barrier | 8.66 |

## Errors
- `W0616 01:04:06.263000 4059301 torch/distributed/elastic/agent/server/api.py:719] Received Signals.SIGTERM death signal, shutting down workers`
- `W0616 01:04:06.264000 4059301 torch/distributed/elastic/multiprocessing/api.py:900] Sending process 4059400 closing signal SIGTERM`
- `W0616 01:04:06.266000 4059301 torch/distributed/elastic/multiprocessing/api.py:900] Sending process 4059401 closing signal SIGTERM`
- `W0616 01:04:06.276000 4059301 torch/distributed/elastic/multiprocessing/api.py:900] Sending process 4059402 closing signal SIGTERM`
- `W0616 01:04:06.281000 4059301 torch/distributed/elastic/multiprocessing/api.py:900] Sending process 4059403 closing signal SIGTERM`
- `W0616 01:04:06.283000 4059301 torch/distributed/elastic/multiprocessing/api.py:900] Sending process 4059404 closing signal SIGTERM`
- `W0616 01:04:06.292000 4059301 torch/distributed/elastic/multiprocessing/api.py:900] Sending process 4059405 closing signal SIGTERM`
- `W0616 01:04:06.298000 4059301 torch/distributed/elastic/multiprocessing/api.py:900] Sending process 4059406 closing signal SIGTERM`
- `W0616 01:04:06.306000 4059301 torch/distributed/elastic/multiprocessing/api.py:900] Sending process 4059407 closing signal SIGTERM`
- `Traceback (most recent call last):`
- `raise SignalException(f"Process {os.getpid()} got signal: {sigval}", sigval=sigval)`
- `torch.distributed.elastic.multiprocessing.api.SignalException: Process 4059301 got signal: 15`
