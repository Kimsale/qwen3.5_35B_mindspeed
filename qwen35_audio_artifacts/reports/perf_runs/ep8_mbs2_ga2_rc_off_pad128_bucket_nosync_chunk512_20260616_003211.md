# ep8_mbs2_ga2_rc_off_pad128_bucket_nosync_chunk512 Performance Analysis

- Config: `/data/sejin/third_party/mindspeed-mm-26.0.0/examples/qwen3_5_audio/perf_tuning/ep8_mbs2_ga2_rc_off_pad128_bucket_nosync_chunk512.yaml`
- Train log: `/data/sejin/baseline_26/logs/ep8_mbs2_ga2_rc_off_pad128_bucket_nosync_chunk512_20260616_003211.log`
- Monitor JSON: `/data/sejin/baseline_26/metrics/ep8_mbs2_ga2_rc_off_pad128_bucket_nosync_chunk512_20260616_003211_npu_full.json`
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
| step time mean | 2.544 s |
| step time p50 / p90 / p95 | 2.236 / 3.331 / 3.488 s |
| samples/s | 12.576 |
| input WPS | 2504.4 |
| label WPS | 247.0 |
| audio-pad WPS | 2036.4 |
| loss first -> last measured | 11.5229 -> 9.5493 |
| AICORE mean / peak | 11.29% / 54.0% |
| HBM mean / peak | 63452.1 / 65528 MB |
| Power mean / peak | 146.72 / 190.4 W |
| active-chip AICORE mean / p90 / peak | 11.29% / 20.0% / 54.0% |
| active-chip HBM mean / p90 / peak | 63452.1 / 65166 / 65528 MB |
| active-chip Power mean / p90 / peak | 146.72 / 167.5 / 190.4 W |
| error category | `terminated_or_hung` |

## Average Step Phase Timing

| Phase | Mean ms | Share |
|---|---:|---:|
| backward | 1195.0 | 47.0% |
| clip | 43.2 | 1.7% |
| forward | 1252.8 | 49.2% |
| get_batch | 0.9 | 0.0% |
| loss_setup | 0.8 | 0.0% |
| lr_scheduler | 0.0 | 0.0% |
| move | 26.1 | 1.0% |
| optimizer | 7.0 | 0.3% |
| pregather | 0.0 | 0.0% |
| profiler | 0.0 | 0.0% |
| zero_grad | 0.3 | 0.0% |

## Run Phase Times

| Phase | Seconds |
|---|---:|
| startup_to_manual_load | 3.54 |
| qwen_safe_open_load | 32.98 |
| whisper_load | 1.39 |
| projector_and_final_barrier | 3.40 |
| post_load_to_iter1_end | 53.83 |
| step1_time | 22.12 |
| all_logged_steps | 79.17 |
| warmup_logged_steps | 43.55 |
| measured_logged_steps | 35.62 |

## Per-Chip Hardware Window

| Chip | AICORE mean / peak | HBM mean / peak MB | Power mean / peak W |
|---:|---:|---:|---:|
| 0 | 11.88 / 20.0 | 63064.8 / 63066 | 150.62 / 170.9 |
| 1 | 13.12 / 25.0 | 63646.3 / 64581 | 156.32 / 183.0 |
| 2 | 11.71 / 24.0 | 63255.1 / 63962 | 143.35 / 160.5 |
| 3 | 9.06 / 23.0 | 62557.4 / 65218 | 142.97 / 176.6 |
| 4 | 13.06 / 44.0 | 62735.1 / 65465 | 142.38 / 180.2 |
| 5 | 11.94 / 54.0 | 62394.4 / 65528 | 149.76 / 190.4 |
| 6 | 9.29 / 44.0 | 64910.8 / 65169 | 146.32 / 183.8 |
| 7 | 10.24 / 46.0 | 65053.0 / 65304 | 142.05 / 182.8 |

## Errors
- `W0616 00:37:22.830000 4015698 torch/distributed/elastic/agent/server/api.py:719] Received Signals.SIGTERM death signal, shutting down workers`
- `W0616 00:37:22.831000 4015698 torch/distributed/elastic/multiprocessing/api.py:900] Sending process 4015788 closing signal SIGTERM`
- `W0616 00:37:22.844000 4015698 torch/distributed/elastic/multiprocessing/api.py:900] Sending process 4015789 closing signal SIGTERM`
- `W0616 00:37:22.865000 4015698 torch/distributed/elastic/multiprocessing/api.py:900] Sending process 4015790 closing signal SIGTERM`
- `W0616 00:37:22.874000 4015698 torch/distributed/elastic/multiprocessing/api.py:900] Sending process 4015791 closing signal SIGTERM`
- `W0616 00:37:22.876000 4015698 torch/distributed/elastic/multiprocessing/api.py:900] Sending process 4015792 closing signal SIGTERM`
- `W0616 00:37:22.884000 4015698 torch/distributed/elastic/multiprocessing/api.py:900] Sending process 4015793 closing signal SIGTERM`
- `W0616 00:37:22.885000 4015698 torch/distributed/elastic/multiprocessing/api.py:900] Sending process 4015794 closing signal SIGTERM`
- `W0616 00:37:22.889000 4015698 torch/distributed/elastic/multiprocessing/api.py:900] Sending process 4015795 closing signal SIGTERM`
- `Traceback (most recent call last):`
- `raise SignalException(f"Process {os.getpid()} got signal: {sigval}", sigval=sigval)`
- `torch.distributed.elastic.multiprocessing.api.SignalException: Process 4015698 got signal: 15`
