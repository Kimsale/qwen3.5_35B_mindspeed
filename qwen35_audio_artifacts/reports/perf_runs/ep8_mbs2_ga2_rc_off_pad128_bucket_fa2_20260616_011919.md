# ep8_mbs2_ga2_rc_off_pad128_bucket_fa2 Performance Analysis

- Config: `/data/sejin/third_party/mindspeed-mm-26.0.0/examples/qwen3_5_audio/perf_tuning/ep8_mbs2_ga2_rc_off_pad128_bucket_fa2.yaml`
- Train log: `/data/sejin/baseline_26/logs/ep8_mbs2_ga2_rc_off_pad128_bucket_fa2_20260616_011919.log`
- Monitor JSON: `/data/sejin/baseline_26/metrics/ep8_mbs2_ga2_rc_off_pad128_bucket_fa2_20260616_011919_npu_full.json`
- Warmup skipped steps: `10`

## Config Snapshot

| Item | Value |
|---|---:|
| model_id | `qwen3_5_audio_manual_ep` |
| attn_implementation | `flash_attention_2` |
| expert_parallel_size | `8` |
| fully_shard_parallel_size | `auto` |
| tensor_parallel_size | `1` |
| ulysses_parallel_size | `1` |
| recompute | `False` |
| param_dtype | `bf16` |
| reduce_dtype | `fp32` |
| micro_batch_size | `2` |
| gradient_accumulation_steps | `2` |
| gradient_accumulation_no_sync | `False` |
| empty_cache_interval | `0` |
| train_iters | `80` |
| lr | `0.0001` |
| clip_grad | `1.0` |
| cutoff_len | `4096` |
| sampler_type | `LengthBucketBatchSampler` |
| length_bucket_size_multiplier | `64` |
| pad_to_multiple_of | `128` |
| chunk_loss_size | `1024` |
| num_workers | `8` |
| prefetch_factor | `None` |
| persistent_workers | `None` |
| dataloader_timeout | `None` |
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
| measured steps | 14 |
| step time mean | 1.938 s |
| step time p50 / p90 / p95 | 1.927 / 1.984 / 2.038 s |
| samples/s | 16.510 |
| input WPS | 3287.8 |
| label WPS | 324.3 |
| audio-pad WPS | 2673.4 |
| loss first -> last measured | 11.5081 -> 9.4059 |
| AICORE mean / peak | 14.94% / 57.0% |
| HBM mean / peak | 53584.7 / 55878 MB |
| Power mean / peak | 156.07 / 216.6 W |
| active-chip AICORE mean / p90 / peak | 14.94% / 25.0% / 57.0% |
| active-chip HBM mean / p90 / peak | 53584.7 / 55877 / 55878 MB |
| active-chip Power mean / p90 / peak | 156.07 / 173.4 / 216.6 W |
| error category | `terminated_or_hung` |

## Average Step Phase Timing

| Phase | Mean ms | Share |
|---|---:|---:|
| backward | 750.0 | 38.7% |
| clip | 30.4 | 1.6% |
| empty_cache | 0.0 | 0.0% |
| forward | 1109.1 | 57.2% |
| get_batch | 0.9 | 0.0% |
| loss_setup | 0.5 | 0.0% |
| lr_scheduler | 0.0 | 0.0% |
| move | 24.2 | 1.2% |
| optimizer | 3.9 | 0.2% |
| pregather | 0.0 | 0.0% |
| profiler | 0.0 | 0.0% |
| zero_grad | 0.3 | 0.0% |

## Run Phase Times

| Phase | Seconds |
|---|---:|
| startup_to_manual_load | 3.47 |
| qwen_safe_open_load | 35.56 |
| whisper_load | 1.30 |
| projector_and_final_barrier | 2.77 |
| post_load_to_iter1_end | 54.54 |
| step1_time | 21.85 |
| all_logged_steps | 67.10 |
| warmup_logged_steps | 39.97 |
| measured_logged_steps | 27.13 |

## Per-Chip Hardware Window

| Chip | AICORE mean / peak | HBM mean / peak MB | Power mean / peak W |
|---:|---:|---:|---:|
| 0 | 15.59 / 41.0 | 52509.9 / 52510 | 156.84 / 187.7 |
| 1 | 16.71 / 55.0 | 53460.6 / 53461 | 163.85 / 216.6 |
| 2 | 14.88 / 56.0 | 53180.9 / 53181 | 154.58 / 196.3 |
| 3 | 11.35 / 26.0 | 52701.5 / 52702 | 153.75 / 177.2 |
| 4 | 14.82 / 32.0 | 52681.5 / 52682 | 151.36 / 182.4 |
| 5 | 13.94 / 57.0 | 52441.5 / 52442 | 156.90 / 199.6 |
| 6 | 18.24 / 55.0 | 55877.8 / 55878 | 158.01 / 184.0 |
| 7 | 14.00 / 35.0 | 55823.9 / 55824 | 153.28 / 179.0 |

## Errors
- `W0616 01:23:35.651000 4086486 torch/distributed/elastic/agent/server/api.py:719] Received Signals.SIGTERM death signal, shutting down workers`
- `W0616 01:23:35.652000 4086486 torch/distributed/elastic/multiprocessing/api.py:900] Sending process 4086588 closing signal SIGTERM`
- `W0616 01:23:35.658000 4086486 torch/distributed/elastic/multiprocessing/api.py:900] Sending process 4086589 closing signal SIGTERM`
- `W0616 01:23:35.662000 4086486 torch/distributed/elastic/multiprocessing/api.py:900] Sending process 4086590 closing signal SIGTERM`
- `W0616 01:23:35.668000 4086486 torch/distributed/elastic/multiprocessing/api.py:900] Sending process 4086591 closing signal SIGTERM`
- `W0616 01:23:35.674000 4086486 torch/distributed/elastic/multiprocessing/api.py:900] Sending process 4086592 closing signal SIGTERM`
- `W0616 01:23:35.685000 4086486 torch/distributed/elastic/multiprocessing/api.py:900] Sending process 4086593 closing signal SIGTERM`
- `W0616 01:23:35.694000 4086486 torch/distributed/elastic/multiprocessing/api.py:900] Sending process 4086594 closing signal SIGTERM`
- `W0616 01:23:35.697000 4086486 torch/distributed/elastic/multiprocessing/api.py:900] Sending process 4086595 closing signal SIGTERM`
- `Traceback (most recent call last):`
- `raise SignalException(f"Process {os.getpid()} got signal: {sigval}", sigval=sigval)`
- `torch.distributed.elastic.multiprocessing.api.SignalException: Process 4086486 got signal: 15`
