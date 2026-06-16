# ep8_mbs2_ga2_rc_off_pad256_bucket_tq1_split256 Performance Analysis

- Config: `/data/sejin/third_party/mindspeed-mm-26.0.0/examples/qwen3_5_audio/perf_tuning/ep8_mbs2_ga2_rc_off_pad256_bucket.yaml`
- Train log: `/data/sejin/baseline_26/logs/ep8_mbs2_ga2_rc_off_pad256_bucket_tq1_split256_20260615_215226.log`
- Monitor JSON: `/data/sejin/baseline_26/metrics/ep8_mbs2_ga2_rc_off_pad256_bucket_tq1_split256_20260615_215226_npu_full.json`
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
| startup_to_manual_load | 3.46 |
| qwen_safe_open_load | 35.08 |
| whisper_load | 1.59 |
| projector_and_final_barrier | 11.66 |
| post_load_to_iter1_end | 41.92 |
| step1_time | 22.62 |
| all_logged_steps | 47.53 |

## Errors
- `[rank7]: Traceback (most recent call last):`
- `[rank7]: RuntimeError: NPU out of memory. Tried to allocate 728.00 MiB (NPU 7; 60.96 GiB total capacity; 46.26 GiB already allocated; 46.26 GiB current active; 126.75 MiB free; 47.23 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.`
- `[rank2]: Traceback (most recent call last):`
- `[rank2]: RuntimeError: NPU out of memory. Tried to allocate 1.66 GiB (NPU 2; 60.96 GiB total capacity; 46.85 GiB already allocated; 46.85 GiB current active; 554.02 MiB free; 47.96 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.`
- `[rank1]: Traceback (most recent call last):`
- `[rank1]: RuntimeError: NPU out of memory. Tried to allocate 728.00 MiB (NPU 1; 60.96 GiB total capacity; 46.99 GiB already allocated; 46.99 GiB current active; 466.16 MiB free; 48.12 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.`
- `[rank6]: Traceback (most recent call last):`
- `[rank6]: RuntimeError: NPU out of memory. Tried to allocate 1.66 GiB (NPU 6; 60.96 GiB total capacity; 46.60 GiB already allocated; 46.60 GiB current active; 1.27 GiB free; 47.66 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.`
- `[rank3]: Traceback (most recent call last):`
- `[rank3]: RuntimeError: NPU out of memory. Tried to allocate 1.66 GiB (NPU 3; 60.96 GiB total capacity; 46.10 GiB already allocated; 46.10 GiB current active; 1.28 GiB free; 47.33 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.`
- `[rank5]: Traceback (most recent call last):`
- `[rank5]: RuntimeError: NPU out of memory. Tried to allocate 1.66 GiB (NPU 5; 60.96 GiB total capacity; 45.94 GiB already allocated; 45.94 GiB current active; 1.41 GiB free; 46.99 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.`
- `[rank4]: Traceback (most recent call last):`
- `[rank4]: RuntimeError: NPU out of memory. Tried to allocate 1.66 GiB (NPU 4; 60.96 GiB total capacity; 46.37 GiB already allocated; 46.37 GiB current active; 1.00 GiB free; 47.47 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.`
- `[ERROR] 2026-06-15-21:55:26 (PID:3755958, Device:7, RankID:-1) ERR99999 UNKNOWN applicaiton exception`
- `[ERROR] 2026-06-15-21:55:27 (PID:3755953, Device:2, RankID:-1) ERR99999 UNKNOWN applicaiton exception`
- `[ERROR] 2026-06-15-21:55:28 (PID:3755952, Device:1, RankID:-1) ERR99999 UNKNOWN applicaiton exception`
- `[ERROR] 2026-06-15-21:55:28 (PID:3755956, Device:5, RankID:-1) ERR99999 UNKNOWN applicaiton exception`
- `[ERROR] 2026-06-15-21:55:28 (PID:3755957, Device:6, RankID:-1) ERR99999 UNKNOWN applicaiton exception`
- `[ERROR] 2026-06-15-21:55:28 (PID:3755954, Device:3, RankID:-1) ERR99999 UNKNOWN applicaiton exception`
