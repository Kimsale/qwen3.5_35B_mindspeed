# Qwen3.5 Audio Manual EP8 MoE Blog Tuning Report

Generated: 2026-06-16 10:20:25

## Scope

- Model architecture, expert count, MoE routing, Whisper encoder, and manual EP8 expert slicing are unchanged.
- Tuning surface: micro batch, gradient accumulation sync, attention backend, chunk loss, length bucketing, recompute, gradient clip, empty cache cadence, watchdog diagnostics.
- Metrics are post-warmup only: skip first 10 steps; init, safe_open load, dataset build, and first compile are excluded.
- The provided Zhihu URL was not directly readable in this environment, so the adopted items are standard MoE tuning practices that match the local code path: larger micro batch, grouped expert matmul, dispatch stability, reduced sync bubbles, and padding/bucket control.

## Current Implementation

- Added optional micro-step phase markers and phase-specific NPU synchronize diagnostics in `TrainEngine`.
- Added `SIGUSR1` Python stack dump registration in the trainer process.
- Added watchdog dump/terminate logic to `run_audio_perf_experiment.sh` for mbs2 hangs.
- Added mbs2 FA2/no_sync candidate configs and an automated tuning suite script.

## New Candidate Results

| Config | Status | Step mean | Input WPS | AICORE mean / peak | HBM mean / peak | Power mean / peak | Analysis |
|---|---:|---:|---:|---:|---:|---:|---|
| `ep8_mbs2_ga2_rc_off_pad128_bucket64_fa2_nosync_chunk512_diag` | not_run | N/A | N/A | N/A | N/A | N/A | N/A |
| `ep8_mbs2_ga2_rc_off_pad128_bucket64_fa2_nosync_chunk512` | not_run | N/A | N/A | N/A | N/A | N/A | N/A |
| `ep8_mbs2_ga2_rc_off_pad128_bucket64_fa2_nosync_chunk512_empty4` | not_run | N/A | N/A | N/A | N/A | N/A | N/A |
| `ep8_mbs2_ga2_rc_off_pad128_bucket32_fa2_nosync_chunk512` | not_run | N/A | N/A | N/A | N/A | N/A | N/A |
| `ep8_mbs2_ga2_rc_off_pad128_bucket64_fa2_nosync_chunk256` | not_run | N/A | N/A | N/A | N/A | N/A | N/A |
| `ep8_mbs2_ga2_rc_off_pad128_bucket64_fa2_nosync_chunk512_clip05` | not_run | N/A | N/A | N/A | N/A | N/A | N/A |
| `ep8_mbs2_ga2_rc_off_pad128_bucket32_fa2_nosync_chunk512_rc_on` | not_run | N/A | N/A | N/A | N/A | N/A | N/A |

## Reference Results

| Config | Status | Step mean | Input WPS | AICORE mean / peak | HBM mean / peak | Power mean / peak | Analysis |
|---|---:|---:|---:|---:|---:|---:|---|
| `ep8_mbs1_ga4_rc_off_pad1280_current` | success | 4.279s | 1295.8 | 23.58 / 38.0 | 51925.4 / 53244 MB | 166.53 / 195.9 W | `/data/sejin/baseline_26/metrics/ep8_mbs1_ga4_rc_off_pad1280_current_20260616_021501_analysis.json` |
| `ep8_mbs1_ga4_rc_off_pad1408_nosync` | success | 4.786s | 1158.3 | 22.48 / 40.0 | 54638.0 / 56111 MB | 162.57 / 203.7 W | `/data/sejin/baseline_26/metrics/ep8_mbs1_ga4_rc_off_pad1408_nosync_20260615_234845_analysis.json` |
| `ep8_mbs1_ga4_rc_off_pad1536_nosync_rerun05` | success | 4.895s | 1132.5 | 23.43 / 43.0 | 56400.7 / 58059 MB | 164.54 / 205.6 W | `/data/sejin/baseline_26/metrics/ep8_mbs1_ga4_rc_off_pad1536_nosync_rerun05_20260616_010537_analysis.json` |
| `ep8_mbs2_ga2_rc_off_pad128_bucket_fa2` | terminated_or_hung | 1.938s | 3287.8 | 14.94 / 57.0 | 53584.7 / 55878 MB | 156.07 / 216.6 W | `/data/sejin/baseline_26/metrics/ep8_mbs2_ga2_rc_off_pad128_bucket_fa2_20260616_011919_analysis.json` |
| `ep8_mbs2_ga2_rc_off_pad128_bucket` | partial_13_steps | 2.132s | 2718.3 | 23.62 / 64.0 | 61235.2 / 64387 MB | 167.72 / 243.2 W | `/data/sejin/baseline_26/metrics/ep8_mbs2_ga2_rc_off_pad128_bucket_20260615_215730_analysis.json` |

## Best Available Recommendation

No new complete mbs2 run is available yet. Best complete reference run remains `ep8_mbs1_ga4_rc_off_pad1536_nosync_rerun05`.

## Phase Timing For Best Available

Source: `ep8_mbs1_ga4_rc_off_pad1536_nosync_rerun05`

| Phase | Mean |
|---|---:|
| backward | 1616.2 ms |
| clip | 139.2 ms |
| empty_cache | 0.0 ms |
| forward | 2662.7 ms |
| get_batch | 1.8 ms |
| loss_setup | 0.9 ms |
| lr_scheduler | 0.0 ms |
| move | 441.9 ms |
| optimizer | 3.9 ms |
| pregather | 0.0 ms |
| profiler | 0.0 ms |
| zero_grad | 0.3 ms |

## Resource Snapshot At Report Generation

```text
| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |
+===========================+===============+====================================================+
| 0       0                 | 2857129       | VLLMWorker               | 111                     |
+===========================+===============+====================================================+
| 1       0                 | 269293        | python3                  | 109                     |
+===========================+===============+====================================================+
| 2       0                 | 253025        | python3                  | 109                     |
| 2       0                 | 234470        | python3                  | 61629                   |
+===========================+===============+====================================================+
| 3       0                 | 253021        | python3                  | 109                     |
+===========================+===============+====================================================+
| 4       0                 | 253023        | python3                  | 61521                   |
+===========================+===============+====================================================+
| 5       0                 | 253022        | python3                  | 109                     |
+===========================+===============+====================================================+
| 6       0                 | 253020        | python3                  | 61535                   |
+===========================+===============+====================================================+
| No running processes found in NPU 7                                                            |
+===========================+===============+====================================================+
```
