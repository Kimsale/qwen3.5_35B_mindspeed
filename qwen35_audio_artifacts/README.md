# Qwen3.5 Audio LoRA Experiment Artifacts

This directory collects the small, reproducible assets from `/data/sejin/baseline_26`.

## Contents

- `scripts/`: launch scripts, data generation helpers, monitor scripts, and report analysis tools.
- `reports/`: baseline, EP sweep, manual EP8 tuning, and perf-run markdown reports.
- `metrics_analysis/`: compact `*_analysis.json` files. Full `npu_full.json`, logs, caches, and checkpoints are not committed.
- `docs/`: local training status and run guides.

## Key Reports

- `reports/qwen35_audio_distfix60_performance_report_20260615.md`: original EP=1/FSDP2 baseline.
- `reports/qwen35_audio_ep_sweep_20260615.md`: early EP sweep showing EP4/EP8 optimizer-step OOM.
- `reports/qwen35_audio_manual_ep8_perf_tuning_20260616.md`: latest manual EP8 tuning summary and recommendation.

## Recommended Configs

- Strict HBM 55-60GB: `examples/qwen3_5_audio/perf_tuning/ep8_mbs1_ga4_rc_off_pad1536_nosync.yaml`
- Highest current-code WPS: `examples/qwen3_5_audio/perf_tuning/ep8_mbs1_ga4_rc_off_pad1280.yaml`
- Near-HBM target with better WPS than pad1536: `examples/qwen3_5_audio/perf_tuning/ep8_mbs1_ga4_rc_off_pad1408_nosync.yaml`
