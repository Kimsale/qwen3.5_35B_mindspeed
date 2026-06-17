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
- `reports/pack_format_validation_report.md`: WPS-first EP8 LLM pack baseline, `2111.4 WPS` for `pack rc_off, mbs=1`.
- `reports/qwen35_audio_manual_ep8_perf_tuning_20260616.md`: manual EP8 non-pack tuning summary and fallback references.
- `reports/qwen35_audio_pipeline_experts_overlap_20260617.md`: Pipeline Experts overlap implementation and 80-step performance report.
- `reports/qwen35_audio_pipeline_branch_summary_20260617.md`: branch-level summary of code changes, attempted runs, issues, and final recommendation.

## Recommended Configs

- Main WPS baseline: EP8 LLM pack `rc_off, mbs=1` from `feat/llm-pad-to-pack-recompute`, `2111.4 WPS`, about `40GB/card HBM`.
- Non-pack historical WPS reference: `examples/qwen3_5_audio/perf_tuning/ep8_mbs1_ga4_rc_off_pad1024_pregather_nosync.yaml`, `1414.6 WPS`.
- Non-pack current-code WPS reference: `examples/qwen3_5_audio/perf_tuning/ep8_mbs1_ga4_rc_off_pad1280.yaml`, `1295.8 WPS`.
- Strict HBM reference only: `examples/qwen3_5_audio/perf_tuning/ep8_mbs1_ga4_rc_off_pad1536_nosync.yaml`, `1132.5 WPS`, about `56.4GB/card HBM`.
- Pipeline Experts experiment: `examples/qwen3_5_audio/perf_tuning/mbs1_pipeline_pad1536_nosync_80.yaml`, stable but regressed to `645.5 WPS`.
