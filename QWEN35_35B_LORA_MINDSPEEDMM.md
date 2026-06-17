# Qwen3.5 35B LoRA MindSpeed-MM Notes

This repository is based on Ascend MindSpeed-MM `26.0.0`.

## Git Layout

- `baseline-mindspeed-mm-26.0.0`: clean upstream baseline tag before local Qwen3.5 audio/LoRA changes.
- `main`: current optimized Qwen3.5 35B LoRA/audio training code.
- `origin`: original MindSpeed-MM upstream from GitCode.
- `github`: personal GitHub remote for this organized working copy.

## Where Things Are

- Core MindSpeed-MM code changes:
  - `mindspeed_mm/fsdp/models/qwen3_5_audio/`
  - `mindspeed_mm/fsdp/train/`
  - `mindspeed_mm/fsdp/params/training_args.py`
  - `mindspeed_mm/fsdp/data/`
  - `mindspeed_mm/fsdp/distributed/expert_parallel/`
  - `mindspeed_mm/fsdp/optimizer/clip_grad_norm.py`
- Training configs:
  - `examples/qwen3_5_audio/`
  - `examples/qwen3_5_audio/perf_tuning/`
- Experiment scripts, reports, and compact metrics:
  - `qwen35_audio_artifacts/scripts/`
  - `qwen35_audio_artifacts/reports/`
  - `qwen35_audio_artifacts/metrics_analysis/`
  - `qwen35_audio_artifacts/docs/`

## Current Recommendation

Use WPS as the primary comparison target. The best current baseline is the EP8 LLM pack run from `feat/llm-pad-to-pack-recompute`, not the strict-HBM padded run:

- `pack rc_off, mbs=1`: stable 80 steps, last-40-step average `2111.4 WPS`, about `3.6s/iter`, about `40GB/card HBM`.
- `pack rc_on, mbs=1`: stable 80 steps, `1475.3 WPS`, about `33GB/card HBM`.
- `pack mbs=2`: blocked by FSDP2 lazy initialization hang because packed variable sequence lengths are not rank-aligned.

The local copy of that baseline report is `qwen35_audio_artifacts/reports/pack_format_validation_report.md`.

## This Branch Result

This branch implements Pipeline Experts overlap for the Qwen3.5 MoE EP path and validates it with:

```bash
MASTER_PORT=6052 bash /data/sejin/baseline_26/scripts/run_audio_perf_experiment.sh \
  qwen35_audio_pipeline_mbs1_pad1536_ep8_8ada09f36_20260617_110652 \
  /data/sejin/third_party/mindspeed-mm-pipeline-experts-overlap/examples/qwen3_5_audio/perf_tuning/mbs1_pipeline_pad1536_nosync_80.yaml \
  1500 10 1.0
```

Post-warmup result, skipping the first 10 steps:

- Status: completed `80/80`, no OOM or runtime error.
- Step time: `8.589s`.
- Input WPS: `645.5`.
- AICORE mean / peak: `12.04% / 42.0%`.
- HBM mean / peak: `56,581.7 / 58,286 MB`.
- Power mean / peak: `135.28 / 182.9 W`.

Conclusion: Pipeline Experts overlap is stable under `mbs1/pad1536`, but it is a WPS regression. It should remain an experimental branch, while WPS optimization should continue from the pack baseline.

## Historical Non-Pack References

Manual EP8 non-pack still matters as a fallback/reference path:

- `pad1024_pregather_nosync`: historical best completed non-pack run, `1414.6 WPS`, `48.75GB/card HBM`.
- `pad1280_current`: current-code best non-pack rerun, `1295.8 WPS`, `51.93GB/card HBM`.
- `pad1536_nosync_rerun05`: strict-HBM reference only, `1132.5 WPS`, `56.40GB/card HBM`.

## What Is Intentionally Not Committed

Large or machine-local artifacts are ignored:

- raw datasets and generated cache files
- logs and PID files
- checkpoints and LoRA `.safetensors`
- local nested dependency clones
- model weights

Keep those under local paths such as `/data/sejin/baseline_26`, `/mnt/shared_data_196/sejin/models`, or another storage location.
