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

## Current Production Recommendation

Use the strict-HBM stable EP8 config:

```bash
MASTER_PORT=6052 bash /data/sejin/baseline_26/scripts/run_audio_perf_experiment.sh \
  ep8_mbs1_ga4_rc_off_pad1536_nosync \
  /data/sejin/third_party/mindspeed-mm-26.0.0/examples/qwen3_5_audio/perf_tuning/ep8_mbs1_ga4_rc_off_pad1536_nosync.yaml \
  1500 10 1.0
```

Main result from `qwen35_audio_artifacts/reports/qwen35_audio_manual_ep8_perf_tuning_20260616.md`:

- Step time: 4.895s
- Input WPS: 1132.5
- AICORE mean: 23.43%
- HBM mean: 56.40GB
- Power mean: 164.54W

If WPS is more important than HBM 55-60GB, use `ep8_mbs1_ga4_rc_off_pad1280_current`:

- Step time: 4.279s
- Input WPS: 1295.8
- HBM mean: 51.93GB

## What Is Intentionally Not Committed

Large or machine-local artifacts are ignored:

- raw datasets and generated cache files
- logs and PID files
- checkpoints and LoRA `.safetensors`
- local nested dependency clones
- model weights

Keep those under local paths such as `/data/sejin/baseline_26`, `/mnt/shared_data_196/sejin/models`, or another storage location.
