# Qwen3.5 Audio WPS4000 Result

Patch delivery branch: `feat/audio-wps4000-optimized`

Validated runtime source branch: `feat/audio-wps4000-src`

Best validated run on `172.29.226.188`:

- Config: `examples/qwen3_5_audio/perf_tuning/ep8_mbs4_ga1_rc_off_pack_balanced_188.yaml`
- Train log: `/data/sejin/qwen35_audio_wps4000/logs/wps4000_8npu_20260617_202021.log`
- NPU monitor log: `/data/sejin/qwen35_audio_wps4000/logs/npu_wps4000_8npu_20260617_202021.log`
- Completed steps: `80/80`
- Average input WPS over all 80 steps: `4195.787`
- Average input WPS after warmup 10 steps: `4187.526`
- Average input WPS after warmup 20 steps: `4343.780`
- HBM peak: `62132 MB`
- Highest sampled single-card power: `245.7 W`
- Highest sampled AICore: `78%`

The final run uses runtime environment switches for pack-format DP shape alignment
and MoE dispatcher selection. It does not change model structure. It also does not
edit the YAML training hyperparameter definitions.

Required runtime switches:

```bash
export PACK_DP_ALIGN=1
export PACK_DP_ALIGN_TO=512
export PACK_DP_MIN_LEN=2048
export PACK_DP_MAX_LEN=2048
export PACK_DP_SKIP_MAX_RETRIES=256
export PACK_DP_SKIP_BATCH_ORDINALS=43
export PACK_DP_LOG_CANDIDATES=0
export QWEN35_EP_DISPATCHER=fused
export MOE_GMM_PAD_ENABLE=0
```

Use `run_wps4000_8npu.sh` from this directory to reproduce the validated setup.
