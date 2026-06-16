#!/bin/bash
# Run the Qwen3.5 audio manual-EP8 MoE blog tuning suite after all 8 NPUs are free.

set -o pipefail

BASE_DIR=/data/sejin/baseline_26
PERF_DIR=/data/sejin/third_party/mindspeed-mm-26.0.0/examples/qwen3_5_audio/perf_tuning
RUNNER="$BASE_DIR/scripts/run_audio_perf_experiment.sh"

export QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-/data/sejin/models/Qwen3.5-35B-A3B}"
export WHISPER_MODEL_PATH="${WHISPER_MODEL_PATH:-/data/sejin/models/whisper-large-v3}"

MAX_BUSY_HBM_MB="${MAX_BUSY_HBM_MB:-10000}"
WAIT_FOR_FREE_NPU="${WAIT_FOR_FREE_NPU:-1}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-86400}"
POLL_SECONDS="${POLL_SECONDS:-60}"

max_hbm_used_mb() {
  python - <<'PY'
import re
import subprocess

out = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=20).stdout
vals = []
for line in out.splitlines():
    m = re.search(r"\|\s*\d+\s+\|\s+[\w:.]+\s+\|\s+\d+\s+\d+\s*/\s*\d+\s+(\d+)\s*/\s*(\d+)", line)
    if m:
        vals.append(int(m.group(1)))
print(max(vals) if vals else 0)
PY
}

wait_for_free_npu() {
  if [ "$WAIT_FOR_FREE_NPU" != "1" ]; then
    return 0
  fi
  local start
  start=$(date +%s)
  while true; do
    local now
    local max_hbm
    now=$(date +%s)
    max_hbm=$(max_hbm_used_mb)
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] max_hbm_used_mb=${max_hbm}, threshold=${MAX_BUSY_HBM_MB}"
    if [ "$max_hbm" -lt "$MAX_BUSY_HBM_MB" ]; then
      return 0
    fi
    if [ $((now - start)) -ge "$WAIT_TIMEOUT_SECONDS" ]; then
      echo "Timed out waiting for free NPUs after ${WAIT_TIMEOUT_SECONDS}s"
      return 1
    fi
    sleep "$POLL_SECONDS"
  done
}

run_one() {
  local port="$1"
  local tag="$2"
  local config="$3"
  local duration="$4"
  local interval="$5"
  local moe_phase="$6"
  echo "===== RUN ${tag} $(date '+%Y-%m-%d %H:%M:%S') ====="
  if [ "$moe_phase" = "1" ]; then
    MOE_PHASE_TIMING=1 MOE_PHASE_TIMING_SYNC=1 MOE_PHASE_RANKS=0 \
    MOE_PHASE_START_CALL=800 MOE_PHASE_LOG_EVERY=80 MOE_PHASE_MAX_LINES=40 \
    MASTER_PORT="$port" WATCHDOG_ENABLE=1 WATCHDOG_STALL_SECONDS=180 WATCHDOG_STARTUP_SECONDS=600 \
      bash "$RUNNER" "$tag" "$config" "$duration" 10 "$interval"
  else
    MASTER_PORT="$port" WATCHDOG_ENABLE=1 WATCHDOG_STALL_SECONDS=180 WATCHDOG_STARTUP_SECONDS=600 \
      bash "$RUNNER" "$tag" "$config" "$duration" 10 "$interval"
  fi
  echo "===== DONE ${tag} rc=$? $(date '+%Y-%m-%d %H:%M:%S') ====="
}

wait_for_free_npu || exit 1

python "$BASE_DIR/scripts/make_audio_perf_configs.py"

run_one 6070 \
  mbs2_fa2_fused_phaseprof_35 \
  "$PERF_DIR/mbs2_fa2_fused_phaseprof_35.yaml" \
  1200 0.5 1

run_one 6071 \
  mbs2_fa2_eager_ablation_35 \
  "$PERF_DIR/mbs2_fa2_eager_ablation_35.yaml" \
  1200 0.5 1

run_one 6072 \
  mbs2_fa2_mc2_probe_35 \
  "$PERF_DIR/mbs2_fa2_mc2_probe_35.yaml" \
  1200 0.5 1

run_one 6073 \
  mbs2_fa2_fused_bucket64_chunk1024_80 \
  "$PERF_DIR/mbs2_fa2_fused_bucket64_chunk1024_80.yaml" \
  1800 0.5 0

run_one 6074 \
  mbs2_fa2_fused_bucket64_chunk512_80 \
  "$PERF_DIR/mbs2_fa2_fused_bucket64_chunk512_80.yaml" \
  1800 0.5 0

run_one 6075 \
  mbs2_fa2_fused_bucket32_chunk512_80 \
  "$PERF_DIR/mbs2_fa2_fused_bucket32_chunk512_80.yaml" \
  1800 0.5 0

run_one 6076 \
  mbs2_fa2_fused_bucket32_chunk512_empty4_80 \
  "$PERF_DIR/mbs2_fa2_fused_bucket32_chunk512_empty4_80.yaml" \
  1800 0.5 0

run_one 6077 \
  mbs2_fa2_fused_bucket32_chunk512_rc_on_80 \
  "$PERF_DIR/mbs2_fa2_fused_bucket32_chunk512_rc_on_80.yaml" \
  1800 0.5 0

python "$BASE_DIR/scripts/write_qwen35_audio_moe_blog_report.py"
