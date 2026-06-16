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
PROBE_SKIP_STEPS="${PROBE_SKIP_STEPS:-20}"
MAIN_SKIP_STEPS="${MAIN_SKIP_STEPS:-30}"
WATCHDOG_STALL_SECONDS_DEFAULT="${WATCHDOG_STALL_SECONDS:-900}"
WATCHDOG_STARTUP_SECONDS_DEFAULT="${WATCHDOG_STARTUP_SECONDS:-900}"
RUN_FUSED_PHASEPROF="${RUN_FUSED_PHASEPROF:-1}"
RUN_EAGER_ABLATION="${RUN_EAGER_ABLATION:-1}"
RUN_MC2_PROBE="${RUN_MC2_PROBE:-1}"
RUN_FUSED_MAIN="${RUN_FUSED_MAIN:-1}"

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

npu_process_count() {
  python - <<'PY'
import re
import subprocess

out = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=20).stdout
in_process_table = False
pids = set()
for line in out.splitlines():
    if "Process id" in line:
        in_process_table = True
        continue
    if not in_process_table:
        continue
    m = re.search(r"\|\s*\d+\s+\d+\s*\|\s*(\d+)\s*\|", line)
    if m:
        pids.add(m.group(1))
print(len(pids))
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
    local proc_count
    now=$(date +%s)
    max_hbm=$(max_hbm_used_mb)
    proc_count=$(npu_process_count)
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] npu_process_count=${proc_count}, max_hbm_used_mb=${max_hbm}, threshold=${MAX_BUSY_HBM_MB}"
    if [ "$proc_count" -eq 0 ] && [ "$max_hbm" -lt "$MAX_BUSY_HBM_MB" ]; then
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
  local skip_steps="$7"
  wait_for_free_npu || return 1
  echo "===== RUN ${tag} $(date '+%Y-%m-%d %H:%M:%S') ====="
  if [ "$moe_phase" = "1" ]; then
    MOE_PHASE_TIMING=1 MOE_PHASE_TIMING_SYNC="${MOE_PHASE_TIMING_SYNC:-1}" MOE_PHASE_RANKS=0 \
    MOE_PHASE_START_CALL=800 MOE_PHASE_LOG_EVERY=80 MOE_PHASE_MAX_LINES=40 \
    MASTER_PORT="$port" WATCHDOG_ENABLE=1 WATCHDOG_STALL_SECONDS="$WATCHDOG_STALL_SECONDS_DEFAULT" WATCHDOG_STARTUP_SECONDS="$WATCHDOG_STARTUP_SECONDS_DEFAULT" \
      bash "$RUNNER" "$tag" "$config" "$duration" "$skip_steps" "$interval"
  else
    MASTER_PORT="$port" WATCHDOG_ENABLE=1 WATCHDOG_STALL_SECONDS="$WATCHDOG_STALL_SECONDS_DEFAULT" WATCHDOG_STARTUP_SECONDS="$WATCHDOG_STARTUP_SECONDS_DEFAULT" \
      bash "$RUNNER" "$tag" "$config" "$duration" "$skip_steps" "$interval"
  fi
  echo "===== DONE ${tag} rc=$? $(date '+%Y-%m-%d %H:%M:%S') ====="
}

wait_for_free_npu || exit 1

python "$BASE_DIR/scripts/make_audio_perf_configs.py"

if [ "$RUN_FUSED_PHASEPROF" = "1" ]; then
  run_one 6070 \
    mbs2_fa2_fused_phaseprof_35 \
    "$PERF_DIR/mbs2_fa2_fused_phaseprof_35.yaml" \
    1800 0.5 1 "$PROBE_SKIP_STEPS"
else
  echo "===== SKIP mbs2_fa2_fused_phaseprof_35 $(date '+%Y-%m-%d %H:%M:%S') ====="
fi

if [ "$RUN_EAGER_ABLATION" = "1" ]; then
  run_one 6071 \
    mbs2_fa2_eager_ablation_35 \
    "$PERF_DIR/mbs2_fa2_eager_ablation_35.yaml" \
    1800 0.5 1 "$PROBE_SKIP_STEPS"
else
  echo "===== SKIP mbs2_fa2_eager_ablation_35 $(date '+%Y-%m-%d %H:%M:%S') ====="
fi

if [ "$RUN_MC2_PROBE" = "1" ]; then
  run_one 6072 \
    mbs2_fa2_mc2_probe_35 \
    "$PERF_DIR/mbs2_fa2_mc2_probe_35.yaml" \
    1800 0.5 1 "$PROBE_SKIP_STEPS"
else
  echo "===== SKIP mbs2_fa2_mc2_probe_35 $(date '+%Y-%m-%d %H:%M:%S') ====="
fi

if [ "$RUN_FUSED_MAIN" = "1" ]; then
  run_one 6073 \
    mbs2_fa2_fused_bucket64_chunk1024_80 \
    "$PERF_DIR/mbs2_fa2_fused_bucket64_chunk1024_80.yaml" \
    2400 0.5 0 "$MAIN_SKIP_STEPS"

  run_one 6074 \
    mbs2_fa2_fused_bucket64_chunk512_80 \
    "$PERF_DIR/mbs2_fa2_fused_bucket64_chunk512_80.yaml" \
    2400 0.5 0 "$MAIN_SKIP_STEPS"

  run_one 6075 \
    mbs2_fa2_fused_bucket32_chunk512_80 \
    "$PERF_DIR/mbs2_fa2_fused_bucket32_chunk512_80.yaml" \
    2400 0.5 0 "$MAIN_SKIP_STEPS"

  run_one 6076 \
    mbs2_fa2_fused_bucket32_chunk512_empty4_80 \
    "$PERF_DIR/mbs2_fa2_fused_bucket32_chunk512_empty4_80.yaml" \
    2400 0.5 0 "$MAIN_SKIP_STEPS"

  run_one 6077 \
    mbs2_fa2_fused_bucket32_chunk512_rc_on_80 \
    "$PERF_DIR/mbs2_fa2_fused_bucket32_chunk512_rc_on_80.yaml" \
    2400 0.5 0 "$MAIN_SKIP_STEPS"
else
  echo "===== SKIP fused main runs $(date '+%Y-%m-%d %H:%M:%S') ====="
fi

python "$BASE_DIR/scripts/write_qwen35_audio_moe_blog_report.py"
