#!/bin/bash
# Run one Qwen3.5 audio training perf experiment with full post-warmup analysis.

set -o pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <tag> <config_path> [duration_s] [skip_steps] [monitor_interval_s]"
  exit 2
fi

TAG="$1"
CONFIG_PATH="$2"
DURATION="${3:-1500}"
SKIP_STEPS="${4:-10}"
MONITOR_INTERVAL="${5:-1.0}"

export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH

BASE_DIR=/data/sejin/baseline_26
MS_DIR=/data/sejin/third_party/mindspeed-mm-26.0.0
VENV=/data/sejin/env/venv_qwen35

cd "$MS_DIR" || exit 1
source /usr/local/Ascend/cann-8.5.0/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"

export NON_MEGATRON=true
export MULTI_STREAM_MEMORY_REUSE="${MULTI_STREAM_MEMORY_REUSE:-2}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-2}"
export ASCEND_LAUNCH_BLOCKING=0
export ACLNN_CACHE_LIMIT=100000
export CPU_AFFINITY_CONF=1
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export HCCL_CONNECT_TIMEOUT=1800
export TOKENIZERS_PARALLELISM=false
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export AUDIO_PLACEHOLDER="<|AUDIO|>"

mkdir -p "$BASE_DIR/logs" "$BASE_DIR/metrics" "$BASE_DIR/reports/perf_runs" "$BASE_DIR/output/$TAG"
TS=$(date +%Y%m%d_%H%M%S)
TRAIN_LOG="$BASE_DIR/logs/${TAG}_${TS}.log"
MONITOR_JSON="$BASE_DIR/metrics/${TAG}_${TS}_npu_full.json"
MONITOR_LOG="$BASE_DIR/logs/${TAG}_${TS}_monitor.log"
WATCHDOG_LOG="$BASE_DIR/logs/${TAG}_${TS}_watchdog.log"
OUT_JSON="$BASE_DIR/metrics/${TAG}_${TS}_analysis.json"
OUT_MD="$BASE_DIR/reports/perf_runs/${TAG}_${TS}.md"

echo "TAG: $TAG"
echo "CONFIG: $CONFIG_PATH"
echo "TRAIN_LOG: $TRAIN_LOG"
echo "MONITOR_JSON: $MONITOR_JSON"
echo "WATCHDOG_LOG: $WATCHDOG_LOG"
echo "ANALYSIS_JSON: $OUT_JSON"
echo "ANALYSIS_MD: $OUT_MD"

pkill -9 -f "trainer.py.*$(basename "$CONFIG_PATH")" 2>/dev/null || true
sleep 3

"$VENV/bin/python" "$BASE_DIR/scripts/npu_monitor_full.py" "$MONITOR_JSON" "$DURATION" "$MONITOR_INTERVAL" > "$MONITOR_LOG" 2>&1 &
MON_PID=$!

collect_descendants() {
  local roots="$*"
  local all=""
  local frontier="$roots"
  while [ -n "$frontier" ]; do
    local next=""
    for pid in $frontier; do
      local children
      children=$(pgrep -P "$pid" 2>/dev/null || true)
      if [ -n "$children" ]; then
        next="$next $children"
        all="$all $children"
      fi
    done
    frontier="$next"
  done
  echo "$all"
}

dump_watchdog_state() {
  local reason="$1"
  local descendants
  descendants=$(collect_descendants "$TRAIN_PID")
  {
    echo "===== WATCHDOG $(date '+%Y-%m-%d %H:%M:%S') ====="
    echo "reason: $reason"
    echo "tag: $TAG"
    echo "config: $CONFIG_PATH"
    echo "train_pid: $TRAIN_PID"
    echo "descendants:$descendants"
    echo
    echo "----- npu-smi info -----"
    npu-smi info 2>&1 || true
    echo
    echo "----- process tree -----"
    ps -eo pid,ppid,stat,etime,pcpu,pmem,cmd | grep -E "($TRAIN_PID|torchrun|trainer.py|npu_monitor_full.py)" | grep -v grep || true
    echo
    echo "----- recent train log -----"
    tail -n 160 "$TRAIN_LOG" 2>&1 || true
    echo
    echo "----- SIGUSR1 python stack dump requested -----"
  } >> "$WATCHDOG_LOG"
  for pid in $descendants; do
    kill -USR1 "$pid" 2>/dev/null || true
  done
  sleep 8
  {
    echo
    echo "----- post-SIGUSR1 train log tail -----"
    tail -n 220 "$TRAIN_LOG" 2>&1 || true
    echo "===== WATCHDOG END $(date '+%Y-%m-%d %H:%M:%S') ====="
    echo
  } >> "$WATCHDOG_LOG"
}

watch_training() {
  local start_epoch
  local last_progress_epoch
  local last_iter
  local seen_iter
  local startup_timeout
  local stall_timeout
  start_epoch=$(date +%s)
  last_progress_epoch="$start_epoch"
  last_iter=""
  seen_iter=0
  startup_timeout="${WATCHDOG_STARTUP_SECONDS:-420}"
  stall_timeout="${WATCHDOG_STALL_SECONDS:-120}"

  while kill -0 "$TRAIN_PID" 2>/dev/null; do
    sleep "${WATCHDOG_POLL_SECONDS:-10}"
    local now
    local iter
    now=$(date +%s)
    iter=$(grep -aoE "iteration[[:space:]]+[0-9]+/" "$TRAIN_LOG" 2>/dev/null | tail -n 1 | grep -oE "[0-9]+" | head -n 1 || true)
    if [ -n "$iter" ] && [ "$iter" != "$last_iter" ]; then
      last_iter="$iter"
      last_progress_epoch="$now"
      seen_iter=1
      continue
    fi
    if [ "$seen_iter" -eq 0 ] && [ $((now - start_epoch)) -gt "$startup_timeout" ]; then
      dump_watchdog_state "no iteration log for ${startup_timeout}s during startup"
      kill -TERM "$TRAIN_PID" 2>/dev/null || true
      sleep 15
      kill -KILL "$TRAIN_PID" 2>/dev/null || true
      return
    fi
    if [ "$seen_iter" -eq 1 ] && [ $((now - last_progress_epoch)) -gt "$stall_timeout" ]; then
      dump_watchdog_state "iteration stalled after step ${last_iter} for ${stall_timeout}s"
      kill -TERM "$TRAIN_PID" 2>/dev/null || true
      sleep 15
      kill -KILL "$TRAIN_PID" 2>/dev/null || true
      return
    fi
  done
}

MASTER_PORT="${MASTER_PORT:-6045}"
"$VENV/bin/torchrun" \
    --nproc_per_node 8 \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr localhost \
    --master_port "$MASTER_PORT" \
    mindspeed_mm/fsdp/train/trainer.py \
    "$CONFIG_PATH" \
    > "$TRAIN_LOG" 2>&1 &
TRAIN_PID=$!

if [ "${WATCHDOG_ENABLE:-1}" = "1" ]; then
  watch_training &
  WATCHDOG_PID=$!
else
  WATCHDOG_PID=""
fi

wait "$TRAIN_PID"
RC=$?

if [ -n "$WATCHDOG_PID" ]; then
  kill "$WATCHDOG_PID" 2>/dev/null || true
  wait "$WATCHDOG_PID" 2>/dev/null || true
fi

kill "$MON_PID" 2>/dev/null || true
wait "$MON_PID" 2>/dev/null || true

if [ -f "$MONITOR_JSON" ]; then
  "$VENV/bin/python" "$BASE_DIR/scripts/analyze_audio_perf_run.py" \
    --tag "$TAG" \
    --config "$CONFIG_PATH" \
    --train-log "$TRAIN_LOG" \
    --monitor-json "$MONITOR_JSON" \
    --out-json "$OUT_JSON" \
    --out-md "$OUT_MD" \
    --skip-steps "$SKIP_STEPS"
fi

echo "EXIT_CODE: $RC"
echo "$TRAIN_LOG"
echo "$MONITOR_JSON"
echo "$OUT_JSON"
echo "$OUT_MD"
exit "$RC"
