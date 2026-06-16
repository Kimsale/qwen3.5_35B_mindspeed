#!/bin/bash
# 自动化多配置训练+指标采集：串行跑完所有配置，每个采集 HBM/步耗时/吞吐
# 用法: bash auto_sweep.sh
set -uo pipefail

ENVS=/data/sejin/baseline_26/scripts/env_cann85.sh
TRAIN=/data/sejin/baseline_26/scripts/train_param.sh
MET=/data/sejin/baseline_26/metrics
LOGD=/data/sejin/baseline_26/logs
MON=/data/sejin/baseline_26/scripts/npu_monitor.py
PY=/data/sejin/env/venv_26b/bin/python
cd /data/sejin/third_party/mindspeed-llm-26.0.0
mkdir -p "$MET"

run_one () {
  local TAG="$1"; local MBS="$2"; local GBS="$3"; local ITERS="$4"; local PAD="$5"; shift 5
  local EXTRA="$*"
  echo "######## RUN $TAG : MBS=$MBS GBS=$GBS ITERS=$ITERS PAD=$PAD EXTRA=$EXTRA ########"
  pkill -9 -f posttrain_gpt.py 2>/dev/null; sleep 4

  local RUNLOG="$LOGD/sweep_${TAG}.log"
  # PAD=fixed 时去掉 --no-pad-to-seq-lengths（固定 seq，占满显存+可比）
  PADFLAG=""
  [ "$PAD" = "fixed" ] && PADFLAG="STRIP_NOPAD=1"

  # 让 train_param 直接写入 RUNLOG（避免 tee 与 nohup 重定向冲突）
  LOG_FILE="$RUNLOG" MBS=$MBS GBS=$GBS ITERS=$ITERS SWEEP_EXTRA="$EXTRA" SWEEP_PAD="$PAD" \
    nohup bash "$TRAIN" > "$LOGD/sweep_${TAG}.stdout" 2>&1 &
  local TPID=$!
  echo "$TPID" > "$LOGD/train.pid"

  # 等初始化（加载66G权重约2分钟），轮询日志出现 iteration 1
  local waited=0
  while [ $waited -lt 240 ]; do
    sleep 15; waited=$((waited+15))
    grep -q "iteration  *1/" "$RUNLOG" 2>/dev/null && break
    grep -qiE "out of memory|OOM|Error:|Traceback|FAILED" "$RUNLOG" 2>/dev/null && { echo "[$TAG] 早期错误"; break; }
    ps -p $TPID >/dev/null 2>&1 || { echo "[$TAG] 进程提前退出"; break; }
  done

  # 训练运行中则采集 HBM/功耗 60s
  if ps -p $TPID >/dev/null 2>&1; then
    "$PY" "$MON" "$MET/npu_${TAG}.json" 60 >/dev/null 2>&1 || true
  fi

  # 等训练自然结束（最多再等 600s）
  local w2=0
  while ps -p $TPID >/dev/null 2>&1 && [ $w2 -lt 600 ]; do sleep 15; w2=$((w2+15)); done
  pkill -9 -f posttrain_gpt.py 2>/dev/null; sleep 4

  # 解析指标
  "$PY" /data/sejin/baseline_26/scripts/parse_metrics.py "$RUNLOG" 4096 5 > "$MET/perf_${TAG}.json" 2>&1 || true
  echo "[$TAG] done -> $MET/perf_${TAG}.json , $MET/npu_${TAG}.json"
}

# ===== 配置清单 =====
# 基线(变长,官方配置) - 25步足够采集稳定指标
run_one "baseline_var_mbs2"   2  16  25 var
# 占满显存探测：固定seq, 逐步加大 mbs
run_one "fixed_mbs2"          2  16  25 fixed
run_one "fixed_mbs4"          4  32  25 fixed
run_one "fixed_mbs8"          8  64  25 fixed

echo "ALL SWEEP DONE"
