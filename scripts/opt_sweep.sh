#!/bin/bash
# 优化迭代 sweep：基于 workflow 优化方案，逐项验证，每轮采指标
# 每个配置: 固定 seq 4096, 25步, save禁用, 算子已预编译, MC2规避基线
set -uo pipefail
TRAIN=/data/sejin/baseline_26/scripts/train_param.sh
MET=/data/sejin/baseline_26/metrics
LOGD=/data/sejin/baseline_26/logs
MON=/data/sejin/baseline_26/scripts/npu_monitor.py
PARSE=/data/sejin/baseline_26/scripts/parse_metrics.py
PY=/data/sejin/env/venv_26b/bin/python
cd /data/sejin/third_party/mindspeed-llm-26.0.0

run_opt () {
  local TAG="$1"; local MBS="$2"; local GBS="$3"; local PAD="$4"; shift 4
  local EXTRA="$*"
  echo "######## OPT $TAG : MBS=$MBS GBS=$GBS PAD=$PAD EXTRA=[$EXTRA] $(date +%H:%M:%S) ########"
  # 彻底清理
  for i in 1 2; do pkill -9 -f posttrain_gpt 2>/dev/null; pkill -9 -f torchrun 2>/dev/null; sleep 3; done
  sleep 3

  local RUNLOG="$LOGD/opt_${TAG}.log"
  rm -f "$RUNLOG" 2>/dev/null
  LOG_FILE="$RUNLOG" MBS=$MBS GBS=$GBS ITERS=25 SWEEP_PAD=$PAD SWEEP_EXTRA="$EXTRA" \
    nohup bash "$TRAIN" > "$LOGD/opt_${TAG}.stdout" 2>&1 &
  local TPID=$!

  # 等初始化 + 跑到 iteration 5 (确认没崩)
  local waited=0; local ok=0
  while [ $waited -lt 420 ]; do
    sleep 20; waited=$((waited+20))
    grep -qE "iteration  *5/" "$RUNLOG" 2>/dev/null && { ok=1; break; }
    grep -qiE "507018|out of memory|CANN error|RuntimeError:|ChildFailedError|ValueError:|\.so: cannot" "$RUNLOG" 2>/dev/null && { echo "[$TAG] CRASH"; break; }
    ps -p $TPID >/dev/null 2>&1 || { echo "[$TAG] EXIT early"; break; }
  done

  # 采 HBM/功耗 (运行中)
  if ps -p $TPID >/dev/null 2>&1; then
    "$PY" "$MON" "$MET/npu_${TAG}.json" 45 >/dev/null 2>&1 || true
  fi
  # 等训练结束
  local w2=0
  while ps -p $TPID >/dev/null 2>&1 && [ $w2 -lt 500 ]; do sleep 15; w2=$((w2+15)); done
  pkill -9 -f posttrain_gpt 2>/dev/null; sleep 3

  # 解析性能
  "$PY" "$PARSE" "$RUNLOG" 4096 3 > "$MET/perf_${TAG}.json" 2>&1 || true
  local PERF=$("$PY" -c "import json;d=json.load(open('$MET/perf_${TAG}.json'));s=d.get('step_ms',{});t=d.get('throughput',{});print(f\"step={s.get('mean')}ms TPS={t.get('samples_per_sec_TPS')} WPS={t.get('tokens_per_sec_WPS')} nan={d.get('nan_count')}\")" 2>/dev/null || echo "PARSE_FAIL")
  local HBM=$("$PY" -c "import json;d=json.load(open('$MET/npu_${TAG}.json'));h=d.get('hbm_used_mb',{});print(f\"HBM={h.get('peak')}/{h.get('total')}\")" 2>/dev/null || echo "HBM=?")
  echo "[$TAG] RESULT: $PERF $HBM"
}

# ===== 优化迭代清单 (基于 workflow 方案, 含 MC2 规避) =====
# R0 基线已有(verify_mc2fix). 这里从优化项开始.
# R1: 增大 batch mbs4 (方案1/5) — full recompute 下显存够不够
run_opt "R1_mbs4"        4  32  fixed ""
# R2: MoE alltoall 通信重叠 (方案2 主力) — 需 moe-tp-extend-ep
run_opt "R2_moeoverlap"  2  16  fixed "--moe-tp-extend-ep --moe-alltoall-overlap-comm"
# R3: DP 通信重叠 (方案3)
run_opt "R3_dpoverlap"   2  16  fixed "--overlap-grad-reduce --overlap-param-gather --reset-bucket-group-order"
# R4: R2+R3 组合
run_opt "R4_combo"       2  16  fixed "--moe-tp-extend-ep --moe-alltoall-overlap-comm --overlap-grad-reduce"

echo "ALL OPT DONE $(date +%H:%M:%S)"
