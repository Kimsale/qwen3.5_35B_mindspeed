#!/bin/bash
# 快速探测最大可用 micro-batch-size（二分 or 指数增长）
set -eo pipefail
source /data/sejin/baseline_26/scripts/env_cann85.sh >/dev/null 2>&1

for BS in 2 4 6 8; do
  echo "=== 测试 micro-batch=$BS (3 步快速测) ==="
  LOG=/data/sejin/baseline_26/logs/bs_test_${BS}.log
  sed "s/--micro-batch-size 1/--micro-batch-size $BS/" /data/sejin/baseline_26/scripts/train_baseline_lora.sh | \
    sed "s/--train-iters 50/--train-iters 3/" > /tmp/train_bs${BS}.sh
  
  timeout 300 bash /tmp/train_bs${BS}.sh > "$LOG" 2>&1 &
  PID=$!
  sleep 180
  
  if ps -p $PID >/dev/null 2>&1; then
    echo "  bs=$BS 运行中（未 OOM），继续测"
    kill $PID 2>/dev/null; wait $PID 2>/dev/null
  else
    grep -iE "out of memory|OOM|alloc.*fail" "$LOG" >/dev/null && echo "  bs=$BS OOM！" && exit 0
    echo "  bs=$BS 其他错误，查看 $LOG"
  fi
done
echo "bs=8 未 OOM，可以更高"
