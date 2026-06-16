#!/bin/bash
# Qwen3.5-35B-A3B + Whisper-large-v3 audio LoRA SFT
# Corrected duration distribution, 60-step performance run with NPU monitor.

set -o pipefail

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
export MULTI_STREAM_MEMORY_REUSE=2
export TASK_QUEUE_ENABLE=2
export ASCEND_LAUNCH_BLOCKING=0
export ACLNN_CACHE_LIMIT=100000
export CPU_AFFINITY_CONF=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_CONNECT_TIMEOUT=1800
export TOKENIZERS_PARALLELISM=false
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export AUDIO_PLACEHOLDER="<|AUDIO|>"

mkdir -p "$BASE_DIR/logs" "$BASE_DIR/metrics" "$BASE_DIR/output/ckpt_audio_distfix60"
TS=$(date +%Y%m%d_%H%M%S)
TRAIN_LOG="$BASE_DIR/logs/audio_distfix60_${TS}.log"
MONITOR_JSON="$BASE_DIR/metrics/audio_distfix60_${TS}_npu.json"
MONITOR_LOG="$BASE_DIR/logs/audio_distfix60_${TS}_monitor.log"

echo "训练日志: $TRAIN_LOG"
echo "监控 JSON: $MONITOR_JSON"

pkill -9 -f "trainer.py.*distfix60_config.yaml" 2>/dev/null || true
sleep 3

"$VENV/bin/python" "$BASE_DIR/scripts/npu_monitor.py" "$MONITOR_JSON" 1200 > "$MONITOR_LOG" 2>&1 &
MON_PID=$!

"$VENV/bin/torchrun" \
    --nproc_per_node 8 \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr localhost \
    --master_port 6031 \
    mindspeed_mm/fsdp/train/trainer.py \
    examples/qwen3_5_audio/distfix60_config.yaml \
    > "$TRAIN_LOG" 2>&1
RC=$?

kill "$MON_PID" 2>/dev/null || true
wait "$MON_PID" 2>/dev/null || true

echo "退出码 $RC"
echo "$TRAIN_LOG"
echo "$MONITOR_JSON"
exit "$RC"
