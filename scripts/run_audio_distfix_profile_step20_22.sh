#!/bin/bash
# Qwen3.5-35B-A3B + Whisper-large-v3 audio LoRA SFT
# Profiler-only run for pure training window iteration 20-22.

set -o pipefail

export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH

BASE_DIR=/data/sejin/baseline_26
MS_DIR=/data/sejin/third_party/mindspeed-mm-26.0.0
VENV=/data/sejin/env/venv_qwen35
PROFILE_DIR="$BASE_DIR/profiling/audio_distfix_step20_22"

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

mkdir -p "$BASE_DIR/logs" "$BASE_DIR/profiling" "$BASE_DIR/output/ckpt_audio_distfix_profile"
rm -rf "$PROFILE_DIR"
mkdir -p "$PROFILE_DIR"

TS=$(date +%Y%m%d_%H%M%S)
TRAIN_LOG="$BASE_DIR/logs/audio_distfix_profile_step20_22_${TS}.log"

echo "训练日志: $TRAIN_LOG"
echo "Profiler 输出: $PROFILE_DIR"

pkill -9 -f "trainer.py.*distfix_profile_step20_22_config.yaml" 2>/dev/null || true
sleep 3

"$VENV/bin/torchrun" \
    --nproc_per_node 8 \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr localhost \
    --master_port 6032 \
    mindspeed_mm/fsdp/train/trainer.py \
    examples/qwen3_5_audio/distfix_profile_step20_22_config.yaml \
    > "$TRAIN_LOG" 2>&1
RC=$?

echo "退出码 $RC"
echo "$TRAIN_LOG"
echo "$PROFILE_DIR"
exit "$RC"
