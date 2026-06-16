#!/bin/bash
# Qwen3.5-35B-A3B + Whisper-large-v3 语音多模态 LoRA SFT 启动脚本
# 使用经验证的 venv_qwen35 环境(transformers fc91372 / torch_npu 2.7.1)

# 必须先设 base PATH，保证 set_env.sh 内的 dirname/uname/cut 等可用
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH

cd /data/sejin/third_party/mindspeed-mm-26.0.0
source /usr/local/Ascend/cann-8.5.0/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# 激活独立 venv(用 PATH 方式,解析到 fc91372 源码版 transformers)
VENV=/data/sejin/env/venv_qwen35
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

mkdir -p logs
TRAIN_LOG="logs/qwen3_5_audio_$(date +%Y%m%d_%H%M%S).log"
echo "训练日志: $TRAIN_LOG"

"$VENV/bin/torchrun" \
    --nproc_per_node 8 \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr localhost \
    --master_port 6008 \
    mindspeed_mm/fsdp/train/trainer.py \
    examples/qwen3_5_audio/qwen3_5_35B_audio_config.yaml \
    > "$TRAIN_LOG" 2>&1
echo "退出码 $?"
echo "$TRAIN_LOG"
