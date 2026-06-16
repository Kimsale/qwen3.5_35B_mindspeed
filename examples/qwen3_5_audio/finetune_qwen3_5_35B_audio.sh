#!/bin/bash
# =============================================================================
# Qwen3.5-35B-A3B + Whisper-large-v3 语音多模态 LoRA SFT 启动脚本（FSDP2 栈）
# 严格遵循项目环境约束：锁定 CANN 8.5，禁用 CANN 8.1。
# =============================================================================

# ---- 固定 CANN 8.5 环境（项目硬性约束）----
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH
source /usr/local/Ascend/cann-8.5.0/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# ---- FSDP2 栈运行开关（参考 qwen3_5 官方脚本）----
export NON_MEGATRON=true
export MULTI_STREAM_MEMORY_REUSE=2
export TASK_QUEUE_ENABLE=2
export ASCEND_LAUNCH_BLOCKING=0
export ACLNN_CACHE_LIMIT=100000
export CPU_AFFINITY_CONF=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# ---- 分布式参数（单机 8 卡，按需改 NNODES/NODE_RANK）----
NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

logfile=$(date +%Y%m%d)_$(date +%H%M%S)
mkdir -p logs

# 入口与 qwen3_5 一致：FSDP2 trainer + 我们的 audio 配置
torchrun $DISTRIBUTED_ARGS mindspeed_mm/fsdp/train/trainer.py \
    examples/qwen3_5_audio/qwen3_5_35B_audio_config.yaml \
    2>&1 | tee logs/train_qwen3_5_audio_${logfile}.log
