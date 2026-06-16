#!/bin/bash
# EP Training: 8 GPUs, EP=8 (each GPU handles 32 experts)
source /usr/local/Ascend/cann-8.5.0/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null
export PATH=/repo1/yjjiang11/env/.local_qwen3p5/bin:$PATH
export PYTHONPATH=/repo1/yjjiang11/env/.local_qwen3p5/lib/python3.10/site-packages:$PYTHONPATH

export CUDA_DEVICE_MAX_CONNECTIONS=1
export ASCEND_GLOBAL_LOG_LEVEL=3
export TASK_QUEUE_ENABLE=2
export HCCL_CONNECT_TIMEOUT=1200
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"

# Increase file descriptor limit for large-scale LMDB training (264+ LMDBs)
ulimit -n 65536 2>/dev/null || true

# Multi-node config (defaults to single node)
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-13525}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs"
mkdir -p ${LOG_DIR}

echo "=========================================="
echo "Multi-node Training Configuration"
echo "=========================================="
echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "Total nodes: ${NNODES}"
echo "Current node rank: ${NODE_RANK}"
echo "GPUs per node: ${NPROC_PER_NODE}"
echo "Total world size: $((NNODES * NPROC_PER_NODE))"
echo "=========================================="

# if [ "$NNODES" -gt 1 ]; then
#     echo "Multi-node mode: Waiting for all ${NNODES} nodes to connect..."
# else
#     echo "Single-node mode"
# fi
# echo ""

torchrun \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --nproc_per_node=${NPROC_PER_NODE} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    train_ep.py \
    --batch_tokens 100000 \
    --max_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-5 \
    --weight_decay 0.001 \
    --warmup_steps 200 \
    --num_epochs 50 \
    --max_tokens_persample 3000 \
    --logging_steps 10 \
    --save_steps 600 \
    --max_grad_norm 1.0 \
    --output_dir output_ep_fast \
    2>&1 | tee ${LOG_DIR}/train_ep_node${NODE_RANK}_${TIMESTAMP}.log
