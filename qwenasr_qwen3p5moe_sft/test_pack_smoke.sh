#!/bin/bash
# Quick smoke test for pack format - verify it can run without errors
# Uses the same verified default paths as run_ep_lora.sh

source /usr/local/Ascend/cann-8.5.0/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null
export PATH=/repo1/yjjiang11/env/.local_qwen3p5/bin:$PATH
export PYTHONPATH=/repo1/yjjiang11/env/.local_qwen3p5/lib/python3.10/site-packages:$PYTHONPATH

export CUDA_DEVICE_MAX_CONNECTIONS=1
export ASCEND_GLOBAL_LOG_LEVEL=3
export TASK_QUEUE_ENABLE=2
export HCCL_CONNECT_TIMEOUT=1200
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"

cd /data/sejin/qwenasr_qwen3p5moe_sft

MODE=${1:-pack}   # pack | pad
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_PORT=${MASTER_PORT:-13577}

if [ "$MODE" = "pack" ]; then
    EXTRA_ARGS="--use_packed_format"
    OUT_DIR="output_pack_smoke"
else
    EXTRA_ARGS=""
    OUT_DIR="output_pad_smoke"
fi

echo "=========================================="
echo "Pack Format Smoke Test (mode=$MODE)"
echo "=========================================="
echo "Running 5 optimizer steps to verify training works..."
echo ""

torchrun --nproc_per_node=${NPROC_PER_NODE} --master_port=${MASTER_PORT} train_ep.py \
    --batch_tokens 100000 \
    --max_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-5 \
    --num_epochs 1 \
    --max_tokens_persample 3000 \
    --max_steps 5 \
    --logging_steps 1 \
    --max_grad_norm 1.0 \
    --use_lora \
    --lora_rank 32 \
    --lora_alpha 64 \
    --output_dir ${OUT_DIR} \
    --sampler_mode global_random \
    ${EXTRA_ARGS} \
    2>&1 | tee ${OUT_DIR}_smoke.log

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Smoke test (mode=$MODE) PASSED - 5 steps completed"
else
    echo "❌ Smoke test (mode=$MODE) FAILED - exit code: $EXIT_CODE"
    echo "Check ${OUT_DIR}_smoke.log for details"
fi
echo "=========================================="
exit $EXIT_CODE
