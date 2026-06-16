#!/bin/bash
# Quick smoke test for pack format - verify it can run without errors

export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH
source /usr/local/Ascend/cann-8.5.0/set_env.sh 2>/dev/null
source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null

cd /data/sejin/qwenasr_qwen3p5moe_sft

echo "=========================================="
echo "Pack Format Smoke Test"
echo "=========================================="
echo "Testing pack format with minimal training steps..."
echo ""

# Run training with pack format for just 5 steps
torchrun --nproc_per_node=8 train_ep.py \
    --llm_path /mnt/shared_data_196/models/Qwen3-30B-A3B-Base \
    --asr_path /mnt/shared_data_196/models/Qwen3-ASR-1.7B \
    --tokenizer_path /mnt/shared_data_196/models/Qwen3-30B-A3B-Base \
    --train_data_path /data/sejin/data/train.json \
    --output_dir /data/sejin/output_pack_test \
    --batch_tokens 80000 \
    --max_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-6 \
    --num_epochs 1 \
    --max_steps 5 \
    --logging_steps 1 \
    --use_lora \
    --lora_rank 8 \
    --lora_alpha 16 \
    --gradient_checkpointing \
    --use_packed_format \
    2>&1 | tee pack_smoke_test.log

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Pack format smoke test PASSED"
    echo "Training completed 5 steps successfully"
else
    echo "❌ Pack format smoke test FAILED"
    echo "Exit code: $EXIT_CODE"
    echo "Check pack_smoke_test.log for details"
fi
echo "=========================================="

exit $EXIT_CODE
