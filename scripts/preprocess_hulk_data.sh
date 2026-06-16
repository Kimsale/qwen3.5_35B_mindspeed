#!/bin/bash
# ============================================================
# 数据预处理: ShareGPT/OpenAI messages jsonl → MindSpeed packed IndexedDataset
# 用于 xuchen2 Qwen3-Omni MoE 的 hulk 对齐训练
# 输入: train.jsonl ({"messages":[{role,content},...]})
# 输出: <prefix>_packed_{input_ids,attention_mask,labels}_document.{bin,idx}
# ============================================================
set -eo pipefail

source /data/sejin/baseline_26/scripts/env_cann85.sh

cd /data/sejin/third_party/mindspeed-llm-26.0.0
PYTHON_BIN=/data/sejin/env/venv_26b/bin/python

# ---- 参数 (可被环境变量覆盖) ----
INPUT="${INPUT:-/data/sejin/baseline_26/data_hulk_dist_30k/train.jsonl}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-/data/sejin/baseline_26/data_hulk_dist_30k/qwen3_sft}"
TOKENIZER="${TOKENIZER:-/data/sejin/models/Qwen3-30B-A3B-Base}"
SEQ_LEN="${SEQ_LEN:-8192}"
WORKERS="${WORKERS:-32}"

# OpenAI messages 格式映射: role/content/user/assistant/system
MAP_KEYS='{"messages":"messages","tags":{"role_tag":"role","content_tag":"content","user_tag":"user","assistant_tag":"assistant","system_tag":"system"}}'

echo "=== 数据预处理 (SharegptStyleInstructionHandler + pack) ==="
echo "输入: $INPUT"
echo "输出前缀: $OUTPUT_PREFIX"
echo "tokenizer: $TOKENIZER"
echo "seq_len(pack): $SEQ_LEN"
echo

$PYTHON_BIN preprocess_data.py \
    --input "$INPUT" \
    --tokenizer-name-or-path "$TOKENIZER" \
    --tokenizer-type PretrainedFromHF \
    --output-prefix "$OUTPUT_PREFIX" \
    --handler-name SharegptStyleInstructionHandler \
    --map-keys "$MAP_KEYS" \
    --prompt-type qwen3 \
    --seq-length $SEQ_LEN \
    --pack \
    --append-eod \
    --workers $WORKERS \
    --log-interval 1000

echo
echo "✅ 数据预处理完成"
ls -lh "${OUTPUT_PREFIX}_packed"*.{bin,idx} 2>/dev/null
