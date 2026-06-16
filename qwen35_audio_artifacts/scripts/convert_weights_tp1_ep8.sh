#!/bin/bash
# ============================================================
# 权重转换: HF Qwen3-Omni-30B-A3B → MCore TP1/PP1/EP8
# 用于 Hulk 对标基线（TP1/CP2/EP8 配置）
# ============================================================
set -eo pipefail

# ---- 固定 CANN 8.5 环境 ----
source /data/sejin/baseline_26/scripts/env_cann85.sh

cd /data/sejin/third_party/mindspeed-llm-26.0.0

# ---- 路径配置 ----
HF_MODEL_DIR="/data/xuchen2/model/Qwen3-Omni-30B-A3B-Captioner"
OUTPUT_DIR="/data/sejin/checkpoints/qwen3_omni_30b_a3b_mcore_tp1_pp1_ep8"
TOKENIZER_PATH="$HF_MODEL_DIR"

# ---- 目标并行配置（与 Hulk 对齐）----
TP=1
PP=1
EP=8

echo "=== 权重转换: HF → MCore TP${TP}/PP${PP}/EP${EP} ==="
echo "源: $HF_MODEL_DIR"
echo "目标: $OUTPUT_DIR"
echo

# ---- 确认源模型存在 ----
if [ ! -f "$HF_MODEL_DIR/config.json" ]; then
  echo "错误: 源模型不存在 $HF_MODEL_DIR/config.json"
  exit 1
fi

# ---- 创建输出目录 ----
mkdir -p "$OUTPUT_DIR"

# ---- 转换命令 ----
python tools/checkpoint/convert_ckpt.py \
  --model-type-hf qwen3-moe \
  --model-type GPT \
  --load-dir "$HF_MODEL_DIR" \
  --save-dir "$OUTPUT_DIR" \
  --tokenizer-model "$TOKENIZER_PATH" \
  --target-tensor-parallel-size $TP \
  --target-pipeline-parallel-size $PP \
  --target-expert-parallel-size $EP \
  --model-type-hf qwen3-moe \
  --num-experts 128 \
  --moe-router-topk 8 \
  --w-pack
