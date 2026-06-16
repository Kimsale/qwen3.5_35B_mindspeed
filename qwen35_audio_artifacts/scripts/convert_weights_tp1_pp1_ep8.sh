#!/bin/bash
# ============================================================
# 权重转换: HF Qwen3-Omni-30B-A3B → MCore TP1/PP1/EP8
# 用于 Hulk 对标基线（TP1/CP2/EP8 配置）
# 基于官方 examples/mcore/qwen3_moe/ckpt_convert_qwen3_moe_hf2mcore.sh
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
echo "源模型: $HF_MODEL_DIR"
echo "目标目录: $OUTPUT_DIR"
echo "预计耗时: 15-30 分钟"
echo "预计输出: ~60GB (30B 模型参数 bf16)"
echo

# ---- 确认源模型存在 ----
if [ ! -f "$HF_MODEL_DIR/config.json" ]; then
  echo "❌ 错误: 源模型不存在 $HF_MODEL_DIR/config.json"
  exit 1
fi

# ---- 确认磁盘空间 ----
avail_gb=$(df -BG "$OUTPUT_DIR" 2>/dev/null | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$avail_gb" -lt 70 ]; then
  echo "⚠️  警告: 磁盘可用空间仅 ${avail_gb}GB，建议至少 70GB"
  echo "按 Ctrl+C 取消，或等待 5 秒继续..."
  sleep 5
fi

# ---- 创建输出目录 ----
mkdir -p "$OUTPUT_DIR"

# ---- 转换命令（基于官方示例，改路径和并行配置）----
# 显式指定 num_layers，避免 Omni 多模态模型的 config.json 解析失败
echo "开始转换 $(date '+%Y-%m-%d %H:%M:%S')"

python convert_ckpt_v2.py \
  --load-model-type hf \
  --save-model-type mg \
  --target-tensor-parallel-size $TP \
  --target-pipeline-parallel-size $PP \
  --target-expert-parallel-size $EP \
  --load-dir "$HF_MODEL_DIR" \
  --save-dir "$OUTPUT_DIR" \
  --model-type-hf qwen3-moe \
  --num-layers 48 \
  --hidden-size 2048 \
  --moe-grouped-gemm

exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo
  echo "✅ 转换成功 $(date '+%Y-%m-%d %H:%M:%S')"
  echo "输出目录: $OUTPUT_DIR"
  ls -lh "$OUTPUT_DIR/" | head -20
else
  echo
  echo "❌ 转换失败，退出码 $exit_code"
  exit $exit_code
fi
