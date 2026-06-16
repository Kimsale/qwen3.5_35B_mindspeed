#!/bin/bash
# ============================================================
# 权重转换: xuchen2 Qwen3-Omni-30B-A3B MoE → MindSpeed MCore
# 目标配置: TP=1/PP=1/EP=8 (对齐 hulk 并行策略)
# 架构: 48层 MoE, 128专家/topk8, hidden_size=2048
# ============================================================
set -eo pipefail

# ---- 固定 CANN 8.5 环境（含 PYTHONPATH + venv_26b）----
source /data/sejin/baseline_26/scripts/env_cann85.sh

cd /data/sejin/third_party/mindspeed-llm-26.0.0
PYTHON_BIN=/data/sejin/env/venv_26b/bin/python

# ---- 路径配置 ----
# 源模型: 已从 Qwen3-Omni MoE 抽取出的 thinker text 子模块 (标准 Qwen3MoeForCausalLM)
# 抽取脚本: extract_thinker_text.py (剥离 audio_tower/visual, 平铺 config)
HF_MODEL_DIR="/data/sejin/models/Qwen3-Omni-30B-A3B-text-extracted"
# 目标路径: MindSpeed MCore 格式 (TP1/PP1/EP8)
OUTPUT_DIR="/data/sejin/models/Qwen3-Omni-30B-A3B-hulk_tp1_pp1_ep8"
TOKENIZER_PATH="$HF_MODEL_DIR"

# ---- 目标并行配置（对齐 hulk）----
TP=1
PP=1
EP=8

echo "=== 权重转换: HF → MCore TP${TP}/PP${PP}/EP${EP} ==="
echo "源模型: $HF_MODEL_DIR"
echo "目标: $OUTPUT_DIR"
echo "预计耗时: 20-40 分钟"
echo "预计输出: ~60GB (30B MoE 模型参数 bf16)"
echo

# ---- 确认源模型存在 ----
if [ ! -f "$HF_MODEL_DIR/config.json" ]; then
  echo "❌ 错误: 源模型不存在 $HF_MODEL_DIR/config.json"
  exit 1
fi

# ---- 确认磁盘空间 ----
mkdir -p "$(dirname "$OUTPUT_DIR")"
avail_gb=$(df -BG "$(dirname "$OUTPUT_DIR")" 2>/dev/null | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$avail_gb" -lt 70 ]; then
  echo "⚠️  警告: 磁盘可用空间仅 ${avail_gb}GB，建议至少 70GB"
  echo "按 Ctrl+C 取消，或等待 5 秒继续..."
  sleep 5
fi

# ---- 创建输出目录 ----
mkdir -p "$OUTPUT_DIR"

# ---- 转换命令 ----
# 基于 MindSpeed 26.0.0 的 convert_ckpt_v2.py
# 参考: examples/mcore/qwen3_moe/ckpt_convert_qwen3_moe_235b_hf2mcore.sh
echo "开始转换 $(date '+%Y-%m-%d %H:%M:%S')"

$PYTHON_BIN convert_ckpt_v2.py \
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
  echo
  ls -lh "$OUTPUT_DIR/" | head -20
  echo
  echo "下一步: 使用 hulk 对齐训练脚本"
  echo "  bash /data/sejin/baseline_26/scripts/train_xuchen2_hulk_aligned.sh"
else
  echo
  echo "❌ 转换失败，退出码 $exit_code"
  exit $exit_code
fi
