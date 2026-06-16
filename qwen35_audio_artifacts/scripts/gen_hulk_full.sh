#!/bin/bash
# 生成符合 Hulk 分布的完整训练数据 (382,746 条)

set -e

OUTPUT_DIR="/data/sejin/baseline_26/data_hulk_dist_full"
mkdir -p "$OUTPUT_DIR"

echo "=== 生成 Hulk 分布数据 (382,746 条) ==="
echo "输出目录: $OUTPUT_DIR"
echo ""

# 需要激活包含 transformers 的环境
# 如果没有，需要先安装：pip install transformers

python3 /data/sejin/baseline_26/scripts/gen_hulk_dist_data.py \
  --output "$OUTPUT_DIR/train.jsonl" \
  --num-samples 382746 \
  --tokenizer /data/sejin/models/Qwen3-30B-A3B-Base \
  --seed 42 \
  --stats

echo ""
echo "生成 200 条测试集..."
python3 /data/sejin/baseline_26/scripts/gen_hulk_dist_data.py \
  --output "$OUTPUT_DIR/test_200.jsonl" \
  --num-samples 200 \
  --tokenizer /data/sejin/models/Qwen3-30B-A3B-Base \
  --seed 43 \
  --stats

echo ""
echo "✅ 完成！jsonl 源数据:"
ls -lh "$OUTPUT_DIR"/*.jsonl
