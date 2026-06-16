#!/bin/bash
# ============================================================
# Qwen3-30B-A3B LoRA 微调 — Hulk 对齐配置（生产就绪版）
# 数据路径、权重路径已修复，可直接运行
# ============================================================
set -eo pipefail

# ---- 固定 CANN 8.5 环境 ----
source /data/sejin/baseline_26/scripts/env_cann85.sh

cd /data/sejin/third_party/mindspeed-llm-26.0.0

# ---- 路径配置（✅ 已修复）----
# 权重：TP1/PP1/EP8 格式（转换输出目录）
CKPT_LOAD_DIR="/data/sejin/models/Qwen3-Omni-30B-A3B-hulk_tp1_pp1_ep8"
CKPT_SAVE_DIR="/data/sejin/baseline_26/output/ckpt_hulk_aligned"
# 数据：✅ 使用新生成的 hulk 对齐数据集
DATA_PATH="/data/sejin/data_hulk_dist_30k_mcore/hulk_sft_packed"
TOKENIZER_PATH="/data/sejin/models/Qwen3-30B-A3B-Base"
LOG_FILE="${LOG_FILE:-/data/sejin/baseline_26/logs/hulk_aligned_$(date +%Y%m%d_%H%M%S).log}"

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# ==== 并行配置（对齐 Hulk）====
TP=1
PP=1
EP=8
CP=2
# 派生: DP = 8 / (TP × PP × CP) = 4 ✓

# ==== 序列长度（对齐 Hulk）====
SEQ_LENGTH=8192

TRAIN_ITERS=${ITERS:-60}

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

# ---- MoE 参数（保持不变）----
MOE_ARGS="
    --num-experts 128 \
    --moe-router-topk 8 \
    --moe-ffn-hidden-size 768 \
    --moe-grouped-gemm \
    --moe-permutation-async-comm \
    --moe-token-dispatcher-type alltoall_seq \
    --moe-router-load-balancing-type aux_loss \
    --moe-layer-freq -1 \
    --first-k-dense-replace -1 \
    --moe-aux-loss-coeff 0.001
"

# ---- 优化器/算子配置 ----
OPTIMIZE_ARGS="
    --use-flash-attn \
    --use-fused-rotary-pos-emb \
    --sequence-parallel \
    --use-rotary-position-embeddings \
    --use-fused-swiglu \
    --use-fused-rmsnorm \
    --no-masked-softmax-fusion \
    --use-distributed-optimizer \
    --no-rope-fusion \
    --recompute-granularity full \
    --recompute-method block \
    --recompute-num-layers 1
"

# ---- 训练超参（对齐 Hulk）====
TRAIN_ARGS="
    --micro-batch-size ${MBS:-1} \
    --global-batch-size ${GBS:-16} \
    --lr 5e-6 \
    --lr-decay-style cosine \
    --min-lr 1e-6 \
    --weight-decay 1e-1 \
    --lr-warmup-fraction 0.0 \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --clip-grad 5.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 4096 \
    --seed 42 \
    --bf16 \
    --train-iters ${TRAIN_ITERS} \
    --seq-length ${SEQ_LENGTH} \
    --no-shared-storage
"

# ---- 并行参数（对齐 Hulk）====
MODEL_PARALLEL_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --expert-model-parallel-size ${EP} \
    --context-parallel-size ${CP} \
    --context-parallel-algo ulysses_cp_algo
"

GPT_ARGS="
    --use-mcore-models \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --kv-channels 128 \
    --qk-layernorm \
    --norm-topk-prob \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --max-position-embeddings ${SEQ_LENGTH} \
    --num-layers 48 \
    --hidden-size 2048 \
    --ffn-hidden-size 6144 \
    --num-attention-heads 32 \
    --tokenizer-type PretrainedFromHF \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 152064 \
    --rotary-base 1000000 \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --norm-epsilon 1e-6 \
    --swiglu \
    --attention-softmax-in-fp32 \
    --no-gradient-accumulation-fusion \
    --group-query-attention \
    --num-query-groups 4
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 999999 \
    --eval-interval ${TRAIN_ITERS} \
    --eval-iters 0 \
    --no-load-optim \
    --no-load-rng
"

# ---- LoRA 参数（对齐 Hulk）====
TUNE_ARGS="
    --finetune \
    --stage sft \
    --is-instruction-dataset \
    --tokenizer-not-use-fast \
    --prompt-type qwen3 \
    --lora-r 32 \
    --lora-alpha 64 \
    --lora-dropout 0.1 \
    --lora-fusion \
    --lora-target-modules linear_qkv linear_proj
"

# 变长序列支持
if [ "${SWEEP_PAD:-var}" = "var" ]; then
    TUNE_ARGS="$TUNE_ARGS --no-pad-to-seq-lengths"
fi

EXTRA_ARGS="${SWEEP_EXTRA:-}"

# ---- 启动前检查 ----
echo "=== Hulk 对齐配置训练（生产就绪版）==="
echo "并行: TP=$TP PP=$PP EP=$EP CP=$CP (DP=$((8/$TP/$PP/$CP)))"
echo "序列: $SEQ_LENGTH"
echo "LoRA: r=32 alpha=64 dropout=0.1 target=qkv+proj"
echo "超参: lr=5e-6 clip=5.0 warmup=0.0"
echo "权重: $CKPT_LOAD_DIR"
echo "数据: $DATA_PATH"
echo "日志: $LOG_FILE"
echo

# 检查数据集
if [ ! -f "${DATA_PATH}_input_ids_document.idx" ]; then
  echo "❌ 错误: 数据集不存在: ${DATA_PATH}_input_ids_document.idx"
  echo "   期望位置: /data/sejin/data_hulk_dist_30k_mcore/hulk_sft_packed_*.{bin,idx}"
  exit 1
fi
echo "✅ 数据集检查通过"

# 检查权重
if [ ! -d "$CKPT_LOAD_DIR" ]; then
  echo "❌ 错误: 权重目录不存在: $CKPT_LOAD_DIR"
  echo "   权重转换可能未完成，请检查进程: ps -p 1861720"
  echo "   或查看转换输出: ls -lh /data/sejin/models/Qwen3-Omni-30B-A3B-hulk_tp1_pp1_ep8/"
  exit 1
fi

# 检查权重文件是否完整（至少要有 iter_0000000 目录）
if [ ! -d "$CKPT_LOAD_DIR/iter_0000000" ]; then
  echo "⚠️  警告: 权重目录存在但可能未完成转换"
  echo "   当前文件: $(ls -1 $CKPT_LOAD_DIR/ 2>/dev/null | head -5)"
  echo "   按 Ctrl+C 取消，或等待 5 秒继续（可能失败）..."
  sleep 5
else
  echo "✅ 权重检查通过"
fi

mkdir -p "$(dirname "$LOG_FILE")" "$CKPT_SAVE_DIR"

# 用 venv_26b 绝对路径的 torchrun
TORCHRUN=/data/sejin/env/venv_26b/bin/torchrun

echo
echo "开始训练 $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo

$TORCHRUN $DISTRIBUTED_ARGS posttrain_gpt.py \
    $TUNE_ARGS \
    $GPT_ARGS \
    $DATA_ARGS \
    $MOE_ARGS \
    $OUTPUT_ARGS \
    $OPTIMIZE_ARGS \
    $TRAIN_ARGS \
    $MODEL_PARALLEL_ARGS \
    $EXTRA_ARGS \
    --load ${CKPT_LOAD_DIR} \
    --distributed-backend nccl \
    --transformer-impl local \
    2>&1 | tee "$LOG_FILE"

echo
echo "训练完成 $(date '+%Y-%m-%d %H:%M:%S')"
