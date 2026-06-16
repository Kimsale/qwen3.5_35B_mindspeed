#!/bin/bash
# ============================================================
# Qwen3-Omni-30B-A3B (xuchen2) LoRA 微调 — 完全对齐 hulk 配置
# 模型: xuchen2 转换的 MoE (48层/128专家/topk8)
# 并行: TP=1/PP=1/EP=8/CP=2 (ulysses)
# LoRA: r=32/alpha=64/dropout=0.1/target=qkv+proj
# 序列: 8192
# 超参: lr=5e-6/clip=5.0/warmup=0.0
# ============================================================
set -eo pipefail

# ---- 固定 CANN 8.5 环境 ----
echo "=== 加载 CANN 8.5 环境 ==="
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH
source /usr/local/Ascend/cann-8.5.0/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_CONNECT_TIMEOUT=1800
export CUDA_DEVICE_MAX_CONNECTIONS=1

cd /data/sejin/third_party/mindspeed-llm-26.0.0

# ---- 路径配置 ----
# 权重: 转换后的 xuchen2 MoE 模型 (TP1/PP1/EP8)
CKPT_LOAD_DIR="/data/sejin/models/Qwen3-Omni-30B-A3B-hulk_tp1_pp1_ep8"
CKPT_SAVE_DIR="/data/sejin/baseline_26/output/ckpt_xuchen2_hulk"
# 数据: 30k hulk 分布数据，已 pack 成 8192 IndexedDataset
# 注意: data_prefix 不含 _packed_*_document 后缀，loader 用 glob 自动发现 input_ids/labels/attention_mask
DATA_PATH="${HULK_DATA_PATH:-/data/sejin/baseline_26/data_hulk_dist_30k/qwen3_sft}"
# Tokenizer: 用 base（与 Captioner 编码一致、含 tokenizer.json、共享 <|im_end|>=151645）
TOKENIZER_PATH="/data/sejin/models/Qwen3-30B-A3B-Base"
LOG_FILE="${LOG_FILE:-/data/sejin/baseline_26/logs/xuchen2_hulk_$(date +%Y%m%d_%H%M%S).log}"

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6002
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# ==== 并行配置（对齐 hulk）====
TP=1
PP=1
EP=8
CP=2
# 派生: DP = 8 / (TP × CP) = 8 / (1×2) = 4 ✓

# ==== 序列长度（对齐 hulk）====
SEQ_LENGTH=8192
TRAIN_ITERS=${ITERS:-60}

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

# ---- MoE 参数 ----
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
# 不使用 swap-optimizer（hulk 纯 GPU 分片）
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

# ---- 训练超参（对齐 hulk）====
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

# ---- 并行参数（对齐 hulk）====
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

# ---- LoRA 参数（对齐 hulk）====
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

# ---- 前置检查 ----
if [ ! -d "$CKPT_LOAD_DIR" ]; then
  echo "❌ 错误: 权重目录不存在: $CKPT_LOAD_DIR"
  echo "   请先执行权重转换脚本:"
  echo "   bash /data/sejin/baseline_26/scripts/convert_xuchen2_qwen3omni_moe.sh"
  exit 1
fi

if [ -z "$(ls ${DATA_PATH}_packed_*_document.bin 2>/dev/null)" ]; then
  echo "❌ 错误: 数据文件不存在: ${DATA_PATH}_packed_*_document.bin"
  echo "   请先执行数据预处理: bash /data/sejin/baseline_26/scripts/preprocess_hulk_data.sh"
  exit 1
fi

mkdir -p "$(dirname "$LOG_FILE")" "$CKPT_SAVE_DIR"

TORCHRUN=/data/sejin/env/venv_26b/bin/torchrun

echo "=== xuchen2 模型 hulk 对齐训练 ==="
echo "并行: TP=$TP PP=$PP EP=$EP CP=$CP (DP=$((8/$TP/$CP)))"
echo "序列: $SEQ_LENGTH"
echo "LoRA: r=32 alpha=64 dropout=0.1 target=qkv+proj"
echo "超参: lr=5e-6 clip=5.0 warmup=0.0"
echo "权重: $CKPT_LOAD_DIR"
echo "数据: $DATA_PATH"
echo "日志: $LOG_FILE"
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
