#!/bin/bash
# ============================================================
# Qwen3-30B-A3B LoRA 基线训练 (MindSpeed-LLM 26.0.0 + CANN 8.5.0)
# 基于官方 examples/mcore/qwen3_moe/tune_qwen3_30b_a3b_4K_lora_ptd.sh
# 适配现有 MG 权重并行配置 TP2/PP1/EP4 (8 卡)
# 约束: 不改模型结构, 仅训练超参/并行/算子
# ============================================================
set -eo pipefail

# ---- 固定 CANN 8.5 环境 ----
source /data/sejin/baseline_26/scripts/env_cann85.sh

cd /data/sejin/third_party/mindspeed-llm-26.0.0

# ---- 路径配置 (全部就绪, 无占位符) ----
CKPT_LOAD_DIR="/data/sejin/checkpoints/qwen3_30b_a3b_mcore_tp2_pp1_ep4"
CKPT_SAVE_DIR="/data/sejin/baseline_26/output/ckpt_baseline"
DATA_PATH="/data/sejin/data/qwen3_sft_mcore/qwen3_sft"
TOKENIZER_PATH="/data/sejin/models/Qwen3-30B-A3B-Base"
LOG_FILE="${LOG_FILE:-/data/sejin/baseline_26/logs/param_mbs${MBS:-2}_$(date +%Y%m%d_%H%M%S).log}"

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# ---- 并行: 匹配现有权重 TP2/PP1/EP4 ----
TP=2
PP=1
EP=4
SEQ_LENGTH=4096
TRAIN_ITERS=${ITERS:-150}          # 基线: 50 步足够采集稳定指标

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

# ---- MoE 参数 (与 config.json 一致, 不改结构) ----
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
    --swap-optimizer \
    --swap-optimizer-times 32 \
    --recompute-activation-function
"

TRAIN_ARGS="
    --micro-batch-size ${MBS:-2} \
    --global-batch-size ${GBS:-16} \
    --lr 1.25e-5 \
    --lr-decay-style cosine \
    --min-lr 1.25e-7 \
    --weight-decay 1e-1 \
    --lr-warmup-fraction 0.01 \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 4096 \
    --seed 42 \
    --bf16 \
    --train-iters ${TRAIN_ITERS} \
    --seq-length ${SEQ_LENGTH} \
    --no-shared-storage
"

MODEL_PARALLEL_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --expert-model-parallel-size ${EP}
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

TUNE_ARGS="
    --finetune \
    --stage sft \
    --is-instruction-dataset \
    --tokenizer-not-use-fast \
    --prompt-type qwen3 \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-fusion \
    --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2
"

# 变长序列(var) vs 固定seq(fixed). fixed 时 pad 到 seq-length 占满显存且每步可比
if [ "${SWEEP_PAD:-var}" = "var" ]; then
    TUNE_ARGS="$TUNE_ARGS --no-pad-to-seq-lengths"
fi

# 额外优化参数注入（优化轮次用）
EXTRA_ARGS="${SWEEP_EXTRA:-}"

mkdir -p "$(dirname "$LOG_FILE")"

# 用 venv_26b 绝对路径的 torchrun，避免 PATH 解析到其他 venv
TORCHRUN=/data/sejin/env/venv_26b/bin/torchrun

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
