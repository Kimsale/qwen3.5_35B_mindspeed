#!/bin/bash
# AutoTuning 搜索: 用 MindSpeed 自动并行配置搜索器, 在与生产训练一致的模型规模下
# 自动遍历 TP/PP/EP/MBS/recompute 等组合, 输出 top-k 推荐.
# 注意: --auto-settings 与 --load/--save 互斥; 走 pretrain_gpt.py (标准入口, AutoSettings 只 hook pretrain)
# 搜出的推荐再回到 LoRA 训练做对比验证. AutoTuning 不感知 LoRA 约束(如 moe-tp-extend-ep
# 与 LoRA 不兼容), 我们后处理过滤.
set -eo pipefail

source /data/sejin/baseline_26/scripts/env_cann85.sh

cd /data/sejin/third_party/mindspeed-llm-26.0.0

WORK_DIR=/data/sejin/baseline_26/autotune_output
mkdir -p "$WORK_DIR"
LOG_FILE="${LOG_FILE:-/data/sejin/baseline_26/logs/autotune_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p "$(dirname "$LOG_FILE")"

NPUS_PER_NODE=8
NNODES=1
NODE_RANK=0
MASTER_ADDR=127.0.0.1
MASTER_PORT=29500
WORLD_SIZE=$((NPUS_PER_NODE * NNODES))

# AutoTuning 内部用 env:// rendezvous，需要从环境变量读分布式信息
export MASTER_ADDR
export MASTER_PORT
export WORLD_SIZE
export RANK=0
export LOCAL_RANK=0

# 与基线一致的模型超参 (Qwen3-30B-A3B)
GPT_ARGS="
    --use-mcore-models \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --kv-channels 128 \
    --qk-layernorm \
    --norm-topk-prob \
    --tokenizer-name-or-path /data/sejin/models/Qwen3-30B-A3B-Base \
    --max-position-embeddings 4096 \
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
    --num-query-groups 4 \
    --transformer-impl local
"

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

TRAIN_ARGS="
    --micro-batch-size 1 \
    --global-batch-size 8 \
    --lr 1.25e-5 \
    --lr-decay-style cosine \
    --min-lr 1.25e-7 \
    --weight-decay 1e-1 \
    --lr-warmup-fraction 0.01 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 4096 \
    --seed 42 \
    --bf16 \
    --train-iters 10 \
    --seq-length 4096 \
    --no-shared-storage
"

OPTIMIZE_ARGS="
    --use-flash-attn \
    --use-fused-rotary-pos-emb \
    --use-rotary-position-embeddings \
    --use-fused-swiglu \
    --use-fused-rmsnorm \
    --no-masked-softmax-fusion \
    --use-distributed-optimizer \
    --no-rope-fusion
"

DATA_ARGS="
    --data-path /data/sejin/data/qwen3_sft_mcore/qwen3_sft_packed_input_ids_document \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 999999 \
    --eval-interval 999999 \
    --eval-iters 0 \
    --tokenizer-not-use-fast
"

AUTOTUNE_ARGS="
    --auto-settings \
    --auto-settings-type mixed \
    --auto-settings-ranks ${WORLD_SIZE} \
    --auto-settings-work-dir ${WORK_DIR} \
    --auto-settings-log-level info \
    --target-nnodes ${NNODES}
"
# 注: nnodes/nproc-per-node/node-rank/master-addr/master-port 不在这里传, 让 AutoTuning
# 内部按 ranks 自己用 torchrun 子进程驱动 profiling. 主进程是 python 单进程入口.

# AutoTuning 入口必须用 torchrun 启动 (rank0 接管 AutoSettings, 其他 rank 早退);
# 单 python 进程会卡在 init_process_group(world_size=8) 等不到其他 rank.
TORCHRUN=/data/sejin/env/venv_26b/bin/torchrun

$TORCHRUN \
    --nproc_per_node=${NPUS_PER_NODE} \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    pretrain_gpt.py \
    $GPT_ARGS \
    $MOE_ARGS \
    $TRAIN_ARGS \
    $OPTIMIZE_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $AUTOTUNE_ARGS \
    --distributed-backend nccl \
    2>&1 | tee "$LOG_FILE"
