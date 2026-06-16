#!/bin/bash
# Qwen3-30B-A3B LoRA 微调脚本 - 基于 MindSpeed-LLM 26.0.0
# 使用 Megatron-Core checkpoint 和 packed 数据格式

# ========== 环境变量设置（按照 CLAUDE.md 要求固定使用 CANN 8.5） ==========
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH
source /usr/local/Ascend/cann-8.5.0/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# 添加 Megatron-LM 到 PYTHONPATH
export PYTHONPATH=/data/sejin/third_party/Megatron-LM-core_v0.12.1:$PYTHONPATH

# 运行时环境变量
export HCCL_CONNECT_TIMEOUT=1800
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=2
export ASCEND_LAUNCH_BLOCKING=0
export ACLNN_CACHE_LIMIT=100000
export CPU_AFFINITY_CONF=1
export TOKENIZERS_PARALLELISM=false

# ========== 分布式训练配置 ==========
NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# ========== 路径配置 ==========
CKPT_LOAD_DIR="/data/sejin/checkpoints/qwen3_30b_a3b_mcore_tp2_pp1_ep4"
CKPT_SAVE_DIR="/data/sejin/output/qwen3_30b_a3b_lora_mindspeed_mm"
DATA_PATH="/data/sejin/data_hulk_dist_30k_mcore/hulk_sft"
TOKENIZER_PATH="/data/sejin/models/Qwen3-30B-A3B-Base"

# ========== 并行策略配置 ==========
# checkpoint 是 TP2-PP1-EP4，训练时保持一致或调整
TP=2
PP=1
EP=4
SEQ_LENGTH=4096

# ========== 训练超参数 ==========
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=8
TRAIN_ITERS=500
LR=1.0e-5
MIN_LR=1.0e-7

# ========== LoRA 超参数 ==========
LORA_R=16
LORA_ALPHA=32

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
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

OPTIMIZE_ARGS="
    --use-flash-attn \
    --use-fused-rotary-pos-emb \
    --sequence-parallel \
    --use-rotary-position-embeddings \
    --use-fused-swiglu \
    --use-fused-rmsnorm \
    --no-masked-softmax-fusion \
    --use-distributed-optimizer
"

TRAIN_ARGS="
    --micro-batch-size ${MICRO_BATCH_SIZE} \
    --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --lr ${LR} \
    --lr-decay-style cosine \
    --min-lr ${MIN_LR} \
    --weight-decay 0.01 \
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
    --load ${CKPT_LOAD_DIR} \
    --save ${CKPT_SAVE_DIR} \
    --log-interval 1 \
    --save-interval 100 \
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
    --lora-r ${LORA_R} \
    --lora-alpha ${LORA_ALPHA} \
    --lora-fusion \
    --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2
"

# ========== 创建日志目录 ==========
mkdir -p /data/sejin/logs
mkdir -p ${CKPT_SAVE_DIR}

# ========== 启动训练 ==========
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
LOG_FILE="/data/sejin/logs/train_qwen3_30b_a3b_lora_${TIMESTAMP}.log"

cd /data/sejin/third_party/mindspeed-llm-26.0.0

echo "========================================" | tee ${LOG_FILE}
echo "开始 Qwen3-30B-A3B LoRA 微调训练" | tee -a ${LOG_FILE}
echo "训练配置：" | tee -a ${LOG_FILE}
echo "  - Checkpoint: ${CKPT_LOAD_DIR}" | tee -a ${LOG_FILE}
echo "  - Data: ${DATA_PATH}" | tee -a ${LOG_FILE}
echo "  - Tokenizer: ${TOKENIZER_PATH}" | tee -a ${LOG_FILE}
echo "  - 并行策略: TP=${TP}, PP=${PP}, EP=${EP}" | tee -a ${LOG_FILE}
echo "  - Batch Size: Micro=${MICRO_BATCH_SIZE}, Global=${GLOBAL_BATCH_SIZE}" | tee -a ${LOG_FILE}
echo "  - LoRA: R=${LORA_R}, Alpha=${LORA_ALPHA}" | tee -a ${LOG_FILE}
echo "  - 训练迭代: ${TRAIN_ITERS}" | tee -a ${LOG_FILE}
echo "  - 序列长度: ${SEQ_LENGTH}" | tee -a ${LOG_FILE}
echo "  - 学习率: ${LR}" | tee -a ${LOG_FILE}
echo "========================================" | tee -a ${LOG_FILE}

torchrun $DISTRIBUTED_ARGS posttrain_gpt.py \
    $TUNE_ARGS \
    $GPT_ARGS \
    $DATA_ARGS \
    $MOE_ARGS \
    $OUTPUT_ARGS \
    $OPTIMIZE_ARGS \
    $TRAIN_ARGS \
    $MODEL_PARALLEL_ARGS \
    --distributed-backend nccl \
    --transformer-impl local \
    2>&1 | tee -a ${LOG_FILE}

# ========== 性能指标提取 ==========
echo "" | tee -a ${LOG_FILE}
echo "========================================" | tee -a ${LOG_FILE}
echo "训练完成，提取性能指标..." | tee -a ${LOG_FILE}
echo "========================================" | tee -a ${LOG_FILE}

# 提取单步耗时（排除前几步和最后几步）
STEP_TIME=$(grep "elapsed time per iteration" ${LOG_FILE} | awk -F ':' '{print$5}' | awk -F '|' '{print$1}' | head -n 150 | tail -n 100 | awk '{sum+=$1} END {if (NR != 0) printf("%.1f",sum/NR)}')

# 提取全局batch size
GBS=$(grep "consumed samples:" ${LOG_FILE} | tail -n 1 | awk -F '|' '{split($1, a, "iteration"); split(a[2], b, "/"); iter=b[1]+0; split($2, c, ":"); samp=c[2]+0; if(iter!=0) printf("%.2f", samp/iter); else print "N/A"}')

# 计算吞吐
SAMPLES_PER_SECOND=$(awk 'BEGIN{if("'${STEP_TIME}'" != "" && "'${STEP_TIME}'" != "0") printf "%.3f\n", '${GBS}'*1000/'${STEP_TIME}'; else print "N/A"}')

# 提取 tokens per sample
LOG_TOKENS_PER_SECOND=$(grep "tokens per sample" ${LOG_FILE})
if [ "$LOG_TOKENS_PER_SECOND" ]; then
    AVERAGE_TOKENS=$(grep "tokens per sample" ${LOG_FILE} | awk -F 'tokens per sample:' '{print$2}' | awk -F '|' '{print$1}' | head -n 150 | tail -n 100 | awk '{sum+=$1} END {if (NR != 0) printf("%.1f",sum/NR)}')
    TOKENS_PER_SECOND=$(awk 'BEGIN{if("'${SAMPLES_PER_SECOND}'" != "N/A" && "'${AVERAGE_TOKENS}'" != "") printf "%.3f\n", '${SAMPLES_PER_SECOND}'*'${AVERAGE_TOKENS}'; else print "N/A"}')
else
    AVERAGE_TOKENS="N/A"
    TOKENS_PER_SECOND="N/A"
fi

echo "性能指标摘要：" | tee -a ${LOG_FILE}
echo "  - 单步耗时 (Step Time): ${STEP_TIME} ms" | tee -a ${LOG_FILE}
echo "  - 样本吞吐 (Samples/s): ${SAMPLES_PER_SECOND}" | tee -a ${LOG_FILE}
echo "  - 平均 Tokens/Sample: ${AVERAGE_TOKENS}" | tee -a ${LOG_FILE}
echo "  - Token 吞吐 (Tokens/s): ${TOKENS_PER_SECOND}" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}
echo "日志文件: ${LOG_FILE}" | tee -a ${LOG_FILE}
echo "Checkpoint 保存路径: ${CKPT_SAVE_DIR}" | tee -a ${LOG_FILE}
echo "========================================" | tee -a ${LOG_FILE}
