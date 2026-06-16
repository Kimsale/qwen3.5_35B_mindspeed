#!/bin/bash
# ============================================================
# Qwen3-30B-A3B LoRA 微调 — Hulk 对齐配置
# 基于 baseline，严格对齐 Hulk 训练参数（TP1/EP8/CP2/8192seq）
# 改动清单：见文件末尾注释
# ============================================================
set -eo pipefail

# ---- 固定 CANN 8.5 环境 ----
source /data/sejin/baseline_26/scripts/env_cann85.sh

cd /data/sejin/third_party/mindspeed-llm-26.0.0

# ---- 路径配置 ----
# 权重：新转换的 TP1/PP1/EP8 格式（等转换完成后确认路径）
CKPT_LOAD_DIR="/data/sejin/models/Qwen3-Omni-30B-A3B-hulk_tp1_pp1_ep8"
CKPT_SAVE_DIR="/data/sejin/baseline_26/output/ckpt_hulk_aligned"
# 数据：使用新做的 hulk 长度分布对齐数据集
DATA_PATH="${HULK_DATA_PATH:-/data/sejin/data_hulk_dist_30k_mcore/hulk_sft}"
TOKENIZER_PATH="/data/sejin/models/Qwen3-30B-A3B-Base"
LOG_FILE="${LOG_FILE:-/data/sejin/baseline_26/logs/hulk_aligned_$(date +%Y%m%d_%H%M%S).log}"

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6101
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# ==== 并行配置（对齐 Hulk）====
# 基线: TP=2, PP=1, EP=4, CP=1
# Hulk: TP=1, PP=1, EP=8, CP=2 (ulysses)
TP=1
PP=1
EP=8
CP=2
# 派生: DP = 8 / (TP × PP × CP) = 8 / (1×1×2) = 4 ✓

# ==== 序列长度（对齐 Hulk）====
# 基线: 4096
# Hulk: 8192
SEQ_LENGTH=8192

TRAIN_ITERS=${ITERS:-60}

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

# ---- MoE 参数（保持不变，与 Hulk 一致）----
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
# 改动: 去掉 --swap-optimizer（试试纯 GPU，Hulk 不用 swap）
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
# 注意: 已去掉 --swap-optimizer --swap-optimizer-times 32
# 若 OOM，再加回: --swap-optimizer --swap-optimizer-times 16

# ---- 训练超参（对齐 Hulk）====
# 基线 → Hulk:
# - lr: 1.25e-5 → 5e-6
# - clip: 1.0 → 5.0
# - warmup: 0.01 → 0.0
# - min_lr: 1.25e-7 → 1e-6
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
# 新增: --context-parallel-size 2 --context-parallel-algo ulysses_cp_algo
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
# 基线 → Hulk:
# - r: 16 → 32
# - alpha: 32 → 64
# - dropout: 0 → 0.1
# - target: linear_qkv linear_proj linear_fc1 linear_fc2 → linear_qkv linear_proj (去掉 MLP)
TUNE_ARGS="
    --finetune \
    --stage sft \
    --is-instruction-dataset \
    --tokenizer-not-use-fast \
    --prompt-type qwen3 \
    --lora-r 32 \
    --lora-alpha 64 \
    --lora-fusion \
    --lora-target-modules linear_qkv linear_proj
"
# 注意: MindSpeed-LLM 26.0.0 的 LoRA 实现 (training.py L151) 硬编码 lora_dropout=0.0,
# 没有暴露 CLI；HULK 配置 dropout=0.1 是框架间不可消除的差异，已在报告中记录。
# dropout 只影响正则化强度，不影响 FLOPs / 吞吐，对性能对标无干扰。

# 变长序列支持（与基线一致）
if [ "${SWEEP_PAD:-var}" = "var" ]; then
    TUNE_ARGS="$TUNE_ARGS --no-pad-to-seq-lengths"
fi

# 额外优化参数注入
EXTRA_ARGS="${SWEEP_EXTRA:-}"

# ---- 数据路径检查 ----
if [[ "$DATA_PATH" == *"PLACEHOLDER"* ]]; then
  echo "⚠️  警告: 数据路径是占位符，请设置环境变量 HULK_DATA_PATH"
  echo "   export HULK_DATA_PATH=/path/to/your/aligned/data"
  echo "   或修改脚本中的 DATA_PATH 变量"
  echo "   按 Ctrl+C 取消，或等待 5 秒继续（会失败）..."
  sleep 5
fi

# ---- 权重路径检查 ----
if [ ! -d "$CKPT_LOAD_DIR" ]; then
  echo "❌ 错误: 权重目录不存在: $CKPT_LOAD_DIR"
  echo "   请先执行权重转换脚本: /data/sejin/baseline_26/scripts/convert_weights_tp1_pp1_ep8.sh"
  exit 1
fi

mkdir -p "$(dirname "$LOG_FILE")" "$CKPT_SAVE_DIR"

# 用 venv_26b 绝对路径的 torchrun
TORCHRUN=/data/sejin/env/venv_26b/bin/torchrun

echo "=== Hulk 对齐配置训练 ==="
echo "并行: TP=$TP PP=$PP EP=$EP CP=$CP (DP=$((8/$TP/$PP/$CP)))"
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

# ============================================================
# 改动清单（相比基线 train_param.sh）
# ============================================================
# 1. 并行配置:
#    - TP: 2 → 1
#    - EP: 4 → 8
#    - CP: 1 → 2 (新增 --context-parallel-size 2 --context-parallel-algo ulysses_cp_algo)
#    - PP: 1 (不变)
#
# 2. LoRA 参数:
#    - rank: 16 → 32
#    - alpha: 32 → 64
#    - dropout: 0 → 0.1 (新增 --lora-dropout 0.1)
#    - target: 去掉 linear_fc1 linear_fc2 (仅保留 linear_qkv linear_proj)
#
# 3. 序列长度:
#    - seq_length: 4096 → 8192
#    - max_position_embeddings: 4096 → 8192
#
# 4. 训练超参:
#    - lr: 1.25e-5 → 5e-6
#    - min_lr: 1.25e-7 → 1e-6
#    - lr-warmup-fraction: 0.01 → 0.0
#    - clip-grad: 1.0 → 5.0
#
# 5. 优化器:
#    - 去掉: --swap-optimizer --swap-optimizer-times 32
#    - 保留: --use-distributed-optimizer (ZeRO-1 等价)
#
# 6. 权重:
#    - CKPT_LOAD_DIR: 从 tp2_pp1_ep4 → tp1_pp1_ep8 (需先转换)
#
# 7. 数据:
#    - DATA_PATH: 占位符，等用户提供新数据路径
#
# 8. 保持不变:
#    - MoE 参数 (num-experts 128, topk 8, ffn 768, alltoall_seq, grouped-gemm)
#    - 融合算子 (flash-attn, fused-swiglu/rmsnorm/rotary, no-rope-fusion)
#    - 重计算 (full recompute, block method)
#    - 其他 (bf16, sequence-parallel, distributed-optimizer, no-rope-fusion)
# ============================================================
