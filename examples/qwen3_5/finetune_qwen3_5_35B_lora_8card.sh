#!/bin/bash
# =============================================================================
# Qwen3.5-35B-A3B  LoRA 微调启动脚本  (单机 8×910B3)
# 框架: MindSpeed-MM 26.0.0  FSDP2 路线
# 配置: examples/qwen3_5/qwen3_5_35B_lora_8card_optimal.yaml
#
# 用法: 在 mindspeed-mm-26.0.0 仓库根目录执行:
#   bash examples/qwen3_5/finetune_qwen3_5_35B_lora_8card.sh
# =============================================================================
set -e

# -------- CANN8.5 环境 (遵循项目强制规则, 禁用系统默认 CANN8.1) --------
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH
source /usr/local/Ascend/cann-8.5.0/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# -------- MindSpeed-MM FSDP 路线环境变量 --------
export NON_MEGATRON=true                              # 关键: 走 FSDP2 而非 Megatron
export MULTI_STREAM_MEMORY_REUSE=2                    # 多流显存复用
export TASK_QUEUE_ENABLE=2                            # 任务下发队列(降低 host 瓶颈, 提 AI Core 占用)
export ASCEND_LAUNCH_BLOCKING=0                       # 异步下发(性能模式)
export ACLNN_CACHE_LIMIT=100000                       # 算子缓存上限
export CPU_AFFINITY_CONF=1                            # CPU 亲和性绑定
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True  # 可扩展显存段, 降碎片
export HCCL_CONNECT_TIMEOUT=1800                      # HCCL 连接超时
export TOKENIZERS_PARALLELISM=false

# -------- 单机分布式参数 (官方默认 NNODES=2, 此处改为单机) --------
NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1                                              # 单机
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))              # = 8

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

CONFIG=examples/qwen3_5/qwen3_5_35B_lora_8card_optimal.yaml

logfile=$(date +%Y%m%d)_$(date +%H%M%S)
mkdir -p logs

echo "=========================================="
echo "🚀 Qwen3.5-35B-A3B LoRA 微调 (单机8卡)"
echo "   配置: ${CONFIG}"
echo "   日志: logs/train_${logfile}.log"
echo "=========================================="

torchrun $DISTRIBUTED_ARGS mindspeed_mm/fsdp/train/trainer.py \
    ${CONFIG} \
    2>&1 | tee logs/train_${logfile}.log
