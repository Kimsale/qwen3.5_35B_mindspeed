#!/bin/bash
# =============================================================================
# Pack format smoke test: 10 steps to verify pack vs pad
# Usage: bash smoke_test_pack.sh [pack|pack-rc|pad]
# =============================================================================

FORMAT=${1:-pack}  # Default to pack format

if [[ "$FORMAT" != "pack" && "$FORMAT" != "pack-rc" && "$FORMAT" != "pad" ]]; then
    echo "Usage: $0 [pack|pack-rc|pad]"
    exit 1
fi

# ---- 固定 CANN 8.5 环境（项目硬性约束）----
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH
source /usr/local/Ascend/cann-8.5.0/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# ---- 激活 baseline 验证过的 venv（transformers fc91372 源码版 / torch_npu 2.7.1）----
VENV=/data/sejin/env/venv_qwen35
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"

# ---- FSDP2 栈运行开关（参考 qwen3_5 官方脚本）----
export NON_MEGATRON=true
export MULTI_STREAM_MEMORY_REUSE=2
export TASK_QUEUE_ENABLE=2
export ASCEND_LAUNCH_BLOCKING=0
export ACLNN_CACHE_LIMIT=100000
export CPU_AFFINITY_CONF=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_CONNECT_TIMEOUT=1800
export TOKENIZERS_PARALLELISM=false
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

# ---- 分布式参数（单机 8 卡）----
NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

# Select config based on format
if [[ "$FORMAT" == "pack" ]]; then
    CONFIG="examples/qwen3_5_audio/perf_tuning/ep8_mbs1_ga4_rc_off_pack.yaml"
    LOGNAME="pack"
elif [[ "$FORMAT" == "pack-rc" ]]; then
    CONFIG="examples/qwen3_5_audio/perf_tuning/ep8_mbs1_ga4_rc_on_pack.yaml"
    LOGNAME="pack_rc"
else
    CONFIG="examples/qwen3_5_audio/perf_tuning/ep8_mbs1_ga4_rc_off_pad1408_nosync.yaml"
    LOGNAME="pad"
fi

logfile=$(date +%Y%m%d)_$(date +%H%M%S)_smoke_${LOGNAME}
mkdir -p logs

echo "=========================================="
echo "  Smoke Test: $FORMAT format (10 steps)"
echo "  Config: $CONFIG"
echo "  Log: logs/train_${logfile}.log"
echo "=========================================="

# Run with max_steps=10
torchrun $DISTRIBUTED_ARGS mindspeed_mm/fsdp/train/trainer.py \
    $CONFIG \
    --training.max_steps 10 \
    --training.logging_steps 1 \
    --training.save_steps 999999 \
    2>&1 | tee logs/train_${logfile}.log

echo ""
echo "=========================================="
echo "  Smoke test completed!"
echo "  Check logs/train_${logfile}.log"
echo "=========================================="
