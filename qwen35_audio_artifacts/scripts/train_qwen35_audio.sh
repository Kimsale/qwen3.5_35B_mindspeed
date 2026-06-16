#!/bin/bash
# =============================================================================
# Qwen3.5-35B-A3B + Whisper-large-v3 语音多模态 LoRA SFT 训练脚本
# 基于 MindSpeed-MM 26.0.0 FSDP2 栈
# 环境: venv_qwen35 (transformers fc91372, torch_npu 2.7.1.post2)
# =============================================================================
set -e

# 必须先设 base PATH，保证 set_env.sh 内的 dirname/uname/cut 等可用
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH

# 切换到 MindSpeed-MM 目录
cd /data/sejin/third_party/mindspeed-mm-26.0.0

# CANN 8.5.0 环境
source /usr/local/Ascend/cann-8.5.0/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# 激活独立 venv_qwen35（经验证的环境）
VENV=/data/sejin/env/venv_qwen35
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"

# 环境变量配置（与之前成功的 60步训练一致）
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

# 配置文件路径
CONFIG_FILE=/data/sejin/baseline_26/scripts/train_qwen35_audio.yaml

# 日志输出
LOG_DIR=/data/sejin/baseline_26/logs
mkdir -p "$LOG_DIR"
TRAIN_LOG="$LOG_DIR/qwen35_audio_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================="
echo "Qwen3.5-35B-A3B + Whisper 音频训练"
echo "=========================================="
echo "配置文件: $CONFIG_FILE"
echo "日志文件: $TRAIN_LOG"
echo "环境: venv_qwen35"
echo "硬件: 单机 8×昇腾910B3"
echo "=========================================="
echo

# 启动训练
"$VENV/bin/torchrun" \
    --nproc_per_node 8 \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr localhost \
    --master_port 6009 \
    mindspeed_mm/fsdp/train/trainer.py \
    "$CONFIG_FILE" \
    2>&1 | tee "$TRAIN_LOG"

EXIT_CODE=$?

echo
echo "=========================================="
echo "训练完成"
echo "退出码: $EXIT_CODE"
echo "日志: $TRAIN_LOG"
if [ $EXIT_CODE -eq 0 ]; then
  echo "✅ 训练成功"
else
  echo "❌ 训练失败，请查看日志"
fi
echo "=========================================="

exit $EXIT_CODE
