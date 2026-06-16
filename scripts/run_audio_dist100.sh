#!/bin/bash
# Qwen3.5-35B-A3B + Whisper-large-v3 语音多模态 LoRA SFT
# 团队同分布音频数据(3200样本) 跑满 100 步
# 基于官方已验证脚本 examples/qwen3_5_audio/run_train_venv.sh

export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH

cd /data/sejin/third_party/mindspeed-mm-26.0.0
source /usr/local/Ascend/cann-8.5.0/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# 独立 venv(transformers fc91372 / torch_npu 2.7.1)
VENV=/data/sejin/env/venv_qwen35
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"

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
# 数据侧音频占位符:数据用 <|AUDIO|>,校验与展开均以此为准
# (mm_plugin 默认 AUDIO_PLACEHOLDER=<audio>,与数据不符会报 token 数不匹配)
export AUDIO_PLACEHOLDER="<|AUDIO|>"

mkdir -p logs /data/sejin/baseline_26/logs
TRAIN_LOG="/data/sejin/baseline_26/logs/audio_dist100_$(date +%Y%m%d_%H%M%S).log"
echo "训练日志: $TRAIN_LOG"

# 启动前清理本任务可能的残留进程,避免 HCCL 端口冲突
pkill -9 -f "trainer.py.*dist100" 2>/dev/null || true
sleep 3

"$VENV/bin/torchrun" \
    --nproc_per_node 8 \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr localhost \
    --master_port 6020 \
    mindspeed_mm/fsdp/train/trainer.py \
    examples/qwen3_5_audio/dist100_config.yaml \
    > "$TRAIN_LOG" 2>&1
echo "退出码 $?"
echo "$TRAIN_LOG"
