#!/bin/bash
# EP Inference: 8 GPUs, EP=8
source /usr/local/Ascend/cann-8.5.0/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null
export PATH=/repo1/yjjiang11/env/.local_qwen3p5/bin:$PATH
export PYTHONPATH=/repo1/yjjiang11/env/.local_qwen3p5/lib/python3.10/site-packages:$PYTHONPATH

export CUDA_DEVICE_MAX_CONNECTIONS=1
export ASCEND_GLOBAL_LOG_LEVEL=3
export TASK_QUEUE_ENABLE=2
export HCCL_CONNECT_TIMEOUT=1200
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"

torchrun --nproc_per_node 8 --master_port 13526 infer_ep.py
