#!/bin/bash
# ============================================================
# CANN 8.5.0 固定环境脚本 (按 CLAUDE.md 强制规则)
# 禁用系统默认 CANN 8.1，全程锁定 8.5.0
# ============================================================
set -eo pipefail

# 1) 修复 PATH，避免 zsh/残缺 PATH 导致 set_env.sh 内 dirname/grep 失败
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH

# 2) 锁定 CANN 8.5.0（绝不 source ascend-toolkit/ 下的 8.1）
source /usr/local/Ascend/cann-8.5.0/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# 3) 屏蔽 8.1 残留环境变量校验
export ASCEND_TOOLKIT_HOME="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-8.5.0}"

# 4) torch_npu 必须显式 import（本机已验证规则）
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

# 5) 训练通用环境
export HCCL_CONNECT_TIMEOUT=1800
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export NPU_ASD_ENABLE=0

# 6) venv：基于 venv_swift 克隆的 venv_26b（torch2.7.1+torch_npu2.7.1.post2+apex+transformers4.57.1）
VENV=/data/sejin/env/venv_26b
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"

# 7) 26.0.0 源码注入式运行（PYTHONPATH 注入正确版本，不 pip install）
TP_ROOT=/data/sejin/third_party
export PYTHONPATH="${TP_ROOT}/mindspeed-llm-26.0.0:${TP_ROOT}/mindspeed-core-26.0.0:${TP_ROOT}/Megatron-LM-core_v0.12.1:${PYTHONPATH:-}"

echo "[env] ASCEND_HOME_PATH=$ASCEND_HOME_PATH"
echo "[env] CANN: $(grep ^version /usr/local/Ascend/cann-8.5.0/aarch64-linux/ascend_toolkit_install.info)"
echo "[env] VENV=$VIRTUAL_ENV"
echo "[env] PYTHONPATH=$PYTHONPATH"
