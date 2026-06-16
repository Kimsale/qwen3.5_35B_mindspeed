#!/bin/bash
# 在新节点上部署环境

set -e

# 使用Python 3.10
PYTHON=python3.10
PIP=pip3.10

echo "=== Ascend NPU 多节点部署脚本 ==="

# 0. 检查Python版本
if ! command -v $PYTHON &> /dev/null; then
    echo "错误: 未找到 $PYTHON"
    echo "请先运行: bash install_python310.sh"
    exit 1
fi

PYTHON_VERSION=$($PYTHON --version | awk '{print $2}')
echo "Python版本: $PYTHON_VERSION ✓"

# 1. 检查Ascend驱动
echo "检查Ascend驱动..."
if [ ! -d "/usr/local/Ascend/ascend-toolkit" ]; then
    echo "错误: 未找到Ascend驱动，请先安装CANN"
    exit 1
fi

# source /usr/local/Ascend/ascend-toolkit/set_env.sh
# source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null

source /usr/local/Ascend/cann-8.5.0/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null

# 2. 设置安装路径
export PYTHONUSERBASE=/data/yjjiang11/env
export PATH=/data/yjjiang11/env/.local/bin:$PATH
export PYTHONPATH=/data/yjjiang11/env/.local/lib/python3.10/site-packages:$PYTHONPATH

echo "Python包将安装到: ${PYTHONUSERBASE}"
mkdir -p ${PYTHONUSERBASE}

# 3. 先安装torch (与节点0相同版本)
echo "安装torch 2.10.0..."
$PIP install --user torch==2.5.1 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 4. 安装torch_npu (与节点0相同版本)
echo "安装torch_npu 2.10.0rc2..."
$PIP install --user torch-npu==2.5.1.post1 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 5. 安装其他依赖
echo "安装其他依赖..."
$PIP install --user transformers accelerate safetensors datasets -i https://pypi.tuna.tsinghua.edu.cn/simple

# 6. 验证安装
echo "验证torch_npu..."
$PYTHON -c "import torch; import torch_npu; print(f'torch: {torch.__version__}, torch_npu: {torch_npu.__version__}')"

echo "=== 部署完成 ==="
echo "环境已安装到: ${PYTHONUSERBASE}"
echo "现在可以运行: bash run_ep_multinode.sh <node_rank> <nnodes> <master_addr>"
