#!/bin/bash
# 检查两个节点的环境是否一致

echo "=== 环境一致性检查 ==="

echo -e "\n1. CANN版本:"
if [ -f /usr/local/Ascend/ascend-toolkit/latest/version.cfg ]; then
    cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg | grep Version
else
    ls -la /usr/local/Ascend/ | grep cann
fi

echo -e "\n2. Python版本:"
python3.10 --version

echo -e "\n3. torch版本:"
export PYTHONPATH=/data/yjjiang11/env/.local/lib/python3.10/site-packages:$PYTHONPATH
python3.10 -c "import torch; print(f'torch: {torch.__version__}')" 2>/dev/null || echo "torch未安装"

echo -e "\n4. torch_npu版本:"
python3.10 -c "import torch_npu; print(f'torch_npu: {torch_npu.__version__}')" 2>/dev/null || echo "torch_npu未安装"

echo -e "\n5. NPU设备:"
npu-smi info 2>/dev/null | grep "NPU Name" || echo "npu-smi未找到"

echo -e "\n=== 请在两个节点上都运行此脚本，对比输出 ==="
