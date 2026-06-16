#!/bin/bash
# Run test with proper CANN 8.5 environment

export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH
source /usr/local/Ascend/cann-8.5.0/set_env.sh 2>/dev/null
source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null

cd /data/sejin/qwenasr_qwen3p5moe_sft
python test_packed_collator.py 2>&1 | grep -v "UserWarning" | grep -v "command not found" | grep -v "setenv_main" | grep -v "remove_env"
