source /usr/local/Ascend/cann-8.5.0/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null
export PATH=/data/yjjiang11/env/.local/bin:$PATH
export PYTHONPATH=/data/yjjiang11/env/.local/lib/python3.10/site-packages:$PYTHONPATH

python parse_lmdb_demo.py