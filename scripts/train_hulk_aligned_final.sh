#!/bin/bash
# ============================================================
# Qwen3-30B-A3B LoRA 微调 — Hulk 对齐配置（生产版）
# 基于 baseline，严格对齐 Hulk 训练参数（TP1/EP8/CP2/8192seq）
# 数据与权重路径已完成配置
# ============================================================
set -eo pipefail

# ---- 固定 CANN 8.5 环境 ----
source /data/sejin/baseline_26/scripts/env_cann85.sh

cd /data/sejin/third_party/mindspeed-llm-26.0.0

# ---- 路径配置（✅ 已完成）----
# 权重：新转换的 TP1/PP1/EP8 格式
CKPT_LOAD_DIR="/data/sejin/models/Qwen3-Omni-30B-A3B-hulk_tp1_pp1_ep8"
CKPT_SAVE_DIR="/data/sejin/baseline_26/output/ckpt_hulk_aligned"
# 数据：✅ 使用新生成的 hulk 对齐数据集
DATA_PATH="/data/sejin/data_hulk_dist_30k_mcore/hulk_sft_packed"
TOKENIZER_PATH="/data/sejin/models/Qwen3-30B-A3B-Base"
LOG_FILE="${LOG_FILE:-/data/sejin/baseline_26/logs/hulk_aligned_$(date +%Y%m%d_%H%M%S).log}"

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# ==== 并行配置（对齐 Hulk）====
