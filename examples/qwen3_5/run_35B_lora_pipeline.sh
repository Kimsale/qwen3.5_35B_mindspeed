#!/bin/bash
# =============================================================================
# Qwen3.5-35B-A3B  LoRA 60-step 全自动流水线
# 阶段: 等待下载完成 → HF→DCP 权重转换 → 启动训练 → 采集性能 → 写报告
# 在 mindspeed-mm-26.0.0 根目录执行:
#   nohup bash examples/qwen3_5/run_35B_lora_pipeline.sh > logs/pipeline.log 2>&1 &
# =============================================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

# -------- 路径 --------
HF_DIR="/mnt/shared_data_196/sejin/models/Qwen3.5-35B-A3B"
DCP_DIR="/mnt/shared_data_196/sejin/models/Qwen3.5-35B-A3B-dcp"
CONFIG="examples/qwen3_5/qwen3_5_35B_lora_8card_optimal.yaml"
LOG_DIR="$REPO_DIR/logs"
TRAIN_LOG="$LOG_DIR/qwen35_lora_60step_$(date +%Y%m%d_%H%M%S).log"
REPORT_DIR="$LOG_DIR"
PIPELINE_LOG="$LOG_DIR/pipeline_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$PIPELINE_LOG") 2>&1

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# -------- CANN8.5 --------
log "=== 激活 CANN8.5 ==="
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH
source /usr/local/Ascend/cann-8.5.0/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# ====================================================================
# 阶段1: 等待模型下载完成
# ====================================================================
log "=== 阶段1: 等待模型分片下载完成 ==="
wait_for_download() {
    local max_wait=10800  # 最多等 3 小时
    local waited=0
    while true; do
        local missing=0
        for i in $(seq -w 1 14); do
            local f="$HF_DIR/model.safetensors-000${i}-of-00014.safetensors"
            if [ ! -f "$f" ]; then
                missing=$((missing+1))
            else
                local sz=$(stat -c%s "$f")
                # 分片14(视觉塔)约2GB, 其余约5GB
                local min_sz=1500000000
                if [ "$i" = "14" ]; then min_sz=1500000000; else min_sz=4500000000; fi
                if [ "$sz" -lt "$min_sz" ]; then missing=$((missing+1)); fi
            fi
        done
        if [ "$missing" -eq 0 ]; then
            log "✅ 全部 14 个分片下载完成"
            return 0
        fi
        log "⏳ 还有 $missing 个分片未完成, 已等 ${waited}s..."
        sleep 60
        waited=$((waited+60))
        if [ "$waited" -ge "$max_wait" ]; then
            log "❌ 等待超时(${max_wait}s), 中止"
            exit 1
        fi
    done
}
wait_for_download

# ====================================================================
# 阶段2: HF → DCP 权重转换
# ====================================================================
log "=== 阶段2: HF → DCP 权重转换 ==="
if [ -d "$DCP_DIR/release" ]; then
    log "✅ DCP 已存在 ($DCP_DIR/release), 跳过转换"
else
    log "开始转换... (约需 10-20 分钟)"
    mkdir -p "$DCP_DIR"
    mm-convert Qwen35Converter hf_to_dcp \
        --hf_dir "$HF_DIR" \
        --dcp_dir "$DCP_DIR" \
        2>&1 | tee "$LOG_DIR/dcp_convert.log"
    if [ -d "$DCP_DIR/release" ]; then
        log "✅ DCP 转换完成: $DCP_DIR"
    else
        log "❌ DCP 转换失败, 查看: $LOG_DIR/dcp_convert.log"
        exit 1
    fi
fi

# ====================================================================
# 阶段3: 训练配置验证 & 启动
# ====================================================================
log "=== 阶段3: 启动 60-step LoRA 训练 ==="

# NPU 训练环境变量
export NON_MEGATRON=true
export MULTI_STREAM_MEMORY_REUSE=2
export TASK_QUEUE_ENABLE=2
export ASCEND_LAUNCH_BLOCKING=0
export ACLNN_CACHE_LIMIT=100000
export CPU_AFFINITY_CONF=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_CONNECT_TIMEOUT=1800
export TOKENIZERS_PARALLELISM=false

NPUS_PER_NODE=8; NNODES=1; NODE_RANK=0
MASTER_ADDR=localhost; MASTER_PORT=6001  # 换端口避免和下载进程冲突

log "训练日志: $TRAIN_LOG"

# 后台跑 npu-smi 监控 (每 10 秒采一次)
npu-smi info -l > /dev/null 2>&1 && {
    NPU_MON_LOG="$LOG_DIR/npu_monitor_35B.log"
    log "启动 NPU 监控 -> $NPU_MON_LOG"
    (while true; do
        echo "=== $(date '+%Y-%m-%d %H:%M:%S') ===" >> "$NPU_MON_LOG"
        npu-smi info >> "$NPU_MON_LOG" 2>&1
        sleep 10
    done) &
    NPU_MON_PID=$!
    log "NPU 监控 PID: $NPU_MON_PID"
}

T_START=$(date +%s)

torchrun \
    --nproc_per_node "$NPUS_PER_NODE" \
    --nnodes "$NNODES" \
    --node_rank "$NODE_RANK" \
    --master_addr "$MASTER_ADDR" \
    --master_port "$MASTER_PORT" \
    mindspeed_mm/fsdp/train/trainer.py \
    "$CONFIG" \
    2>&1 | tee "$TRAIN_LOG"

T_END=$(date +%s)
ELAPSED=$((T_END-T_START))

# 停止 NPU 监控
[ -n "$NPU_MON_PID" ] && kill "$NPU_MON_PID" 2>/dev/null || true

log "=== 训练结束, 总耗时: ${ELAPSED}s ==="

# ====================================================================
# 阶段4: 采集性能指标 → 写报告
# ====================================================================
log "=== 阶段4: 解析性能指标 ==="

REPORT_FILE="$REPORT_DIR/qwen35_35B_lora_60step_perf_$(date +%Y%m%d_%H%M%S).md"

python3 - <<'PYEOF' "$TRAIN_LOG" "$NPU_MON_LOG" "$REPORT_FILE" "$ELAPSED"
import sys, re, os, json
from statistics import mean, stdev

train_log  = sys.argv[1]
npu_log    = sys.argv[2] if len(sys.argv)>2 else ""
report_out = sys.argv[3] if len(sys.argv)>3 else "/tmp/report.md"
total_elapsed = int(sys.argv[4]) if len(sys.argv)>4 else 0

# ---------- 解析训练日志 ----------
step_times, losses, tps_list, wps_list = [], [], [], []
trainable_params = None

if os.path.exists(train_log):
    with open(train_log) as f:
        for line in f:
            # step 耗时: elapsed time per iteration
            m = re.search(r'elapsed time per iteration.*?([0-9]+\.[0-9]+)\s*ms', line, re.I)
            if m: step_times.append(float(m.group(1)))
            # loss
            m = re.search(r'\bloss[: ]+([0-9]+\.[0-9]+)', line, re.I)
            if m: losses.append(float(m.group(1)))
            # tps
            m = re.search(r'tps[: ]+([0-9]+\.?[0-9]*)', line, re.I)
            if m: tps_list.append(float(m.group(1)))
            # wps / tokens_per_sec
            m = re.search(r'(?:wps|tokens[_/]per[_/]sec)[: ]+([0-9]+)', line, re.I)
            if m: wps_list.append(int(m.group(1)))
            # trainable params
            m = re.search(r'trainable.*?([0-9,]+).*?all.*?([0-9,]+)', line, re.I)
            if m:
                t = int(m.group(1).replace(',',''))
                a = int(m.group(2).replace(',',''))
                trainable_params = (t, a, t/a*100)

# ---------- 解析 NPU 监控 ----------
hbm_vals, aicore_vals, power_vals = [], [], []
if npu_log and os.path.exists(npu_log):
    with open(npu_log) as f:
        for line in f:
            m = re.search(r'([0-9]+)\s*/\s*65536', line)
            if m: hbm_vals.append(int(m.group(1)))
            m = re.search(r'AICore\(%\)\s+([0-9]+)', line)
            if m: aicore_vals.append(int(m.group(1)))
            m = re.search(r'Power\(W\)\s+([0-9.]+)', line)
            if m: power_vals.append(float(m.group(1)))

# ---------- 计算汇总指标 ----------
# 跳过前3步 warmup
step_times_stable = step_times[3:] if len(step_times)>3 else step_times
mean_step = mean(step_times_stable) if step_times_stable else 0
std_step  = stdev(step_times_stable) if len(step_times_stable)>1 else 0
mean_tps  = mean(tps_list[3:]) if len(tps_list)>3 else (mean(tps_list) if tps_list else 0)
mean_wps  = mean(wps_list[3:]) if len(wps_list)>3 else (mean(wps_list) if wps_list else 0)
hbm_peak  = max(hbm_vals) if hbm_vals else 0
hbm_mean  = mean(hbm_vals) if hbm_vals else 0
aic_peak  = max(aicore_vals) if aicore_vals else 0
aic_mean  = mean(aicore_vals) if aicore_vals else 0
pwr_mean  = mean(power_vals) if power_vals else 0
loss_start = losses[0] if losses else 0
loss_end   = losses[-1] if losses else 0

# ---------- 写 Markdown 报告 ----------
with open(report_out, 'w') as f:
    f.write("# Qwen3.5-35B-A3B LoRA 60-step 性能报告\n\n")
    f.write(f"- **生成时间**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"- **框架**: MindSpeed-MM 26.0.0 FSDP2\n")
    f.write(f"- **硬件**: 单机 8×910B3 (64GB HBM)\n")
    f.write(f"- **配置**: `examples/qwen3_5/qwen3_5_35B_lora_8card_optimal.yaml`\n\n")

    f.write("## 模型 & 微调配置\n\n")
    f.write("| 项 | 值 |\n|---|---|\n")
    f.write("| 模型 | Qwen3.5-35B-A3B (VL+MoE, 40层, 256专家/层, 激活8) |\n")
    f.write("| 微调方式 | LoRA r=16, α=32 |\n")
    f.write("| LoRA 目标 | self_attn.{q,k,v,o}_proj + linear_attn.{in_proj_qkv,out_proj} |\n")
    if trainable_params:
        f.write(f"| 可训练参数 | {trainable_params[0]:,} ({trainable_params[2]:.2f}%) |\n")
    f.write("| 序列长度 | 4096 |\n")
    f.write("| micro batch | 1 |\n")
    f.write("| EP | 8 (mc2 dispatcher) |\n")
    f.write("| 通信重叠 | 前/反向 prefetch=1 |\n")
    f.write("| 重计算 | 选择性 (仅 experts MLP) |\n\n")

    f.write("## 吞吐 & 延迟\n\n")
    f.write("| 指标 | 值 | 说明 |\n|---|---|---|\n")
    f.write(f"| 单步均值 | {mean_step:.1f} ms | 跳过前3步 warmup, n={len(step_times_stable)} |\n")
    f.write(f"| 单步标准差 | {std_step:.1f} ms | 稳定性 |\n")
    f.write(f"| TPS (samples/s) | {mean_tps:.3f} | 全局批大小/单步时间 |\n")
    f.write(f"| WPS (tokens/s) | {int(mean_wps):,} | 含 seq=4096 |\n")
    f.write(f"| 60步总耗时 | {total_elapsed}s ({total_elapsed//60}min {total_elapsed%60}s) | |\n\n")

    f.write("## 硬件利用\n\n")
    f.write("| 指标 | 峰值 | 均值 |\n|---|---|---|\n")
    f.write(f"| HBM 占用 (MB) | {hbm_peak:,} / 65536 ({hbm_peak/65536*100:.1f}%) | {hbm_mean:.0f} MB |\n")
    f.write(f"| AI Core 利用率 | {aic_peak}% | {aic_mean:.1f}% |\n")
    f.write(f"| 整机功耗 | - | {pwr_mean:.1f} W/卡 |\n\n")

    f.write("## 训练质量\n\n")
    f.write("| 指标 | 值 |\n|---|---|\n")
    f.write(f"| 初始 loss | {loss_start:.4f} |\n")
    f.write(f"| 末尾 loss | {loss_end:.4f} |\n")
    f.write(f"| loss 变化 | {'↓ 收敛' if loss_end < loss_start else '→ 稳定'} |\n\n")

    f.write("## 关键优化项 vs 官方默认\n\n")
    f.write("| 优化项 | 官方35B默认 | 本次配置 | 预期收益 |\n|---|---|---|---|\n")
    f.write("| 通信重叠 | 关(prefetch=0) | 开(prefetch=1) | AI Core 利用率↑ |\n")
    f.write("| 专家并行 EP | 1(未开) | 8 | 专家显存↓,通信局域化 |\n")
    f.write("| EP分发策略 | fused | mc2 (昇腾融合) | 专家路由开销↓ |\n")
    f.write("| 重计算 | 整层全量 | 选择性(仅experts) | 重算FLOPs↓,吞吐↑ |\n")
    f.write("| 序列长度 | 1024 | 4096 | HBM利用率↑ |\n")
    f.write("| 微调方式 | 全参 | LoRA r=16 | 显存↓,可调参数2-3% |\n\n")

    f.write("## 数据来源\n\n")
    f.write(f"- 训练日志: `{train_log}`\n")
    if npu_log and os.path.exists(npu_log):
        f.write(f"- NPU 监控: `{npu_log}`\n")
    f.write(f"- 步时样本数: {len(step_times)}\n")
    f.write(f"- HBM 采样数: {len(hbm_vals)}\n")

print(f"报告已写入: {report_out}")
PYEOF

log "=== 流水线全部完成 ==="
log "训练日志: $TRAIN_LOG"
log "报告文件: $REPORT_DIR/qwen35_35B_lora_60step_perf_*.md"
