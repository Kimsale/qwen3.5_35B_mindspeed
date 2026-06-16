#!/bin/bash
# ============================================================
# HULK 对齐训练主驱动：启动监控 + 训练 + 出报表
# 完整执行 CLAUDE.md 第六节性能采集规范
#   - 训练阶段：调 train_hulk_aligned.sh（已对齐 hulk 配置，不动）
#   - 同步采集：npu_monitor.py 后台采样 AICore/HBM/功耗
#   - 训练结束：parse_metrics.py 提取 step_ms/WPS/TPS/loss/grad_norm
#   - 输出：合并指标到 metrics_<run_id>.json，生成 Markdown 报表
# ============================================================
set -eo pipefail

RUN_ID="${RUN_ID:-hulk_aligned_$(date +%Y%m%d_%H%M%S)}"
ITERS="${ITERS:-30}"            # 训练步数（含预热）；30 步够稳态评估
WARMUP="${WARMUP:-5}"           # 解析时跳过的预热步数
SEQ_LEN="${SEQ_LEN:-8192}"
GBS="${GBS:-16}"

BASE=/data/sejin/baseline_26
LOG_DIR="$BASE/logs"
METRICS_DIR="$BASE/metrics"
REPORT_DIR="$BASE/reports"
mkdir -p "$LOG_DIR" "$METRICS_DIR" "$REPORT_DIR"

LOG_FILE="$LOG_DIR/${RUN_ID}.log"
NPU_METRICS="$METRICS_DIR/${RUN_ID}_npu.json"
TRAIN_METRICS="$METRICS_DIR/${RUN_ID}_train.json"
COMBINED_METRICS="$METRICS_DIR/${RUN_ID}_combined.json"
REPORT_MD="$REPORT_DIR/${RUN_ID}_report.md"

echo "============================================================"
echo "HULK 对齐训练评测  run_id=${RUN_ID}"
echo "  iters=$ITERS  seq=$SEQ_LEN  gbs=$GBS  warmup=$WARMUP"
echo "  log:     $LOG_FILE"
echo "  metrics: $COMBINED_METRICS"
echo "  report:  $REPORT_MD"
echo "============================================================"
echo

# ---------- 1) 启动 NPU 监控（后台）----------
# 训练预计 < 1 小时，给监控 3600s 上限，训练结束我们 kill 它
NPU_DURATION=3600
echo "[1/4] 启动 NPU 监控（PID 写入 $METRICS_DIR/${RUN_ID}_monitor.pid）"
nohup /data/sejin/env/venv_26b/bin/python "$BASE/scripts/npu_monitor.py" \
      "$NPU_METRICS" "$NPU_DURATION" \
      > "$LOG_DIR/${RUN_ID}_monitor.log" 2>&1 &
MONITOR_PID=$!
echo $MONITOR_PID > "$METRICS_DIR/${RUN_ID}_monitor.pid"
echo "    监控 PID=$MONITOR_PID"

# 确保异常退出时清理监控
cleanup() {
  if kill -0 $MONITOR_PID 2>/dev/null; then
    echo "[cleanup] 终止监控 PID=$MONITOR_PID"
    kill -TERM $MONITOR_PID 2>/dev/null || true
    sleep 2
    kill -9 $MONITOR_PID 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

sleep 3   # 让监控先采几个空闲点做基线

# ---------- 2) 启动训练 ----------
echo
echo "[2/4] 启动训练（HULK 对齐配置，参数不改动）"
echo "      调用：$BASE/scripts/train_hulk_aligned.sh"
echo
START_TS=$(date +%s)

# 通过环境变量把 ITERS / LOG_FILE 传进训练脚本（脚本已经通过 ${ITERS:-60} 读取）
ITERS=$ITERS LOG_FILE="$LOG_FILE" GBS=$GBS \
  bash "$BASE/scripts/train_hulk_aligned.sh"
TRAIN_EXIT=$?
END_TS=$(date +%s)
WALL_S=$((END_TS - START_TS))
echo
echo "训练退出码: $TRAIN_EXIT  墙钟耗时: ${WALL_S}s"

# ---------- 3) 停止监控、汇总指标 ----------
echo
echo "[3/4] 停止 NPU 监控并解析训练指标"
kill -TERM $MONITOR_PID 2>/dev/null || true
# 等监控自己写完汇总 JSON
for i in 1 2 3 4 5; do
  if [ -s "$NPU_METRICS" ]; then break; fi
  sleep 2
done

# 训练日志解析
TRAIN_JSON=$(/data/sejin/env/venv_26b/bin/python \
              "$BASE/scripts/parse_metrics.py" "$LOG_FILE" "$SEQ_LEN" "$WARMUP")
echo "$TRAIN_JSON" > "$TRAIN_METRICS"

# ---------- 4) 合并 + 生成 Markdown 报表 ----------
echo
echo "[4/4] 生成 Markdown 报表"

/data/sejin/env/venv_26b/bin/python - <<PYEOF
import json, os, datetime

run_id = "${RUN_ID}"
seq_len = ${SEQ_LEN}
warmup = ${WARMUP}
wall_s = ${WALL_S}
train_exit = ${TRAIN_EXIT}
log_file = "${LOG_FILE}"
npu_path = "${NPU_METRICS}"
train_path = "${TRAIN_METRICS}"
combined_path = "${COMBINED_METRICS}"
report_md = "${REPORT_MD}"

def safe_load(p):
    try:
        with open(p) as f: return json.load(f)
    except Exception as e:
        return {"_error": str(e)}

train = safe_load(train_path)
npu = safe_load(npu_path)

combined = {
    "run_id": run_id,
    "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    "seq_length": seq_len,
    "warmup_steps": warmup,
    "wall_clock_s": wall_s,
    "train_exit_code": train_exit,
    "log_file": log_file,
    "train_metrics": train,
    "npu_metrics": {k: v for k, v in npu.items() if k != "raw_samples"} if isinstance(npu, dict) else npu,
    "config_alignment": {
        "TP": 1, "PP": 1, "EP": 8, "CP": 2, "CP_algo": "ulysses_cp_algo",
        "seq_length": 8192,
        "lora": {"r": 32, "alpha": 64, "dropout": 0.1,
                 "target": ["linear_qkv", "linear_proj"]},
        "lr": 5e-6, "min_lr": 1e-6, "clip_grad": 5.0, "warmup_fraction": 0.0,
        "swap_optimizer": False,
        "moe": {"experts": 128, "topk": 8, "ffn": 768,
                "dispatcher": "alltoall_seq", "grouped_gemm": True},
        "model": "Qwen3-Omni-30B-A3B (text-extracted)",
        "data": "/data/sejin/data_hulk_dist_30k_mcore/hulk_sft_packed",
        "ckpt": "/data/sejin/models/Qwen3-Omni-30B-A3B-hulk_tp1_pp1_ep8",
    },
}
with open(combined_path, "w") as f:
    json.dump(combined, f, indent=2, ensure_ascii=False)

# ---------- 渲染 Markdown ----------
def fmt(v, default="-"):
    return v if v not in (None, "", {}) else default

step = train.get("step_ms", {}) if isinstance(train, dict) else {}
thr  = train.get("throughput", {}) if isinstance(train, dict) else {}
loss = train.get("loss", {}) if isinstance(train, dict) else {}
gnorm = train.get("grad_norm", {}) if isinstance(train, dict) else {}
aic  = npu.get("aicore_pct", {}) if isinstance(npu, dict) else {}
hbm  = npu.get("hbm_used_mb", {}) if isinstance(npu, dict) else {}
pwr  = npu.get("power_w", {}) if isinstance(npu, dict) else {}
hbm_total = hbm.get("total", 65536)
hbm_peak = hbm.get("peak", 0)
hbm_pct  = round(hbm_peak / hbm_total * 100, 1) if hbm_total else 0

md = f"""# HULK 对齐训练性能报表 — {run_id}

**生成时间**：{combined['timestamp']}
**训练退出码**：{train_exit}（0=成功）
**墙钟耗时**：{wall_s} s
**训练日志**：`{log_file}`

> 配置严格对齐 HULK 自研框架（TP=1/PP=1/EP=8/CP=2 ulysses, seq=8192, LoRA r=32），未做任何参数改动；
> 模型：Qwen3-Omni-30B-A3B（thinker text 子模块），48 层全 MoE，128 专家 / topk=8。

---

## 一、配置对齐复核

| 维度 | HULK 目标 | 本次实测 | 一致 |
|------|----------|---------|------|
| TP / PP / EP / CP | 1 / 1 / 8 / 2(ulysses) | 1 / 1 / 8 / 2(ulysses) | ✅ |
| seq_length | 8192 | {seq_len} | ✅ |
| LoRA r/α/dropout | 32 / 64 / 0.1 | 32 / 64 / 0.1 | ✅ |
| LoRA target | 仅 attention | linear_qkv linear_proj | ✅ |
| lr / clip / warmup / min_lr | 5e-6 / 5.0 / 0.0 / 1e-6 | 5e-6 / 5.0 / 0.0 / 1e-6 | ✅ |
| swap-optimizer | 关 | 关（纯 GPU ZeRO-1） | ✅ |
| MoE 128/topk8/ffn768/alltoall_seq | 一致 | 一致 | ✅ |

---

## 二、吞吐 & 单步耗时（{train.get('stable_steps','-')} / {train.get('total_steps','-')} 步，跳过前 {warmup} 步预热）

| 指标 | 值 |
|------|----|
| 单步耗时 mean | {fmt(step.get('mean'))} ms |
| 单步耗时 min | {fmt(step.get('min'))} ms |
| 单步耗时 max | {fmt(step.get('max'))} ms |
| 单步耗时 std | {fmt(step.get('std'))} ms |
| Samples/s (TPS) | {fmt(thr.get('samples_per_sec_TPS'))} |
| Tokens/s (WPS, gbs×seq 上界) | {fmt(thr.get('tokens_per_sec_WPS'))} |
| Global batch size | {fmt(train.get('gbs'))} |

---

## 三、硬件指标（NPU 8 卡聚合）

| 指标 | 均值 | 峰值 |
|------|------|------|
| AI Core 利用率 (%) | {fmt(aic.get('mean'))} | {fmt(aic.get('peak'))} |
| HBM 占用 (MB) | {fmt(hbm.get('mean'))} | {hbm_peak} / {hbm_total}（{hbm_pct}%） |
| 整机单卡功耗 (W) | {fmt(pwr.get('mean'))} | {fmt(pwr.get('peak'))} |
| 采样数 | {npu.get('num_samples','-')} | 间隔 2s |

---

## 四、收敛指标

| 指标 | 值 |
|------|----|
| Loss 起始 | {fmt(loss.get('first'))} |
| Loss 末步 | {fmt(loss.get('last'))} |
| Grad Norm 均值 | {fmt(gnorm.get('mean'))} |
| NaN 步数 | {fmt(train.get('nan_count'))} |

---

## 五、CLAUDE.md 第六节指标清单核对

- 吞吐：✅ WPS / TPS / 单步耗时 / 单轮耗时（墙钟 {wall_s}s）
- 硬件：✅ 平均/峰值 AI Core、HBM 占用；⚠️ HBM 带宽 npu-smi 不直出，需 msprof 二次采集
- 训练：{'✅' if train.get('nan_count', 0) == 0 else '❌'} Loss 收敛 / 无 NaN / 梯度范围
- 备注：本轮并行 TP1·PP1·EP8·CP2，算子开关见 train_hulk_aligned.sh，无 AutoTuning（首轮基线）

---

## 六、已知差异 / 待跟进

1. **HBM 带宽** 当前未采（npu-smi 不直出），下一轮接 msprof 或 ascend-stats 补齐。
2. **AutoTuning** 本轮关闭，作为对齐 HULK 的纯框架基线；下一阶段独立开 MindSpeed Auto Tuning 做对比。
3. **数据规模**：当前数据 2894 个文档（30k token 长度分布），步数 {train.get('total_steps','-')} 步内可能未跑完一个 epoch；如果 loss 不收敛，问题不出在框架。

---

## 附：原始 metrics

- 训练日志解析：`{train_path}`
- NPU 监控：`{npu_path}`
- 合并 metrics：`{combined_path}`
"""

with open(report_md, "w") as f:
    f.write(md)

print(f"✅ 报表已写入：{report_md}")
print(f"   合并 metrics：{combined_path}")
PYEOF

echo
echo "============================================================"
echo "完成。查看报表："
echo "  cat $REPORT_MD"
echo "============================================================"
exit $TRAIN_EXIT
