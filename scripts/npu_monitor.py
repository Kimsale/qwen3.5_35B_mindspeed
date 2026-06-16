#!/usr/bin/env python3
# NPU 性能指标采集器：训练运行时每隔 N 秒采样 npu-smi，输出峰值/均值
# 采集: AICore%、HBM占用(MB)、HBM带宽、功耗(W)
import subprocess, time, re, sys, json, signal

INTERVAL = 2.0
OUT = sys.argv[1] if len(sys.argv) > 1 else "/data/sejin/baseline_26/metrics/npu_metrics.json"
DURATION = int(sys.argv[2]) if len(sys.argv) > 2 else 600

samples = []  # 每条: {t, per_chip:[{aicore,hbm_mb,power}]}
stop = False
def handler(s, f):
    global stop; stop = True
signal.signal(signal.SIGTERM, handler)
signal.signal(signal.SIGINT, handler)

def sample_once():
    """解析 npu-smi info（两行一卡）：
    行1: | N  910B3 | OK | <power> <temp> <hugepage> |  -> power
    行2: | 0  <bus>  | <aicore> <mem>/<mem> <hbm_used>/<hbm_total> | -> aicore,hbm
    """
    try:
        out = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return None
    lines = out.splitlines()
    chips = []
    for i, line in enumerate(lines):
        # 行1: chip id + 910B + OK + power
        m1 = re.search(r"\|\s*(\d+)\s+910B\d?\s+\|\s+OK\s+\|\s+([\d.]+)\s+(\d+)", line)
        if m1 and i + 1 < len(lines):
            chip_id = int(m1.group(1))
            power = float(m1.group(2))
            # 下一行: aicore% + ... + hbm_used/hbm_total
            m2 = re.search(r"\|\s*\d+\s+\|\s+[\w:.]+\s+\|\s+(\d+)\s+\d+\s*/\s*\d+\s+(\d+)\s*/\s*(\d+)", lines[i+1])
            if m2:
                chips.append({"chip": chip_id,
                              "power_w": power,
                              "aicore": float(m2.group(1)),
                              "hbm_used": int(m2.group(2)),
                              "hbm_total": int(m2.group(3))})
    return chips

t0 = time.time()
while not stop and (time.time() - t0) < DURATION:
    chips = sample_once()
    if chips:
        samples.append({"t": round(time.time() - t0, 1), "chips": chips})
    time.sleep(INTERVAL)

# 汇总：跳过前 20% 预热样本
warm = samples[len(samples)//5:] if len(samples) > 5 else samples
chip_count = sum(len(s["chips"]) for s in warm)
power_vals = [c["power_w"] for s in warm for c in s["chips"] if c.get("power_w") is not None]

result = {
    "num_samples": len(samples),
    "duration_s": round(time.time() - t0, 1),
    "aicore_pct": {"mean": round(sum(c["aicore"] for s in warm for c in s["chips"]) / chip_count, 1) if chip_count else None,
                   "peak": max((c["aicore"] for s in warm for c in s["chips"]), default=None)},
    "hbm_used_mb": {"mean": round(sum(c["hbm_used"] for s in warm for c in s["chips"]) / chip_count, 0) if chip_count else None,
                    "peak": max((c["hbm_used"] for s in warm for c in s["chips"]), default=None),
                    "total": warm[0]["chips"][0]["hbm_total"] if warm and warm[0]["chips"] else None},
    "power_w": {"mean": round(sum(power_vals) / len(power_vals), 1) if power_vals else None,
                "peak": max((c["power_w"] for s in warm for c in s["chips"] if c.get("power_w")), default=None)},
    "raw_samples": samples[-50:],
}
with open(OUT, "w") as f:
    json.dump(result, f, indent=2)
print(json.dumps({k: v for k, v in result.items() if k != "raw_samples"}, indent=2))
