#!/usr/bin/env python3
# 从训练日志提取性能指标：单步耗时、WPS、TPS、loss、TFLOP
import sys, re, json

log = sys.argv[1]
seq_len = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
warmup = int(sys.argv[3]) if len(sys.argv) > 3 else 5

steps = []
with open(log, errors="ignore") as f:
    for line in f:
        m = re.search(r"iteration\s+(\d+)/\s*(\d+).*?consumed samples:\s*(\d+).*?elapsed time per iteration \(ms\):\s*([\d.]+).*?global batch size:\s*(\d+).*?lm loss:\s*([\d.E+-]+).*?grad norm:\s*([\d.]+)", line)
        if m:
            steps.append({
                "iter": int(m.group(1)),
                "consumed": int(m.group(3)),
                "ms": float(m.group(4)),
                "gbs": int(m.group(5)),
                "loss": float(m.group(6)),
                "grad_norm": float(m.group(7)),
            })

if not steps:
    print(json.dumps({"error": "no iterations parsed", "log": log}))
    sys.exit(0)

stable = steps[warmup:] if len(steps) > warmup else steps
ms_vals = [s["ms"] for s in stable]
gbs = stable[0]["gbs"]
mean_ms = sum(ms_vals) / len(ms_vals)
# WPS = tokens/sec (实际可变长，这里按 gbs*seq 估上界), TPS = samples/sec
tokens_per_step = gbs * seq_len
result = {
    "log": log,
    "total_steps": len(steps),
    "stable_steps": len(stable),
    "gbs": gbs,
    "seq_len": seq_len,
    "step_ms": {"mean": round(mean_ms, 1), "min": round(min(ms_vals), 1), "max": round(max(ms_vals), 1),
                "std": round((sum((x-mean_ms)**2 for x in ms_vals)/len(ms_vals))**0.5, 1)},
    "throughput": {
        "samples_per_sec_TPS": round(gbs / (mean_ms/1000), 2),
        "tokens_per_sec_WPS": round(tokens_per_step / (mean_ms/1000), 0),
    },
    "loss": {"first": steps[0]["loss"], "last": steps[-1]["loss"]},
    "grad_norm": {"mean": round(sum(s["grad_norm"] for s in stable)/len(stable), 3)},
    "nan_count": sum(1 for line in open(log, errors="ignore") if "nan iterations:   1" in line or "nan iterations:  1" in line),
}
print(json.dumps(result, indent=2, ensure_ascii=False))
