#!/usr/bin/env python3
import json
import re
import signal
import subprocess
import sys
import time
from datetime import datetime


OUT = sys.argv[1] if len(sys.argv) > 1 else "/data/sejin/baseline_26/metrics/npu_metrics_full.json"
DURATION = int(sys.argv[2]) if len(sys.argv) > 2 else 1200
INTERVAL = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0

samples = []
stop = False


def _handle_stop(_sig, _frame):
    global stop
    stop = True


signal.signal(signal.SIGTERM, _handle_stop)
signal.signal(signal.SIGINT, _handle_stop)


def sample_once():
    try:
        out = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return None

    lines = out.splitlines()
    chips = []
    for i, line in enumerate(lines):
        m1 = re.search(r"\|\s*(\d+)\s+910B\d?\s+\|\s+OK\s+\|\s+([\d.]+)\s+(\d+)", line)
        if not m1 or i + 1 >= len(lines):
            continue
        chip_id = int(m1.group(1))
        power = float(m1.group(2))
        m2 = re.search(r"\|\s*\d+\s+\|\s+[\w:.]+\s+\|\s+(\d+)\s+\d+\s*/\s*\d+\s+(\d+)\s*/\s*(\d+)", lines[i + 1])
        if not m2:
            continue
        chips.append({
            "chip": chip_id,
            "power_w": power,
            "aicore": float(m2.group(1)),
            "hbm_used": int(m2.group(2)),
            "hbm_total": int(m2.group(3)),
        })
    return chips


def summarize(window_samples):
    chip_rows = [chip for sample in window_samples for chip in sample.get("chips", [])]
    if not chip_rows:
        return {
            "num_samples": len(window_samples),
            "aicore_pct": {"mean": None, "peak": None},
            "hbm_used_mb": {"mean": None, "peak": None, "total": None},
            "power_w": {"mean": None, "peak": None},
        }
    return {
        "num_samples": len(window_samples),
        "aicore_pct": {
            "mean": round(sum(chip["aicore"] for chip in chip_rows) / len(chip_rows), 2),
            "peak": max(chip["aicore"] for chip in chip_rows),
        },
        "hbm_used_mb": {
            "mean": round(sum(chip["hbm_used"] for chip in chip_rows) / len(chip_rows), 1),
            "peak": max(chip["hbm_used"] for chip in chip_rows),
            "total": chip_rows[0].get("hbm_total"),
        },
        "power_w": {
            "mean": round(sum(chip["power_w"] for chip in chip_rows) / len(chip_rows), 2),
            "peak": max(chip["power_w"] for chip in chip_rows),
        },
    }


start_epoch = time.time()
start_iso = datetime.fromtimestamp(start_epoch).strftime("%Y-%m-%d %H:%M:%S.%f")

while not stop and time.time() - start_epoch < DURATION:
    chips = sample_once()
    now = time.time()
    if chips:
        samples.append({
            "t": round(now - start_epoch, 3),
            "ts_epoch": now,
            "ts": datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S.%f"),
            "chips": chips,
        })
    time.sleep(INTERVAL)

end_epoch = time.time()
result = {
    "start_epoch": start_epoch,
    "end_epoch": end_epoch,
    "start_time": start_iso,
    "end_time": datetime.fromtimestamp(end_epoch).strftime("%Y-%m-%d %H:%M:%S.%f"),
    "duration_s": round(end_epoch - start_epoch, 3),
    "interval_s": INTERVAL,
    "summary_all": summarize(samples),
    "raw_samples": samples,
}

with open(OUT, "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps({k: v for k, v in result.items() if k != "raw_samples"}, indent=2))
