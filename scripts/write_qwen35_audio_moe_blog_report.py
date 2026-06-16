#!/usr/bin/env python3
import json
import subprocess
from datetime import datetime
from pathlib import Path


BASE_DIR = Path("/data/sejin/baseline_26")
METRICS_DIR = BASE_DIR / "metrics"
REPORT = BASE_DIR / "reports" / "qwen35_audio_moe_blog_tuning_20260616.md"

NEW_TAGS = [
    "mbs2_fa2_fused_phaseprof_35",
    "mbs2_fa2_eager_ablation_35",
    "mbs2_fa2_mc2_probe_35",
    "mbs2_fa2_fused_bucket64_chunk1024_80",
    "mbs2_fa2_fused_bucket64_chunk512_80",
    "mbs2_fa2_fused_bucket32_chunk512_80",
    "mbs2_fa2_fused_bucket32_chunk512_empty4_80",
    "mbs2_fa2_fused_bucket32_chunk512_rc_on_80",
]

REFERENCE_TAGS = [
    "ep8_mbs1_ga4_rc_off_pad1280_current",
    "ep8_mbs1_ga4_rc_off_pad1408_nosync",
    "ep8_mbs1_ga4_rc_off_pad1536_nosync_rerun05",
    "ep8_mbs2_ga2_rc_off_pad128_bucket_fa2",
    "ep8_mbs2_ga2_rc_off_pad128_bucket64_fa2_nosync_chunk512",
]


def latest_analysis(tag):
    candidates = []
    for path in METRICS_DIR.glob(f"{tag}_*_analysis.json"):
        try:
            with path.open() as f:
                data = json.load(f)
        except Exception:
            continue
        if data.get("tag") == tag:
            candidates.append((path.stat().st_mtime, path, data))
    if not candidates:
        return None
    _, path, data = sorted(candidates, key=lambda item: item[0])[-1]
    data["_analysis_path"] = str(path)
    return data


def fmt(value, digits=2, suffix=""):
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return f"{value}{suffix}"
    return f"{value:.{digits}f}{suffix}"


def status_of(run):
    if run is None:
        return "not_run"
    measured = run.get("measured", {})
    error = run.get("error_category")
    steps = measured.get("num_steps") or 0
    cfg = run.get("config_summary", {})
    train_iters = cfg.get("train_iters") or 0
    skip_steps = run.get("skip_steps") or 10
    complete_steps = max(train_iters - skip_steps, 0) if train_iters else 70
    if error not in (None, "none", "None"):
        return error
    if steps >= complete_steps:
        return "success"
    return f"partial_{steps}_steps"


def row(tag, run):
    if run is None:
        return f"| `{tag}` | not_run | N/A | N/A | N/A | N/A | N/A | N/A | N/A |"
    measured = run.get("measured", {})
    hw = run.get("hardware_window", {})
    cfg = run.get("config_summary", {})
    step = (measured.get("step_time_s") or {}).get("mean")
    aic = hw.get("aicore_pct") or {}
    hbm = hw.get("hbm_used_mb") or {}
    power = hw.get("power_w") or {}
    return (
        f"| `{tag}` | {status_of(run)} | `{cfg.get('ep_dispatcher')}` | {fmt(step, 3)}s | "
        f"{fmt(measured.get('input_wps'), 1)} | "
        f"{fmt(aic.get('mean'), 2)} / {fmt(aic.get('peak'), 1)} | "
        f"{fmt(hbm.get('mean'), 1)} / {fmt(hbm.get('peak'), 0)} MB | "
        f"{fmt(power.get('mean'), 2)} / {fmt(power.get('peak'), 1)} W | "
        f"`{run.get('_analysis_path')}` |"
    )


def phase_rows(run):
    if not run:
        return []
    timing = run.get("phase_timing_ms", {})
    if not timing:
        return []
    return [
        f"| {name} | {fmt(value, 1)} ms |"
        for name, value in sorted(timing.items())
    ]


def moe_rows(run):
    if not run:
        return []
    moe = run.get("moe_phase", {}) or {}
    rows = []
    phase_ms = moe.get("phase_ms", {}) or {}
    for name, stats in phase_ms.items():
        rows.append(f"| {name} | {fmt(stats.get('mean'), 3)} | {fmt(stats.get('p90'), 3)} |")
    return rows


def moe_load_rows(run):
    if not run:
        return []
    moe = run.get("moe_phase", {}) or {}
    rows = []
    for name, stats in (moe.get("load_balance", {}) or {}).items():
        rows.append(f"| {name} | {fmt(stats.get('mean'), 3)} | {fmt(stats.get('p90'), 3)} |")
    return rows


def npu_snapshot():
    try:
        out = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=20).stdout
    except Exception as exc:
        return f"npu-smi failed: {exc}"
    process_lines = []
    capture = False
    for line in out.splitlines():
        if "Process id" in line:
            capture = True
        if capture:
            process_lines.append(line)
    return "\n".join(process_lines[-40:]) if process_lines else "No process section parsed."


def pick_best(runs):
    successful = [r for r in runs if status_of(r) == "success"]
    if not successful:
        return None

    def key(run):
        measured = run.get("measured", {})
        hw = run.get("hardware_window", {})
        aic = (hw.get("aicore_pct") or {}).get("mean") or 0
        hbm = (hw.get("hbm_used_mb") or {}).get("mean") or 0
        power = (hw.get("power_w") or {}).get("mean") or 0
        wps = measured.get("input_wps") or 0
        hbm_score = 1 if 55000 <= hbm <= 60000 else 0
        power_score = 1 if 220 <= power <= 260 else 0
        return (hbm_score, power_score, aic, wps)

    return sorted(successful, key=key)[-1]


def main():
    new_runs = [latest_analysis(tag) for tag in NEW_TAGS]
    ref_runs = [latest_analysis(tag) for tag in REFERENCE_TAGS]
    best_new = pick_best(new_runs)
    best_ref = pick_best(ref_runs)
    best = best_new or best_ref
    phase_probe = next((run for run in new_runs if run and (run.get("moe_phase") or {}).get("num_windows")), None)

    lines = []
    lines.append("# Qwen3.5 Audio Manual EP8 MoE Blog Tuning Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- Model architecture, expert count, Top-K routing semantics, Whisper encoder, and manual EP8 expert slicing are unchanged.")
    lines.append("- Adopted blog items that match this task: explicit fused/eager dispatcher comparison, CANN permute/unpermute and grouped matmul confirmation, EP AllToAll phase timing, and expert-count load-balance monitoring.")
    lines.append("- Deferred by design: group-limited routing, dynamic expert bias, capacity/drop policy, FP8 dispatch, TP/SP changes, and 2DH multi-node AllToAll.")
    lines.append("- Metrics are post-warmup only: skip first 10 logged steps; init, safe_open load, dataset build, and first compile are excluded from reported means.")
    lines.append("")
    lines.append("## Implementation")
    lines.append("")
    lines.append("- `ep_dispatcher.py` now has optional `MOE_PHASE_TIMING=1` profiling for `dispatch_preprocess`, pre/post AllToAll permute, dispatch/combine AllToAll, two GMMs, SwiGLU, and unpermute phases.")
    lines.append("- Per-run analysis records config snapshot, warmup-excluded step timing, NPU AICORE/HBM/power window, run phase times, and MoE phase/load-balance summaries when enabled.")
    lines.append("- New configs explicitly set `parallel.ep_plan.dispatcher` and regenerate model paths from `QWEN_MODEL_PATH` / `WHISPER_MODEL_PATH` for the remote host.")
    lines.append("")
    lines.append("## New Candidate Results")
    lines.append("")
    lines.append("| Config | Status | Dispatcher | Step mean | Input WPS | AICORE mean / peak | HBM mean / peak | Power mean / peak | Analysis |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for tag, run in zip(NEW_TAGS, new_runs):
        lines.append(row(tag, run))
    lines.append("")
    lines.append("## Reference Results")
    lines.append("")
    lines.append("| Config | Status | Dispatcher | Step mean | Input WPS | AICORE mean / peak | HBM mean / peak | Power mean / peak | Analysis |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for tag, run in zip(REFERENCE_TAGS, ref_runs):
        lines.append(row(tag, run))
    lines.append("")
    lines.append("## Best Available Recommendation")
    lines.append("")
    if best_new:
        lines.append(f"Best new complete run: `{best_new['tag']}`.")
    elif best_ref:
        lines.append(f"No new complete run is available yet. Best complete reference run remains `{best_ref['tag']}`.")
    else:
        lines.append("No complete run is available in the metrics directory.")
    lines.append("")
    lines.append("## Step Phase Timing For Best Available")
    lines.append("")
    if best:
        lines.append(f"Source: `{best['tag']}`")
        lines.append("")
        lines.append("| Phase | Mean |")
        lines.append("|---|---:|")
        lines.extend(phase_rows(best))
    else:
        lines.append("N/A")
    lines.append("")
    lines.append("## MoE EP Phase Timing Probe")
    lines.append("")
    if phase_probe:
        moe = phase_probe.get("moe_phase") or {}
        lines.append(f"Source: `{phase_probe['tag']}`, windows: {moe.get('num_windows')}")
        lines.append(f"Last input_splits: `{moe.get('last_input_splits')}`")
        lines.append(f"Last output_splits: `{moe.get('last_output_splits')}`")
        lines.append("")
        lines.append("| MoE phase | Mean ms | P90 ms |")
        lines.append("|---|---:|---:|")
        lines.extend(moe_rows(phase_probe) or ["| N/A | N/A | N/A |"])
        lines.append("")
        lines.append("| Expert load metric | Mean | P90 |")
        lines.append("|---|---:|---:|")
        lines.extend(moe_load_rows(phase_probe) or ["| N/A | N/A | N/A |"])
    else:
        lines.append("No MoE phase probe has completed yet.")
    lines.append("")
    lines.append("## Resource Snapshot At Report Generation")
    lines.append("")
    lines.append("```text")
    lines.append(npu_snapshot())
    lines.append("```")
    lines.append("")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines))
    print(REPORT)


if __name__ == "__main__":
    main()
