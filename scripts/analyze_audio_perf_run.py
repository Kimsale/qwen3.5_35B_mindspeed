#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
from datetime import datetime
from pathlib import Path


OUTER_TS_RE = re.compile(r"^\[Rank\s+\d+\s+\|\s+Local Rank\s+\d+\]\s+(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})")
ITER_RE = re.compile(
    r"iteration\s+(\d+)/\s*(\d+)\s+\|\s+consumed samples:\s+(\d+)\s+\|"
    r"\s+elapsed time per iteration \(ms\):\s+([\d.]+)\s+\|"
    r"\s+learning rate:\s+([-\d.E+]+)\s+\|\s+global batch size:\s+(\d+)\s+\|"
    r"\s+loss:\s+([-\d.E+]+)\s+\|(?:\s+grad norm:\s+([\d.]+)\s+\|)?"
)
INNER_TS_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")
PERF_RE = re.compile(
    r"perf timing ms:\s+(.*?)\s+\|\s+tokens:\s+input=(\d+),\s+label=(\d+),\s+audio=(\d+)\s+\|"
)
MOE_PHASE_RE = re.compile(r"\[moe_phase\]\s+(.*)")


MARKERS = {
    "first_log": None,
    "manual_qwen_load_start": "[manual_ep] loading Qwen",
    "manual_qwen_loaded": "[manual_ep] qwen tensors loaded",
    "manual_whisper_loaded": "[manual_ep] whisper encoder tensors loaded",
    "manual_audio_projector_init": "[manual_ep] audio projector tensors initialized",
    "manual_load_complete": "[manual_ep] weight loading complete",
    "lora_trainable_fixed": "[LoRA fix] post-FSDP trainable tensors",
    "training_example": "training example:",
    "save_lora": "LoRA_SAVE_MARKER",
}


def parse_ts(line):
    m = OUTER_TS_RE.search(line)
    if m:
        return datetime.strptime(f"{m.group(1)}.{m.group(2)}", "%Y-%m-%d %H:%M:%S.%f").timestamp()
    m = INNER_TS_RE.search(line)
    if m:
        return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
    return None


def percentile(values, pct):
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * pct / 100.0
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return values[int(k)]
    return values[lo] * (hi - k) + values[hi] * (k - lo)


def summarize_numbers(values):
    if not values:
        return {"mean": None, "median": None, "p90": None, "p95": None, "min": None, "max": None}
    return {
        "mean": sum(values) / len(values),
        "median": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "min": min(values),
        "max": max(values),
    }


def parse_key_values(payload):
    values = {}
    for item in payload.split():
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        values[key] = value
    return values


def summarize_moe_phase(rows):
    phase_keys = sorted({key for row in rows for key in row if key.endswith("_ms")})
    scalar_keys = sorted({
        key for row in rows for key in row
        if key.startswith("expert_counts_") and not key.endswith("_ms")
    })
    return {
        "num_windows": len(rows),
        "phase_ms": {
            key[:-3]: summarize_numbers([row[key] for row in rows if key in row])
            for key in phase_keys
        },
        "load_balance": {
            key: summarize_numbers([row[key] for row in rows if key in row])
            for key in scalar_keys
        },
        "last_input_splits": rows[-1].get("input_splits") if rows else None,
        "last_output_splits": rows[-1].get("output_splits") if rows else None,
    }


def _summarize_chip_rows(chips, num_samples):
    if not chips:
        return {
            "num_samples": num_samples,
            "num_chip_rows": 0,
            "aicore_pct": {"mean": None, "peak": None},
            "hbm_used_mb": {"mean": None, "peak": None, "p50": None, "p90": None, "total": None},
            "power_w": {"mean": None, "peak": None},
        }
    aicores = [chip["aicore"] for chip in chips]
    hbms = [chip["hbm_used"] for chip in chips]
    powers = [chip["power_w"] for chip in chips]
    return {
        "num_samples": num_samples,
        "num_chip_rows": len(chips),
        "aicore_pct": {
            "mean": round(sum(aicores) / len(aicores), 2),
            "p50": round(percentile(aicores, 50), 2),
            "p90": round(percentile(aicores, 90), 2),
            "peak": max(aicores),
        },
        "hbm_used_mb": {
            "mean": round(sum(hbms) / len(hbms), 1),
            "p50": round(percentile(hbms, 50), 1),
            "p90": round(percentile(hbms, 90), 1),
            "peak": max(hbms),
            "total": chips[0].get("hbm_total"),
        },
        "power_w": {
            "mean": round(sum(powers) / len(powers), 2),
            "p50": round(percentile(powers, 50), 2),
            "p90": round(percentile(powers, 90), 2),
            "peak": max(powers),
        },
    }


def summarize_chips(samples, active_hbm_threshold_mb=None):
    chips = []
    for sample in samples:
        for chip in sample.get("chips", []):
            if active_hbm_threshold_mb is not None and chip.get("hbm_used", 0) < active_hbm_threshold_mb:
                continue
            chips.append(chip)
    return _summarize_chip_rows(chips, len(samples))


def summarize_per_chip(samples):
    by_chip = {}
    for sample in samples:
        for chip in sample.get("chips", []):
            by_chip.setdefault(chip["chip"], []).append(chip)
    return {
        str(chip_id): _summarize_chip_rows(chip_rows, len(chip_rows))
        for chip_id, chip_rows in sorted(by_chip.items())
    }


def classify_errors(errors):
    joined = "\n".join(errors)
    if "out of memory" in joined or "NPU out of memory" in joined or "Memory_Allocation_Failure" in joined:
        return "oom"
    if "DataLoader timed out" in joined or "RuntimeError: DataLoader worker" in joined:
        return "dataloader_timeout"
    if "SignalException" in joined or "got signal: 15" in joined or "Received Signals.SIGTERM" in joined:
        return "terminated_or_hung"
    if "ChildFailedError" in joined or "RuntimeError" in joined or "Traceback" in joined:
        return "runtime_error"
    return "none" if not errors else "unknown"


def parse_train_log(path):
    steps = []
    markers = {}
    errors = []
    moe_phase_rows = []
    with open(path, "r", errors="replace") as f:
        for line in f:
            ts = parse_ts(line)
            if ts is not None and "first_log" not in markers:
                markers["first_log"] = ts
            for name, marker in MARKERS.items():
                if (
                    name == "save_lora"
                    and "Saved " in line
                    and " LoRA parameters" in line
                    and name not in markers
                    and ts is not None
                ):
                    markers[name] = ts
                    continue
                if marker and marker in line and name not in markers and ts is not None:
                    markers[name] = ts
            if any(token in line for token in ("Traceback", "RuntimeError", "OOM", "out of memory", "ChildFailedError", "ERR99999", "SignalException", "SIGTERM")):
                errors.append(line.strip())
            mm = MOE_PHASE_RE.search(line)
            if mm:
                parsed = parse_key_values(mm.group(1))
                row = {}
                for key, value in parsed.items():
                    if value.startswith("[") and value.endswith("]"):
                        try:
                            row[key] = [int(item) for item in value.strip("[]").split(",") if item != ""]
                        except ValueError:
                            row[key] = value
                    else:
                        try:
                            row[key] = float(value)
                        except ValueError:
                            row[key] = value
                moe_phase_rows.append(row)
            im = ITER_RE.search(line)
            if not im:
                continue
            step = {
                "iteration": int(im.group(1)),
                "train_iters": int(im.group(2)),
                "consumed_samples": int(im.group(3)),
                "elapsed_ms": float(im.group(4)),
                "lr": float(im.group(5)),
                "global_batch_size": int(im.group(6)),
                "loss": float(im.group(7)),
                "grad_norm": float(im.group(8)) if im.group(8) else None,
                "end_epoch": ts,
                "perf_ms": {},
                "tokens": {},
            }
            pm = PERF_RE.search(line)
            if pm:
                for item in pm.group(1).split(","):
                    if "=" not in item:
                        continue
                    k, v = item.strip().split("=", 1)
                    try:
                        step["perf_ms"][k] = float(v)
                    except ValueError:
                        pass
                step["tokens"] = {
                    "input": int(pm.group(2)),
                    "label": int(pm.group(3)),
                    "audio": int(pm.group(4)),
                }
            if ts is not None:
                step["start_epoch"] = ts - step["elapsed_ms"] / 1000.0
            steps.append(step)
    return steps, markers, errors, moe_phase_rows


def load_yaml_summary(config_path):
    try:
        import yaml
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
    except Exception:
        return {}
    parallel = cfg.get("parallel", {}) or {}
    training = cfg.get("training", {}) or {}
    data = ((cfg.get("data", {}) or {}).get("dataset_param", {}) or {}).get("basic_parameters", {}) or {}
    dataloader = (cfg.get("data", {}) or {}).get("dataloader_param", {}) or {}
    collate = dataloader.get("collate_param", {}) or {}
    model = cfg.get("model", {}) or {}
    lora = (training.get("lora", {}) or {})
    perf_timing = (training.get("perf_timing", {}) or {})
    fsdp_plan = parallel.get("fsdp_plan", {}) or {}
    return {
        "model_id": model.get("model_id"),
        "attn_implementation": model.get("attn_implementation"),
        "expert_parallel_size": parallel.get("expert_parallel_size"),
        "ep_dispatcher": (parallel.get("ep_plan", {}) or {}).get("dispatcher", "fused"),
        "fully_shard_parallel_size": parallel.get("fully_shard_parallel_size"),
        "tensor_parallel_size": parallel.get("tensor_parallel_size"),
        "ulysses_parallel_size": parallel.get("ulysses_parallel_size"),
        "recompute": parallel.get("recompute"),
        "param_dtype": fsdp_plan.get("param_dtype"),
        "reduce_dtype": fsdp_plan.get("reduce_dtype"),
        "micro_batch_size": training.get("micro_batch_size"),
        "gradient_accumulation_steps": training.get("gradient_accumulation_steps"),
        "gradient_accumulation_no_sync": training.get("gradient_accumulation_no_sync", False),
        "empty_cache_interval": training.get("empty_cache_interval", 0),
        "train_iters": training.get("train_iters"),
        "lr": training.get("lr"),
        "clip_grad": training.get("clip_grad"),
        "cutoff_len": data.get("cutoff_len"),
        "sampler_type": dataloader.get("sampler_type"),
        "length_bucket_size_multiplier": dataloader.get("length_bucket_size_multiplier"),
        "pad_to_multiple_of": collate.get("pad_to_multiple_of"),
        "chunk_loss_size": ((model.get("chunkloss_plan", {}) or {}).get("chunk_size")),
        "use_grouped_expert_matmul": model.get("use_grouped_expert_matmul"),
        "num_workers": dataloader.get("num_workers"),
        "prefetch_factor": dataloader.get("prefetch_factor"),
        "persistent_workers": dataloader.get("persistent_workers"),
        "dataloader_timeout": dataloader.get("timeout"),
        "perf_timing_sync": perf_timing.get("sync"),
        "perf_timing_log_micro_steps": perf_timing.get("log_micro_steps", False),
        "perf_timing_diagnostic_sync_phases": perf_timing.get("diagnostic_sync_phases", []),
        "lora_rank": lora.get("rank"),
        "lora_alpha": lora.get("alpha"),
        "lora_dropout": lora.get("dropout"),
    }


def fmt(value, digits=2):
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def write_markdown(path, result):
    measured = result["measured"]
    hw = result["hardware_window"]
    timing = result["phase_timing_ms"]
    cfg = result.get("config_summary", {})
    lines = []
    lines.append(f"# {result['tag']} Performance Analysis")
    lines.append("")
    lines.append(f"- Config: `{result['config']}`")
    lines.append(f"- Train log: `{result['train_log']}`")
    lines.append(f"- Monitor JSON: `{result['monitor_json']}`")
    lines.append(f"- Warmup skipped steps: `{result['skip_steps']}`")
    lines.append("")
    lines.append("## Config Snapshot")
    lines.append("")
    lines.append("| Item | Value |")
    lines.append("|---|---:|")
    for key, value in cfg.items():
        lines.append(f"| {key} | `{value}` |")
    lines.append("")
    runtime_env = result.get("runtime_env", {})
    if runtime_env:
        lines.append("## Runtime Env")
        lines.append("")
        lines.append("| Item | Value |")
        lines.append("|---|---:|")
        for key, value in runtime_env.items():
            lines.append(f"| {key} | `{value}` |")
        lines.append("")
    lines.append("## Post-Warmup Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| measured steps | {measured['num_steps']} |")
    lines.append(f"| step time mean | {fmt(measured['step_time_s']['mean'], 3)} s |")
    lines.append(f"| step time p50 / p90 / p95 | {fmt(measured['step_time_s']['median'], 3)} / {fmt(measured['step_time_s']['p90'], 3)} / {fmt(measured['step_time_s']['p95'], 3)} s |")
    lines.append(f"| samples/s | {fmt(measured['samples_per_s'], 3)} |")
    lines.append(f"| input WPS | {fmt(measured['input_wps'], 1)} |")
    lines.append(f"| label WPS | {fmt(measured['label_wps'], 1)} |")
    lines.append(f"| audio-pad WPS | {fmt(measured['audio_wps'], 1)} |")
    lines.append(f"| loss first -> last measured | {fmt(measured['loss_first'], 4)} -> {fmt(measured['loss_last'], 4)} |")
    lines.append(f"| AICORE mean / peak | {fmt(hw['aicore_pct']['mean'], 2)}% / {fmt(hw['aicore_pct']['peak'], 1)}% |")
    lines.append(f"| HBM mean / peak | {fmt(hw['hbm_used_mb']['mean'], 1)} / {fmt(hw['hbm_used_mb']['peak'], 0)} MB |")
    lines.append(f"| Power mean / peak | {fmt(hw['power_w']['mean'], 2)} / {fmt(hw['power_w']['peak'], 1)} W |")
    active_hw = result.get("hardware_window_active", {})
    if active_hw:
        lines.append(f"| active-chip AICORE mean / p90 / peak | {fmt(active_hw['aicore_pct']['mean'], 2)}% / {fmt(active_hw['aicore_pct'].get('p90'), 1)}% / {fmt(active_hw['aicore_pct']['peak'], 1)}% |")
        lines.append(f"| active-chip HBM mean / p90 / peak | {fmt(active_hw['hbm_used_mb']['mean'], 1)} / {fmt(active_hw['hbm_used_mb'].get('p90'), 0)} / {fmt(active_hw['hbm_used_mb']['peak'], 0)} MB |")
        lines.append(f"| active-chip Power mean / p90 / peak | {fmt(active_hw['power_w']['mean'], 2)} / {fmt(active_hw['power_w'].get('p90'), 1)} / {fmt(active_hw['power_w']['peak'], 1)} W |")
    lines.append(f"| error category | `{result.get('error_category', 'none')}` |")
    lines.append("")
    lines.append("## Average Step Phase Timing")
    lines.append("")
    lines.append("| Phase | Mean ms | Share |")
    lines.append("|---|---:|---:|")
    total_ms = measured["step_time_s"]["mean"] * 1000.0 if measured["step_time_s"]["mean"] else None
    for key, value in timing.items():
        share = value / total_ms * 100.0 if total_ms else None
        lines.append(f"| {key} | {fmt(value, 1)} | {fmt(share, 1)}% |")
    moe_phase = result.get("moe_phase", {}) or {}
    if moe_phase.get("num_windows"):
        lines.append("")
        lines.append("## MoE EP Phase Timing")
        lines.append("")
        lines.append(f"- MoE timing windows: {moe_phase['num_windows']}")
        lines.append(f"- Last input_splits: `{moe_phase.get('last_input_splits')}`")
        lines.append(f"- Last output_splits: `{moe_phase.get('last_output_splits')}`")
        lines.append("")
        lines.append("| MoE phase | Mean ms | P90 ms |")
        lines.append("|---|---:|---:|")
        for key, stats in moe_phase.get("phase_ms", {}).items():
            lines.append(f"| {key} | {fmt(stats.get('mean'), 3)} | {fmt(stats.get('p90'), 3)} |")
        lines.append("")
        lines.append("| Expert load metric | Mean | P90 |")
        lines.append("|---|---:|---:|")
        for key, stats in moe_phase.get("load_balance", {}).items():
            lines.append(f"| {key} | {fmt(stats.get('mean'), 3)} | {fmt(stats.get('p90'), 3)} |")
    lines.append("")
    lines.append("## Run Phase Times")
    lines.append("")
    lines.append("| Phase | Seconds |")
    lines.append("|---|---:|")
    for key, value in result["run_phase_s"].items():
        lines.append(f"| {key} | {fmt(value, 2)} |")
    per_chip = result.get("hardware_window_per_chip", {})
    if per_chip:
        lines.append("")
        lines.append("## Per-Chip Hardware Window")
        lines.append("")
        lines.append("| Chip | AICORE mean / peak | HBM mean / peak MB | Power mean / peak W |")
        lines.append("|---:|---:|---:|---:|")
        for chip_id, chip_summary in per_chip.items():
            lines.append(
                f"| {chip_id} | {fmt(chip_summary['aicore_pct']['mean'], 2)} / {fmt(chip_summary['aicore_pct']['peak'], 1)} | "
                f"{fmt(chip_summary['hbm_used_mb']['mean'], 1)} / {fmt(chip_summary['hbm_used_mb']['peak'], 0)} | "
                f"{fmt(chip_summary['power_w']['mean'], 2)} / {fmt(chip_summary['power_w']['peak'], 1)} |"
            )
    if result["errors"]:
        lines.append("")
        lines.append("## Errors")
        for err in result["errors"][:20]:
            lines.append(f"- `{err}`")
    lines.append("")
    Path(path).write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-log", required=True)
    parser.add_argument("--monitor-json", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--skip-steps", type=int, default=10)
    args = parser.parse_args()

    steps, markers, errors, moe_phase_rows = parse_train_log(args.train_log)
    measured_steps = [s for s in steps if s["iteration"] > args.skip_steps]
    step_times_s = [s["elapsed_ms"] / 1000.0 for s in measured_steps]
    total_time_s = sum(step_times_s)
    total_samples = sum(s["global_batch_size"] for s in measured_steps)
    token_sums = {
        "input": sum(s.get("tokens", {}).get("input", 0) for s in measured_steps),
        "label": sum(s.get("tokens", {}).get("label", 0) for s in measured_steps),
        "audio": sum(s.get("tokens", {}).get("audio", 0) for s in measured_steps),
    }

    perf_keys = sorted({k for s in measured_steps for k in s.get("perf_ms", {})})
    phase_timing_ms = {}
    for key in perf_keys:
        vals = [s["perf_ms"][key] for s in measured_steps if key in s.get("perf_ms", {})]
        if vals:
            phase_timing_ms[key] = sum(vals) / len(vals)

    window_start = min((s.get("start_epoch") for s in measured_steps if s.get("start_epoch")), default=None)
    window_end = max((s.get("end_epoch") for s in measured_steps if s.get("end_epoch")), default=None)
    monitor = json.loads(Path(args.monitor_json).read_text())
    raw_samples = monitor.get("raw_samples", [])
    if window_start is not None and window_end is not None:
        window_samples = [s for s in raw_samples if window_start <= s.get("ts_epoch", 0) <= window_end]
    else:
        window_samples = []

    run_phase_s = {}
    if markers.get("first_log") and markers.get("manual_qwen_load_start"):
        run_phase_s["startup_to_manual_load"] = markers["manual_qwen_load_start"] - markers["first_log"]
    if markers.get("manual_qwen_load_start") and markers.get("manual_qwen_loaded"):
        run_phase_s["qwen_safe_open_load"] = markers["manual_qwen_loaded"] - markers["manual_qwen_load_start"]
    if markers.get("manual_qwen_loaded") and markers.get("manual_whisper_loaded"):
        run_phase_s["whisper_load"] = markers["manual_whisper_loaded"] - markers["manual_qwen_loaded"]
    if markers.get("manual_whisper_loaded") and markers.get("manual_load_complete"):
        run_phase_s["projector_and_final_barrier"] = markers["manual_load_complete"] - markers["manual_whisper_loaded"]
    if markers.get("manual_load_complete") and steps:
        run_phase_s["post_load_to_iter1_end"] = steps[0]["end_epoch"] - markers["manual_load_complete"]
    if steps:
        run_phase_s["step1_time"] = steps[0]["elapsed_ms"] / 1000.0
        run_phase_s["all_logged_steps"] = sum(s["elapsed_ms"] for s in steps) / 1000.0
        warmup_steps = [s for s in steps if s["iteration"] <= args.skip_steps]
        run_phase_s["warmup_logged_steps"] = sum(s["elapsed_ms"] for s in warmup_steps) / 1000.0
        run_phase_s["measured_logged_steps"] = sum(s["elapsed_ms"] for s in measured_steps) / 1000.0
    if markers.get("save_lora") and steps:
        run_phase_s["last_step_to_lora_save"] = markers["save_lora"] - steps[-1]["end_epoch"]

    measured = {
        "num_steps": len(measured_steps),
        "step_time_s": summarize_numbers(step_times_s),
        "samples": total_samples,
        "samples_per_s": total_samples / total_time_s if total_time_s > 0 else None,
        "input_tokens": token_sums["input"],
        "label_tokens": token_sums["label"],
        "audio_tokens": token_sums["audio"],
        "input_wps": token_sums["input"] / total_time_s if total_time_s > 0 and token_sums["input"] else None,
        "label_wps": token_sums["label"] / total_time_s if total_time_s > 0 and token_sums["label"] else None,
        "audio_wps": token_sums["audio"] / total_time_s if total_time_s > 0 and token_sums["audio"] else None,
        "loss_first": measured_steps[0]["loss"] if measured_steps else None,
        "loss_last": measured_steps[-1]["loss"] if measured_steps else None,
    }

    result = {
        "tag": args.tag,
        "config": args.config,
        "train_log": args.train_log,
        "monitor_json": args.monitor_json,
        "skip_steps": args.skip_steps,
        "window_start_epoch": window_start,
        "window_end_epoch": window_end,
        "config_summary": load_yaml_summary(args.config),
        "runtime_env": {
            key: os.environ.get(key)
            for key in (
                "MULTI_STREAM_MEMORY_REUSE",
                "TASK_QUEUE_ENABLE",
                "PYTORCH_NPU_ALLOC_CONF",
                "ACLNN_CACHE_LIMIT",
                "CPU_AFFINITY_CONF",
                "HCCL_CONNECT_TIMEOUT",
            )
        },
        "measured": measured,
        "phase_timing_ms": phase_timing_ms,
        "moe_phase": summarize_moe_phase(moe_phase_rows),
        "run_phase_s": run_phase_s,
        "hardware_window": summarize_chips(window_samples),
        "hardware_window_active": summarize_chips(window_samples, active_hbm_threshold_mb=10000),
        "hardware_window_per_chip": summarize_per_chip(window_samples),
        "hardware_all": monitor.get("summary_all"),
        "markers": markers,
        "errors": errors,
        "error_category": classify_errors(errors),
    }

    Path(args.out_json).write_text(json.dumps(result, indent=2))
    write_markdown(args.out_md, result)
    print(json.dumps({
        "tag": result["tag"],
        "measured": result["measured"],
        "hardware_window": result["hardware_window"],
        "errors": len(errors),
    }, indent=2))


if __name__ == "__main__":
    main()
