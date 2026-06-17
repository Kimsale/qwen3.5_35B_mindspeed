#!/bin/bash

# Legacy zero2 migration: keep the pack collator, but run through the
# Megatron-based pretrain_transformers entry with custom FSDP + distributed optimizer.

export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH
cd /data/sejin/third_party/mindspeed-mm-26.0.0
source /usr/local/Ascend/cann-8.5.0/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

VENV=/data/sejin/env/venv_qwen35
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"
export PYTHONPATH="/data/sejin/third_party/Megatron-LM-core_v0.12.1:/data/sejin/third_party/megatron-lm:/data/sejin/third_party/mindspeed-mm-26.0.0:${PYTHONPATH:-}"

export TASK_QUEUE_ENABLE=2
export ASCEND_LAUNCH_BLOCKING=0
export ACLNN_CACHE_LIMIT=100000
export CPU_AFFINITY_CONF=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_CONNECT_TIMEOUT=1800
export TOKENIZERS_PARALLELISM=false
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

LOG_DIR="/data/sejin/baseline_26/logs"
REPORT_DIR="/data/sejin/baseline_26/reports/perf_runs"
mkdir -p "$LOG_DIR" "$REPORT_DIR"
RUN_TS=$(date +%Y%m%d_%H%M%S)
RUN_TAG="${RUN_TAG:-pack_legacy_zero2}"
TRAIN_LOG="$LOG_DIR/qwen3_5_audio_${RUN_TAG}_${RUN_TS}.log"
echo "训练日志: $TRAIN_LOG"
logfile=$(basename "$TRAIN_LOG" .log)
config_path=${PERF_CONFIG:-examples/qwen3_5_audio/perf_tuning/ep8_mbs1_ga4_pack_legacy_zero2.yaml}

if [ -z "${MASTER_PORT:-}" ]; then
    MASTER_PORT=$("$VENV/bin/python" - <<'PY'
import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
    )
fi

"$VENV/bin/torchrun" \
    --nproc_per_node 8 \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr localhost \
    --master_port "$MASTER_PORT" \
    pretrain_transformers.py \
    "${config_path}" \
    --distributed-backend nccl \
    > "$TRAIN_LOG" 2>&1
RUN_EXIT_CODE=$?
export RUN_EXIT_CODE

chmod 440 "$TRAIN_LOG"
SUMMARY_JSON=$(python - "$TRAIN_LOG" "$config_path" <<'PY'
import json
import math
import os
import re
import statistics as stats
import sys
from pathlib import Path

import yaml

log_path = Path(sys.argv[1])
config_path = Path(sys.argv[2])
text = log_path.read_text(errors="ignore")
lines = text.splitlines()

cfg = yaml.safe_load(config_path.read_text())
gpt = cfg.get("gpt_args", {})
lora = cfg.get("lora_args", {})
data = cfg.get("data", {})
model = cfg.get("model", {})

step_times_ms = [float(x) for x in re.findall(r"elapsed time per iteration \(ms\): ([0-9.]+)", text)]
tokens_per_sample = [float(x) for x in re.findall(r"tokens per sample: ([0-9.]+)", text)]
mem_reserved = [int(x) for x in re.findall(r"max_memory_reserved: (\d+)", text)]
mem_allocated = [int(x) for x in re.findall(r"max_memory_allocated: (\d+)", text)]
phase_names = ["pregather", "get_batch", "move", "loss_setup", "forward", "backward", "clip", "optimizer", "lr_scheduler", "zero_grad", "profiler"]
phase_records = {name: [] for name in phase_names}
token_wps_records = {"input": [], "label": [], "audio": []}
global_batch_size = gpt.get("global_batch_size")

step_line_re = re.compile(
    r"iteration\s+(\d+)/\s*(\d+)\s+\|\s+consumed samples:\s+(\d+)\s+\|\s+elapsed time per iteration \(ms\): ([0-9.]+)"
)
token_line_re = re.compile(
    r"tokens per sample: ([0-9.]+)|tokens: input=(\d+), label=(\d+), audio=(\d+)"
)
wps_line_re = re.compile(r"wps: input=([0-9.]+), label=([0-9.]+), audio=([0-9.]+)")
phase_re = re.compile(
    r"perf timing ms: pregather=([0-9.]+), get_batch=([0-9.]+), move=([0-9.]+), "
    r"loss_setup=([0-9.]+), forward=([0-9.]+), backward=([0-9.]+), clip=([0-9.]+), "
    r"optimizer=([0-9.]+), lr_scheduler=([0-9.]+), zero_grad=([0-9.]+), profiler=([0-9.]+)"
)

consumed_samples = []
iterations = []
for line in lines:
    m = step_line_re.search(line)
    if m:
        iterations.append(int(m.group(1)))
        consumed_samples.append(int(m.group(3)))
        token_match = token_line_re.search(line)
        if token_match:
            if token_match.group(1):
                tokens_per_sample.append(float(token_match.group(1)))
            elif token_match.group(2) and global_batch_size:
                total_tokens = int(token_match.group(2)) + int(token_match.group(3)) + int(token_match.group(4))
                tokens_per_sample.append(total_tokens / float(global_batch_size))
        wps_match = wps_line_re.search(line)
        if wps_match:
            token_wps_records["input"].append(float(wps_match.group(1)))
            token_wps_records["label"].append(float(wps_match.group(2)))
            token_wps_records["audio"].append(float(wps_match.group(3)))
    p = phase_re.search(line)
    if p:
        for idx, name in enumerate(phase_names, 1):
            phase_records[name].append(float(p.group(idx)))

def pct(values, q):
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return stats.quantiles(values, n=100, method="inclusive")[q - 1]

def mean(values):
    return sum(values) / len(values) if values else None

avg_step_ms = mean(step_times_ms)
avg_step_s = avg_step_ms / 1000.0 if avg_step_ms is not None else None
samples_per_second = (global_batch_size / avg_step_s) if (global_batch_size and avg_step_s) else None
avg_tokens = mean(tokens_per_sample)
tokens_per_second = (samples_per_second * avg_tokens) if (samples_per_second and avg_tokens) else None
phase_means = {name: mean(values) for name, values in phase_records.items() if values}
phase_total_ms = sum(step_times_ms) if step_times_ms else None
phase_shares = {
    name: (sum(values) / phase_total_ms * 100.0) if phase_total_ms else None
    for name, values in phase_records.items() if values
}

summary = {
    "run_exit_code": int(os.environ.get("RUN_EXIT_CODE", "0")),
    "log_path": str(log_path),
    "config_path": str(config_path),
    "branch": os.popen("git -C /data/sejin/third_party/mindspeed-mm-26.0.0 branch --show-current").read().strip(),
    "run_time": os.popen("date +%Y-%m-%dT%H:%M:%S%z").read().strip(),
    "measured_steps": len(step_times_ms),
    "last_iteration": iterations[-1] if iterations else None,
    "last_consumed_samples": consumed_samples[-1] if consumed_samples else None,
    "step_time_mean_s": round(avg_step_s, 3) if avg_step_s is not None else None,
    "step_time_p50_s": round(pct(step_times_ms, 50) / 1000.0, 3) if pct(step_times_ms, 50) is not None else None,
    "step_time_p90_s": round(pct(step_times_ms, 90) / 1000.0, 3) if pct(step_times_ms, 90) is not None else None,
    "step_time_p95_s": round(pct(step_times_ms, 95) / 1000.0, 3) if pct(step_times_ms, 95) is not None else None,
    "samples_per_second": round(samples_per_second, 3) if samples_per_second is not None else None,
    "tokens_per_sample_mean": round(avg_tokens, 3) if avg_tokens is not None else None,
    "tokens_per_second": round(tokens_per_second, 3) if tokens_per_second is not None else None,
    "peak_reserved_mb": round(max(mem_reserved) / 1024 / 1024, 1) if mem_reserved else None,
    "peak_allocated_mb": round(max(mem_allocated) / 1024 / 1024, 1) if mem_allocated else None,
    "mean_reserved_mb": round(mean(mem_reserved) / 1024 / 1024, 1) if mem_reserved else None,
    "mean_allocated_mb": round(mean(mem_allocated) / 1024 / 1024, 1) if mem_allocated else None,
    "phase_means_ms": {k: round(v, 1) for k, v in phase_means.items() if v is not None},
    "phase_shares_pct": {k: round(v, 1) for k, v in phase_shares.items() if v is not None},
    "input_wps_mean": round(mean(token_wps_records["input"]), 1) if token_wps_records["input"] else None,
    "label_wps_mean": round(mean(token_wps_records["label"]), 1) if token_wps_records["label"] else None,
    "audio_wps_mean": round(mean(token_wps_records["audio"]), 1) if token_wps_records["audio"] else None,
    "global_batch_size": global_batch_size,
    "micro_batch_size": gpt.get("micro_batch_size"),
    "gradient_accumulation_steps": gpt.get("gradient_accumulation_steps"),
    "train_iters": gpt.get("train_iters"),
    "bf16": gpt.get("bf16"),
    "use_custom_fsdp": gpt.get("use_custom_fsdp"),
    "use_distributed_optimizer": gpt.get("use_distributed_optimizer"),
    "data_parallel_sharding_strategy": gpt.get("data_parallel_sharding_strategy"),
    "tensor_model_parallel_size": gpt.get("tensor_model_parallel_size"),
    "pipeline_model_parallel_size": gpt.get("pipeline_model_parallel_size"),
    "context_parallel_size": gpt.get("context_parallel_size"),
    "expert_model_parallel_size": gpt.get("expert_model_parallel_size"),
    "model_id": model.get("model_id"),
    "attn_implementation": model.get("attn_implementation"),
    "collate_model": data.get("dataloader_param", {}).get("collate_param", {}).get("model_name"),
    "seq_length": gpt.get("seq_length"),
    "run_tag": os.environ.get("RUN_TAG", "pack_legacy_zero2"),
    "lora_rank": lora.get("lora_r"),
    "lora_alpha": lora.get("lora_alpha"),
    "lora_dropout": lora.get("lora_dropout"),
    "dataset": data.get("dataset_param", {}).get("basic_parameters", {}).get("dataset"),
    "cache_dir": data.get("dataset_param", {}).get("basic_parameters", {}).get("cache_dir"),
    "error_lines": [],
}

for line in lines:
    if any(token in line for token in ("Traceback (most recent call last)", "RuntimeError:", "OOM", "ChildFailedError", "SIGTERM", "Error", "error")):
        if len(summary["error_lines"]) < 20:
            summary["error_lines"].append(line)

print(json.dumps(summary, ensure_ascii=False))
PY
)

export SUMMARY_JSON

STEP_TIME=$(python - <<'PY'
import json, os
s = json.loads(os.environ["SUMMARY_JSON"])
print("" if s["step_time_mean_s"] is None else f'{s["step_time_mean_s"]:.3f}')
PY
)
SAMPLES_PER_SECOND=$(python - <<'PY'
import json, os
s = json.loads(os.environ["SUMMARY_JSON"])
print("" if s["samples_per_second"] is None else f'{s["samples_per_second"]:.3f}')
PY
)
TOKENS_PER_SECOND=$(python - <<'PY'
import json, os
s = json.loads(os.environ["SUMMARY_JSON"])
print("" if s["tokens_per_second"] is None else f'{s["tokens_per_second"]:.3f}')
PY
)

echo "退出码 $RUN_EXIT_CODE"
echo "日志文件 $TRAIN_LOG"
echo "Elapsed Time Per iteration: $STEP_TIME"
echo "Average Samples per Second: $SAMPLES_PER_SECOND"
if [ -n "$TOKENS_PER_SECOND" ]; then
    echo "Consumed Tokens per Second: $TOKENS_PER_SECOND"
fi

REPORT_PATH="$REPORT_DIR/${logfile}.md"
python - "$REPORT_PATH" "$SUMMARY_JSON" <<'PY'
import json
import os
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
s = json.loads(sys.argv[2])

def fmt(v, suffix=""):
    if v is None or v == "":
        return "N/A"
    return f"{v}{suffix}"

lines = [
    "# Legacy Zero2 Pack Perf Report",
    "",
    "## Run Summary",
    "",
    "| Item | Value |",
    "|---|---|",
    f"| Branch | `{fmt(s['branch'])}` |",
    f"| Config | `{fmt(s['config_path'])}` |",
    f"| Log | `{fmt(s['log_path'])}` |",
    f"| Exit code | `{fmt(s['run_exit_code'])}` |",
    f"| Run time | `{fmt(s['run_time'])}` |",
    f"| Measured steps | `{fmt(s['measured_steps'])}` |",
    f"| Last iteration / consumed samples | `{fmt(s['last_iteration'])} / {fmt(s['last_consumed_samples'])}` |",
    f"| Step time mean | `{fmt(s['step_time_mean_s'], ' s')}` |",
    f"| Step time p50 / p90 / p95 | `{fmt(s['step_time_p50_s'], ' s')} / {fmt(s['step_time_p90_s'], ' s')} / {fmt(s['step_time_p95_s'], ' s')}` |",
    f"| Samples / second | `{fmt(s['samples_per_second'])}` |",
    f"| Tokens / sample | `{fmt(s['tokens_per_sample_mean'])}` |",
    f"| Tokens / second | `{fmt(s['tokens_per_second'])}` |",
    f"| Input / label / audio WPS | `{fmt(s['input_wps_mean'])} / {fmt(s['label_wps_mean'])} / {fmt(s['audio_wps_mean'])}` |",
    f"| HBM reserved mean / peak | `{fmt(s['mean_reserved_mb'], ' MB')} / {fmt(s['peak_reserved_mb'], ' MB')}` |",
    f"| HBM allocated mean / peak | `{fmt(s['mean_allocated_mb'], ' MB')} / {fmt(s['peak_allocated_mb'], ' MB')}` |",
    "",
    "## Config Snapshot",
    "",
    "| Item | Value |",
    "|---|---|",
    f"| model_id | `{fmt(s['model_id'])}` |",
    f"| entry | `pretrain_transformers.py` |",
    f"| attention | `{fmt(s['attn_implementation'])}` |",
    f"| optimizer path | `{('use_custom_fsdp + use_distributed_optimizer') if s.get('use_custom_fsdp') else ('DDP / regular optimizer' if not s.get('use_distributed_optimizer') else 'DDP / distributed optimizer')}` |",
    f"| data parallel sharding strategy | `{fmt(s['data_parallel_sharding_strategy'])}` |",
    f"| tensor / pipeline / context / expert parallel | `{fmt(s['tensor_model_parallel_size'])} / {fmt(s['pipeline_model_parallel_size'])} / {fmt(s['context_parallel_size'])} / {fmt(s['expert_model_parallel_size'])}` |",
    f"| micro batch size | `{fmt(s['micro_batch_size'])}` |",
    f"| gradient accumulation steps | `{fmt(s['gradient_accumulation_steps'])}` |",
    f"| global batch size | `{fmt(s['global_batch_size'])}` |",
    f"| precision | `bf16` |",
    f"| train iters | `{fmt(s['train_iters'])}` |",
    f"| seq length | `{fmt(s['seq_length'])}` |",
    f"| collator | `{fmt(s['collate_model'])}` |",
    f"| LoRA rank / alpha / dropout | `{fmt(s['lora_rank'])} / {fmt(s['lora_alpha'])} / {fmt(s['lora_dropout'])}` |",
    f"| dataset | `{fmt(s['dataset'])}` |",
    f"| cache dir | `{fmt(s['cache_dir'])}` |",
    "",
    "## Runtime Env",
    "",
    "| Item | Value |",
    "|---|---|",
    f"| TASK_QUEUE_ENABLE | `{fmt(os.environ.get('TASK_QUEUE_ENABLE'))}` |",
    f"| ASCEND_LAUNCH_BLOCKING | `{fmt(os.environ.get('ASCEND_LAUNCH_BLOCKING'))}` |",
    f"| ACLNN_CACHE_LIMIT | `{fmt(os.environ.get('ACLNN_CACHE_LIMIT'))}` |",
    f"| CPU_AFFINITY_CONF | `{fmt(os.environ.get('CPU_AFFINITY_CONF'))}` |",
    f"| PYTORCH_NPU_ALLOC_CONF | `{fmt(os.environ.get('PYTORCH_NPU_ALLOC_CONF'))}` |",
    f"| HCCL_CONNECT_TIMEOUT | `{fmt(os.environ.get('HCCL_CONNECT_TIMEOUT'))}` |",
    "",
]

phase_means = s.get("phase_means_ms", {})
phase_shares = s.get("phase_shares_pct", {})
if phase_means:
    lines += [
        "## Average Step Phase Timing",
        "",
        "| Phase | Mean ms | Share |",
        "|---|---:|---:|",
    ]
    for phase in ["pregather", "get_batch", "move", "loss_setup", "forward", "backward", "clip", "optimizer", "lr_scheduler", "zero_grad", "profiler"]:
        if phase in phase_means:
            share = phase_shares.get(phase, "N/A")
            lines.append(f"| {phase} | {phase_means[phase]} | {share}% |")
    lines.append("")

if s["error_lines"]:
    lines += [
        "## Log Excerpt",
        "",
        "| Line | Value |",
        "|---|---|",
    ]
    for i, line in enumerate(s["error_lines"], 1):
        safe_line = line.replace("`", "\\`")
        lines.append(f"| {i} | `{safe_line}` |")
    lines.append("")

report_path.write_text("\n".join(lines))
PY
echo "报告文件 $REPORT_PATH"
