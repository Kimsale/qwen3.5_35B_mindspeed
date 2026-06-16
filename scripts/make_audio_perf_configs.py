#!/usr/bin/env python3
import os
from copy import deepcopy
from pathlib import Path

import yaml


BASE = Path("/data/sejin/third_party/mindspeed-mm-26.0.0/examples/qwen3_5_audio/distfix60_manual_ep8_config.yaml")
OUT_DIR = Path("/data/sejin/third_party/mindspeed-mm-26.0.0/examples/qwen3_5_audio/perf_tuning")
OUT_DIR.mkdir(parents=True, exist_ok=True)

QWEN_MODEL_PATH = os.getenv("QWEN_MODEL_PATH", "/mnt/shared_data_196/sejin/models/Qwen3.5-35B-A3B")
WHISPER_MODEL_PATH = os.getenv("WHISPER_MODEL_PATH", "/mnt/shared_data_196/sejin/models/whisper-large-v3")


COMMON = {
    "micro_batch_size": 2,
    "gradient_accumulation_steps": 2,
    "pad_to_multiple_of": 128,
    "sampler_type": "LengthBucketBatchSampler",
    "gradient_accumulation_no_sync": True,
    "attn_implementation": "flash_attention_2",
}

VARIANTS = [
    {
        **COMMON,
        "tag": "mbs2_fa2_fused_phaseprof_35",
        "recompute": False,
        "length_bucket_size_multiplier": 64,
        "chunk_loss_size": 512,
        "dispatcher": "fused",
        "train_iters": 35,
        "log_micro_steps": True,
    },
    {
        **COMMON,
        "tag": "mbs2_fa2_eager_ablation_35",
        "recompute": False,
        "length_bucket_size_multiplier": 64,
        "chunk_loss_size": 512,
        "dispatcher": "eager",
        "train_iters": 35,
    },
    {
        **COMMON,
        "tag": "mbs2_fa2_mc2_probe_35",
        "recompute": False,
        "length_bucket_size_multiplier": 64,
        "chunk_loss_size": 512,
        "dispatcher": "mc2",
        "train_iters": 35,
    },
    {
        **COMMON,
        "tag": "mbs2_fa2_fused_bucket64_chunk1024_80",
        "recompute": False,
        "length_bucket_size_multiplier": 64,
        "chunk_loss_size": 1024,
        "dispatcher": "fused",
        "train_iters": 80,
    },
    {
        **COMMON,
        "tag": "mbs2_fa2_fused_bucket64_chunk512_80",
        "recompute": False,
        "length_bucket_size_multiplier": 64,
        "chunk_loss_size": 512,
        "dispatcher": "fused",
        "train_iters": 80,
    },
    {
        **COMMON,
        "tag": "mbs2_fa2_fused_bucket32_chunk512_80",
        "recompute": False,
        "length_bucket_size_multiplier": 32,
        "chunk_loss_size": 512,
        "dispatcher": "fused",
        "train_iters": 80,
    },
    {
        **COMMON,
        "tag": "mbs2_fa2_fused_bucket32_chunk512_empty4_80",
        "recompute": False,
        "length_bucket_size_multiplier": 32,
        "chunk_loss_size": 512,
        "dispatcher": "fused",
        "train_iters": 80,
        "empty_cache_interval": 4,
    },
    {
        **COMMON,
        "tag": "mbs2_fa2_fused_bucket32_chunk512_rc_on_80",
        "recompute": True,
        "length_bucket_size_multiplier": 32,
        "chunk_loss_size": 512,
        "dispatcher": "fused",
        "train_iters": 80,
    },
]


def make_variant(base_cfg, variant):
    cfg = deepcopy(base_cfg)
    tag = variant["tag"]

    cfg["parallel"]["recompute"] = variant["recompute"]
    if not variant["recompute"]:
        cfg["parallel"]["recompute_plan"] = {"apply_modules": []}
    cfg["parallel"].setdefault("ep_plan", {})
    cfg["parallel"]["ep_plan"]["dispatcher"] = variant["dispatcher"]

    cfg["training"]["micro_batch_size"] = variant["micro_batch_size"]
    cfg["training"]["gradient_accumulation_steps"] = variant["gradient_accumulation_steps"]
    cfg["training"]["train_iters"] = variant["train_iters"]
    cfg["training"]["save_interval"] = 0
    cfg["training"]["save"] = f"/data/sejin/baseline_26/output/{tag}"
    cfg["training"]["gradient_accumulation_no_sync"] = bool(variant["gradient_accumulation_no_sync"])
    cfg["training"]["perf_timing"] = {
        "enable": True,
        "sync": False,
        "log_tokens": True,
        "log_micro_steps": bool(variant.get("log_micro_steps", False)),
    }
    if "empty_cache_interval" in variant:
        cfg["training"]["empty_cache_interval"] = variant["empty_cache_interval"]
    else:
        cfg["training"].pop("empty_cache_interval", None)

    cfg["data"]["dataset_param"]["basic_parameters"]["cache_dir"] = (
        f"/data/sejin/baseline_26/data_audio_distfix_3200/cache_{tag}/"
    )
    cfg["data"]["dataloader_param"]["sampler_type"] = variant["sampler_type"]
    cfg["data"]["dataloader_param"]["length_bucket_size_multiplier"] = variant["length_bucket_size_multiplier"]
    cfg["data"]["dataloader_param"]["collate_param"]["pad_to_multiple_of"] = variant["pad_to_multiple_of"]

    cfg["model"]["attn_implementation"] = variant["attn_implementation"]
    cfg["model"]["chunkloss_plan"]["chunk_size"] = variant["chunk_loss_size"]
    cfg["model"]["use_grouped_expert_matmul"] = True

    cfg["model"]["model_name_or_path"] = QWEN_MODEL_PATH
    cfg["model"]["whisper_path"] = WHISPER_MODEL_PATH
    cfg["data"]["dataset_param"]["preprocess_parameters"]["model_name_or_path"] = QWEN_MODEL_PATH
    cfg["data"]["dataset_param"]["preprocess_parameters"]["audio_feature_extractor_path"] = WHISPER_MODEL_PATH
    cfg["training"]["manual_ep_hf_load"]["qwen_hf_dir"] = QWEN_MODEL_PATH
    cfg["training"]["manual_ep_hf_load"]["whisper_hf_dir"] = WHISPER_MODEL_PATH

    cfg["tools"]["profile"]["enable"] = False
    cfg["tools"]["memory_profile"]["enable"] = False
    return cfg


def main():
    base_cfg = yaml.safe_load(BASE.read_text())
    for variant in VARIANTS:
        tag = variant["tag"]
        path = OUT_DIR / f"{tag}.yaml"
        cfg = make_variant(base_cfg, variant)
        path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False))
        print(path)


if __name__ == "__main__":
    main()
