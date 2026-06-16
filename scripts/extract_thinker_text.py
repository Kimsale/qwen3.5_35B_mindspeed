#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 Qwen3-Omni-30B-A3B (xuchen2) 抽取 thinker text 子模块，
转换为标准 Qwen3MoeForCausalLM HF 格式，供 MindSpeed 的 qwen3-moe
转换器使用。

源:  /data/xuchen2/model/Qwen3-Omni-30B-A3B-Captioner
     - 嵌套 config: thinker_config.text_config (model_type=qwen3_omni_moe_text)
     - 权重前缀: thinker.model.* 和 thinker.lm_head
     - 包含: audio_tower (525 张量) + visual (351 张量) [本次丢弃]

目标: <out>/
     - config.json: 平铺为标准 Qwen3MoeConfig (model_type=qwen3_moe)
     - model.safetensors.index.json + 16 个分片 (剥离 thinker. 前缀)
     - tokenizer.* 直接复制
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm


def build_qwen3_moe_config(omni_cfg: dict) -> dict:
    """从 Omni 嵌套 config 构造标准 Qwen3MoeConfig (HF 4.51+ 支持)。"""
    tc = omni_cfg["thinker_config"]["text_config"]

    # rope_scaling: Omni 的 mrope/interleaved 是多模态位置编码，
    # 纯文本 LLM 训练用不到 → 直接置 None，使用默认 RoPE
    cfg = {
        "architectures": ["Qwen3MoeForCausalLM"],
        "model_type": "qwen3_moe",
        "torch_dtype": "bfloat16",
        # 维度
        "hidden_size": tc["hidden_size"],                  # 2048
        "intermediate_size": tc["intermediate_size"],      # 768 (dense FFN, 不用)
        "moe_intermediate_size": tc["moe_intermediate_size"],  # 768
        "num_hidden_layers": tc["num_hidden_layers"],      # 48
        "num_attention_heads": tc["num_attention_heads"],  # 32
        "num_key_value_heads": tc["num_key_value_heads"],  # 4
        "head_dim": tc["head_dim"],                        # 128
        "vocab_size": tc["vocab_size"],                    # 152064
        "max_position_embeddings": tc["max_position_embeddings"],  # 65536
        # 激活/归一化
        "hidden_act": tc["hidden_act"],                    # silu
        "rms_norm_eps": tc["rms_norm_eps"],                # 1e-6
        "tie_word_embeddings": tc.get("tie_word_embeddings", False),
        "use_cache": tc.get("use_cache", True),
        "use_qk_norm": tc.get("use_qk_norm", True),
        # MoE
        "num_experts": tc["num_experts"],                  # 128
        "num_experts_per_tok": tc["num_experts_per_tok"],  # 8
        "norm_topk_prob": tc["norm_topk_prob"],
        "decoder_sparse_step": tc.get("decoder_sparse_step", 1),
        "mlp_only_layers": tc.get("mlp_only_layers", []),
        "router_aux_loss_coef": tc.get("router_aux_loss_coef", 0.001),
        "shared_expert_intermediate_size": tc.get("shared_expert_intermediate_size", 0),
        "output_router_logits": False,
        # RoPE & attention
        "rope_theta": tc["rope_theta"],                    # 1000000
        "rope_scaling": None,                              # 多模态 mrope 丢弃
        "attention_bias": tc.get("attention_bias", False),
        "attention_dropout": tc.get("attention_dropout", 0.0),
        "sliding_window": None,
        "use_sliding_window": False,
        "max_window_layers": tc["num_hidden_layers"],
        # 初始化
        "initializer_range": tc.get("initializer_range", 0.02),
        # tokens (从顶层取，便于 generation)
        "bos_token_id": None,
        "eos_token_id": [
            omni_cfg.get("im_end_token_id", 151645),
        ],
        "pad_token_id": None,
        "transformers_version": "4.57.1",
    }
    return cfg


def remap_key(k: str) -> str:
    """thinker.model.* -> model.*; thinker.lm_head -> lm_head"""
    if k.startswith("thinker.model."):
        return "model." + k[len("thinker.model."):]
    if k.startswith("thinker.lm_head"):
        return "lm_head" + k[len("thinker.lm_head"):]
    return None  # drop (audio_tower, visual)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Omni 源模型目录")
    ap.add_argument("--dst", required=True, help="抽取后的目标目录")
    ap.add_argument("--shard-bytes", type=int, default=4 * 1024**3,
                    help="每个 safetensors 分片最大字节数 (默认 4GB)")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    # ---- 1. 写 config.json (平铺) ----
    print("[1/4] 构造平铺 config.json")
    omni_cfg = json.load(open(src / "config.json"))
    new_cfg = build_qwen3_moe_config(omni_cfg)
    with open(dst / "config.json", "w") as f:
        json.dump(new_cfg, f, indent=2, ensure_ascii=False)
    print(f"      hidden={new_cfg['hidden_size']}, layers={new_cfg['num_hidden_layers']}, "
          f"experts={new_cfg['num_experts']}, topk={new_cfg['num_experts_per_tok']}, "
          f"vocab={new_cfg['vocab_size']}")

    # ---- 2. 复制 tokenizer / generation config ----
    print("[2/4] 复制 tokenizer/chat_template/generation_config")
    tok_files = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                 "chat_template.json", "special_tokens_map.json"]
    gen_files = ["generation_config.json"]
    for fn in tok_files + gen_files:
        s = src / fn
        if s.exists():
            shutil.copy2(s, dst / fn)
            print(f"      copied {fn}")

    # ---- 3. 加载 weight map, 筛选 text 权重 ----
    print("[3/4] 扫描权重")
    idx = json.load(open(src / "model.safetensors.index.json"))
    wm = idx["weight_map"]
    text_keys = [k for k in wm if k.startswith("thinker.model.") or k.startswith("thinker.lm_head")]
    print(f"      keep {len(text_keys)} text weights, drop {len(wm) - len(text_keys)} multimodal")

    # 按源 shard 分组以减少打开次数
    by_shard = {}
    for k in text_keys:
        by_shard.setdefault(wm[k], []).append(k)

    # ---- 4. 流式重写为新 shards ----
    print(f"[4/4] 流式写新 shard (最大 {args.shard_bytes/1e9:.1f}GB/shard)")
    new_index = {"metadata": {"total_size": 0}, "weight_map": {}}
    cur_shard = {}
    cur_bytes = 0
    shard_idx = 1
    total_shards_est = max(1, int(63.4e9 / args.shard_bytes) + 1)

    def flush(shard_dict, idx_no):
        if not shard_dict:
            return idx_no
        fn = f"model-{idx_no:05d}.safetensors"
        save_file(shard_dict, str(dst / fn), metadata={"format": "pt"})
        for k in shard_dict:
            new_index["weight_map"][k] = fn
        sz = sum(t.numel() * t.element_size() for t in shard_dict.values())
        new_index["metadata"]["total_size"] += sz
        print(f"      wrote {fn}: {len(shard_dict)} tensors, {sz/1e9:.2f} GB")
        return idx_no + 1

    pbar = tqdm(total=len(text_keys), desc="copy")
    for src_shard, keys in sorted(by_shard.items()):
        with safe_open(str(src / src_shard), framework="pt") as f:
            for k in keys:
                t = f.get_tensor(k)
                new_k = remap_key(k)
                if new_k is None:
                    pbar.update(1)
                    continue
                tsz = t.numel() * t.element_size()
                if cur_bytes > 0 and cur_bytes + tsz > args.shard_bytes:
                    shard_idx = flush(cur_shard, shard_idx)
                    cur_shard = {}
                    cur_bytes = 0
                cur_shard[new_k] = t
                cur_bytes += tsz
                pbar.update(1)
    pbar.close()
    flush(cur_shard, shard_idx)

    # 写 index
    # 修正 index 中的文件名为按总数补零
    n = shard_idx if not cur_shard else shard_idx
    # save_file 已经写好了文件，名字是 model-NNNNN.safetensors（5位），
    # HF 习惯是 model-XXXXX-of-YYYYY，但 5 位连续编号也可被 transformers 认（用 weight_map 索引），
    # 这里我们重命名为标准 of- 形式
    wm_old = new_index["weight_map"]
    files_used = sorted(set(wm_old.values()))
    total = len(files_used)
    rename_map = {}
    for i, fn in enumerate(files_used, 1):
        new_fn = f"model-{i:05d}-of-{total:05d}.safetensors"
        os.rename(dst / fn, dst / new_fn)
        rename_map[fn] = new_fn
    new_index["weight_map"] = {k: rename_map[v] for k, v in wm_old.items()}

    with open(dst / "model.safetensors.index.json", "w") as f:
        json.dump(new_index, f, indent=2, ensure_ascii=False)

    print()
    print(f"✅ 完成: {dst}")
    print(f"   shards: {total}")
    print(f"   total size: {new_index['metadata']['total_size']/1e9:.2f} GB")
    print(f"   weights: {len(new_index['weight_map'])}")


if __name__ == "__main__":
    main()
