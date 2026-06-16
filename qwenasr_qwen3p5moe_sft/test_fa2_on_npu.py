#!/usr/bin/env python3
"""
在NPU上验证flash_attention_2 concat模式与per-sample模式的数值一致性

必须在NPU环境下运行: python3.10 test_fa2_on_npu.py
"""
import os
import json
import torch
import torch_npu
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoder,
    Qwen3OmniMoeAudioEncoderConfig,
)
from safetensors.torch import load_file


def load_encoder(asr_path, attn_impl, device):
    with open(os.path.join(asr_path, "config.json")) as f:
        cfg = json.load(f)
    audio_config = Qwen3OmniMoeAudioEncoderConfig(**cfg["thinker_config"]["audio_config"])
    audio_config._attn_implementation = attn_impl
    enc = Qwen3OmniMoeAudioEncoder(audio_config)

    index_path = os.path.join(asr_path, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    state = {}
    for shard_file in set(index["weight_map"].values()):
        shard = load_file(os.path.join(asr_path, shard_file))
        for k, v in shard.items():
            if k.startswith("thinker.audio_tower."):
                state[k.replace("thinker.audio_tower.", "")] = v
    enc.load_state_dict(state, strict=True)
    enc = enc.to(dtype=torch.bfloat16, device=device)
    enc.eval()
    return enc


def main():
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    asr_path = "/data/yjjiang11/work/vibecoding/Qwen3.5/qwen3.5_MOE_AST_sft/models/Qwen/Qwen3-ASR-1.7B"

    # 准备测试数据
    torch.manual_seed(42)
    len1, len2 = 340, 510
    audio1 = torch.randn(128, len1, dtype=torch.bfloat16, device=device)
    audio2 = torch.randn(128, len2, dtype=torch.bfloat16, device=device)

    print("=" * 70)
    print("NPU Flash Attention Concat vs Per-Sample Test")
    print("=" * 70)
    print(f"Sample A: (128, {len1})    Sample B: (128, {len2})")
    print(f"Device: {device}")
    print()

    # ====== 1. Per-sample (FA2模式，每个样本独立编码) ======
    print("[1] Per-sample mode (flash_attention_2)")
    enc_fa2 = load_encoder(asr_path, "flash_attention_2", device)
    with torch.no_grad():
        out_a = enc_fa2(audio1, feature_lens=torch.tensor([len1], device=device)).last_hidden_state
        out_b = enc_fa2(audio2, feature_lens=torch.tensor([len2], device=device)).last_hidden_state
    embed_persample = torch.cat([out_a, out_b], dim=0)
    print(f"    Sample A: {out_a.shape[0]} tokens, Sample B: {out_b.shape[0]} tokens")
    print(f"    Total: {embed_persample.shape}")

    # ====== 2. Concat + FA2 (新方案) ======
    print("\n[2] Concat + flash_attention_2")
    audio_concat = torch.cat([audio1, audio2], dim=1)  # (128, 850)
    feature_lens = torch.tensor([len1, len2], dtype=torch.int64, device=device)
    with torch.no_grad():
        out_concat = enc_fa2(audio_concat, feature_lens=feature_lens).last_hidden_state
    embed_concat = out_concat
    print(f"    Output: {embed_concat.shape}")
    del enc_fa2

    # ====== 3. Concat + SDPA (对照: 有信息泄露) ======
    print("\n[3] Concat + sdpa (对照, 有泄露)")
    enc_sdpa = load_encoder(asr_path, "sdpa", device)
    with torch.no_grad():
        out_sdpa = enc_sdpa(audio_concat, feature_lens=feature_lens).last_hidden_state
    embed_sdpa = out_sdpa
    print(f"    Output: {embed_sdpa.shape}")
    del enc_sdpa

    # ====== 对比 ======
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    len_a = out_a.shape[0]

    diff_fa2 = (embed_persample - embed_concat).abs()
    print(f"\n[Per-sample FA2] vs [Concat FA2]:")
    print(f"    Max diff:       {diff_fa2.max().item():.6e}")
    print(f"    Mean diff:      {diff_fa2.mean().item():.6e}")
    print(f"    Sample A max:   {diff_fa2[:len_a].max().item():.6e}")
    print(f"    Sample B max:   {diff_fa2[len_a:].max().item():.6e}")

    diff_sdpa = (embed_persample - embed_sdpa).abs()
    print(f"\n[Per-sample FA2] vs [Concat SDPA] (has leakage):")
    print(f"    Max diff:       {diff_sdpa.max().item():.6e}")
    print(f"    Mean diff:      {diff_sdpa.mean().item():.6e}")

    print("\n" + "=" * 70)
    fa2_pass = diff_fa2.max().item() < 1e-2  # bf16精度
    improvement = diff_sdpa.max().item() / max(diff_fa2.max().item(), 1e-10)

    if fa2_pass:
        print(f"PASS: FA2 concat与per-sample一致 (max diff={diff_fa2.max().item():.6e})")
    else:
        print(f"FAIL: FA2 concat与per-sample不一致 (max diff={diff_fa2.max().item():.6e})")

    print(f"FA2相比SDPA改善: {improvement:.0f}x")
    print("=" * 70)


if __name__ == "__main__":
    main()
