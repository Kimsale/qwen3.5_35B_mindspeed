#!/usr/bin/env python3
"""CPU-level test for packed attention mask correctness.

Validates the block-diagonal causal mask logic extracted from
EPSpeechTranslationModel._create_packed_attention_mask, without needing the
full model or NPU. Catches NaN bugs and cross-sample leakage.
"""
import torch


def create_packed_attention_mask(cu_seqlens, total_len, device, dtype=torch.float32):
    """Standalone copy of model_ep.py mask builder (kept in sync)."""
    batch_size = len(cu_seqlens) - 1
    min_val = torch.finfo(dtype).min
    mask = torch.full((total_len, total_len), min_val, device=device, dtype=dtype)
    for b in range(batch_size):
        start = cu_seqlens[b].item()
        end = cu_seqlens[b + 1].item()
        seq_len = end - start
        causal_block = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        block = torch.where(
            causal_block,
            torch.zeros((), device=device, dtype=dtype),
            torch.full((), min_val, device=device, dtype=dtype),
        )
        mask[start:end, start:end] = block
    return mask.unsqueeze(0).unsqueeze(0)


def test_no_nan():
    print("=" * 60)
    print("Test 1: mask has no NaN/inf")
    print("=" * 60)
    cu = torch.tensor([0, 3, 5])
    m = create_packed_attention_mask(cu, 5, "cpu")
    assert not torch.isnan(m).any(), "Mask contains NaN!"
    assert not torch.isinf(m).any(), "Mask contains inf (should use finfo.min)!"
    print("✓ No NaN, no inf")


def test_block_diagonal_causal():
    print("\n" + "=" * 60)
    print("Test 2: block-diagonal causal structure")
    print("=" * 60)
    # Two samples: [0:3] and [3:5]
    cu = torch.tensor([0, 3, 5])
    m = create_packed_attention_mask(cu, 5, "cpu")[0, 0]  # (5,5)
    allowed = (m == 0.0)
    print("Allowed-attention matrix (True=can attend):")
    print(allowed.int())

    # Sample 1 (rows 0-2): lower-triangular within [0:3], masked to [3:5]
    expected = torch.tensor([
        [1, 0, 0, 0, 0],  # tok0 attends to tok0
        [1, 1, 0, 0, 0],  # tok1 attends to tok0,1
        [1, 1, 1, 0, 0],  # tok2 attends to tok0,1,2
        [0, 0, 0, 1, 0],  # tok3 (sample2) attends to tok3 only
        [0, 0, 0, 1, 1],  # tok4 attends to tok3,4
    ], dtype=torch.bool)
    assert torch.equal(allowed, expected), f"Mask structure wrong:\n{allowed.int()}"
    print("✓ Block-diagonal causal structure correct")
    print("✓ No cross-sample attention (sample1 cannot see sample2 and vice versa)")


def test_softmax_no_nan():
    print("\n" + "=" * 60)
    print("Test 3: softmax over masked logits produces no NaN")
    print("=" * 60)
    cu = torch.tensor([0, 4, 7])
    total = 7
    m = create_packed_attention_mask(cu, total, "cpu", dtype=torch.float32)[0, 0]
    # Simulate attention scores
    torch.manual_seed(0)
    scores = torch.randn(total, total)
    masked = scores + m
    probs = torch.softmax(masked, dim=-1)
    assert not torch.isnan(probs).any(), "Softmax produced NaN!"
    # Each row should sum to 1
    row_sums = probs.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones(total), atol=1e-5), f"Rows don't sum to 1: {row_sums}"
    # Verify sample1 token (row 0) puts ZERO prob on sample2 tokens (cols 4,5,6)
    assert probs[0, 4:].abs().sum() < 1e-6, "Leakage: sample1 attends to sample2!"
    print("✓ Softmax stable, rows sum to 1, no cross-sample leakage")


def test_batch1_reshape():
    print("\n" + "=" * 60)
    print("Test 4: packed-as-batch1 reshape logic")
    print("=" * 60)
    total_len, hidden = 7, 16
    inputs_embeds = torch.randn(total_len, hidden)  # packed 2D
    position_ids = torch.tensor([0, 1, 2, 3, 0, 1, 2])  # 1D
    labels = torch.tensor([-100, -100, 10, 20, -100, 30, 40])  # 1D

    # Apply the model's reshape
    ie = inputs_embeds.unsqueeze(0)
    pid = position_ids.unsqueeze(0)
    lab = labels.unsqueeze(0)
    assert ie.shape == (1, total_len, hidden), f"embeds shape {ie.shape}"
    assert pid.shape == (1, total_len), f"pos shape {pid.shape}"
    assert lab.shape == (1, total_len), f"labels shape {lab.shape}"

    # Simulate logits out, then squeeze back
    logits = torch.randn(1, total_len, 100)
    sq = logits.squeeze(0)
    assert sq.shape == (total_len, 100), f"squeezed logits {sq.shape}"
    print("✓ (total_len,H)->(1,total_len,H), logits (1,T,V)->(T,V) round-trips correctly")


def test_boundary_loss_shift():
    print("\n" + "=" * 60)
    print("Test 5: causal-shift loss boundary is masked by -100")
    print("=" * 60)
    import torch.nn.functional as F
    # Packed labels: sample1=[−100,−100,10], sample2=[−100,20]
    labels = torch.tensor([-100, -100, 10, -100, 20])
    # Global shift (what HF does internally / training loop bucketing)
    shift_labels = F.pad(labels, (0, 1), value=-100)[1:]
    print(f"labels:       {labels.tolist()}")
    print(f"shift_labels: {shift_labels.tolist()}")
    # Position 2 is last token of sample1; its shifted target = labels[3] = -100 (sample2 start)
    assert shift_labels[2].item() == -100, "Boundary token must shift to -100 (no contamination)"
    print("✓ Sample-boundary shift target is -100 -> no cross-sample loss contamination")


if __name__ == "__main__":
    test_no_nan()
    test_block_diagonal_causal()
    test_softmax_no_nan()
    test_batch1_reshape()
    test_boundary_loss_shift()
    print("\n" + "=" * 60)
    print("🎉 All packed-mask correctness tests passed!")
    print("=" * 60)
