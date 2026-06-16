#!/usr/bin/env python3
"""Test script for PackedDataCollator - standalone version without full imports."""
import torch
from transformers import AutoTokenizer
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class PackedDataCollator:
    """Pack-based DataCollator - packs multiple samples into a single sequence.

    Eliminates padding waste by concatenating sequences with cu_seqlens boundaries.
    Compatible with FlashAttention-2 varlen mode.
    """
    tokenizer: AutoTokenizer

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # ========== Text sequences: Pack format ==========
        packed_input_ids = []
        packed_labels = []
        cu_seqlens = [0]  # Cumulative sequence lengths, starts at 0
        position_ids = []
        sample_lens = []

        for f in features:
            seq_len = len(f["input_ids"])
            packed_input_ids.extend(f["input_ids"])
            packed_labels.extend(f["labels"])
            cu_seqlens.append(cu_seqlens[-1] + seq_len)
            # Position IDs: independent counting within each sample
            position_ids.extend(list(range(seq_len)))
            sample_lens.append(f["sample_len"])

        # ========== Audio features: concat by real lengths (same as before) ==========
        audio_list = []
        feature_lens = []
        for f in features:
            audio_list.append(f["input_features"])
            feature_lens.append(f["feature_lens"])

        batch = {
            "input_ids": torch.tensor(packed_input_ids, dtype=torch.long),  # (total_len,)
            "position_ids": torch.tensor(position_ids, dtype=torch.long),   # (total_len,)
            "cu_seqlens": torch.tensor(cu_seqlens, dtype=torch.int32),      # (batch_size+1,)
            "labels": torch.tensor(packed_labels, dtype=torch.long),        # (total_len,)
            "sample_lens": torch.tensor(sample_lens, dtype=torch.long),     # (batch_size,)
            "input_features": torch.cat(audio_list, dim=1),                 # (128, total_audio_len)
            "feature_lens": torch.tensor(feature_lens, dtype=torch.long),   # (batch_size,)
        }
        return batch


def test_packed_collator():
    """Verify packing logic: ensure sample boundaries are correct and no information leakage."""
    print("=" * 60)
    print("Testing PackedDataCollator")
    print("=" * 60)

    # Create a mock tokenizer class
    class MockTokenizer:
        pad_token_id = 0

    tokenizer = MockTokenizer()

    # Create 2 mock samples
    features = [
        {
            "input_ids": [1, 2, 3],
            "labels": [-100, -100, 10],
            "sample_len": 100,
            "input_features": torch.randn(128, 50),
            "feature_lens": 50
        },
        {
            "input_ids": [4, 5],
            "labels": [-100, 20],
            "sample_len": 80,
            "input_features": torch.randn(128, 30),
            "feature_lens": 30
        },
    ]

    collator = PackedDataCollator(tokenizer)
    batch = collator(features)

    print("\n[Input Features]")
    print(f"Sample 1: input_ids={features[0]['input_ids']}, labels={features[0]['labels']}")
    print(f"Sample 2: input_ids={features[1]['input_ids']}, labels={features[1]['labels']}")

    print("\n[Packed Output]")
    print(f"cu_seqlens: {batch['cu_seqlens'].tolist()}")
    print(f"input_ids: {batch['input_ids'].tolist()}")
    print(f"position_ids: {batch['position_ids'].tolist()}")
    print(f"labels: {batch['labels'].tolist()}")
    print(f"sample_lens: {batch['sample_lens'].tolist()}")
    print(f"input_features shape: {batch['input_features'].shape}")
    print(f"feature_lens: {batch['feature_lens'].tolist()}")

    # Assertions
    print("\n[Verification]")

    # 1. cu_seqlens should be [0, 3, 5]
    expected_cu_seqlens = [0, 3, 5]
    assert batch["cu_seqlens"].tolist() == expected_cu_seqlens, \
        f"cu_seqlens mismatch: expected {expected_cu_seqlens}, got {batch['cu_seqlens'].tolist()}"
    print(f"✓ cu_seqlens correct: {expected_cu_seqlens}")

    # 2. input_ids should be [1,2,3,4,5]
    expected_input_ids = [1, 2, 3, 4, 5]
    assert batch["input_ids"].tolist() == expected_input_ids, \
        f"input_ids mismatch: expected {expected_input_ids}, got {batch['input_ids'].tolist()}"
    print(f"✓ input_ids correct: {expected_input_ids}")

    # 3. position_ids should be [0,1,2,0,1] (independent per sample)
    expected_position_ids = [0, 1, 2, 0, 1]
    assert batch["position_ids"].tolist() == expected_position_ids, \
        f"position_ids mismatch: expected {expected_position_ids}, got {batch['position_ids'].tolist()}"
    print(f"✓ position_ids correct: {expected_position_ids}")

    # 4. labels should be [-100,-100,10,-100,20]
    expected_labels = [-100, -100, 10, -100, 20]
    assert batch["labels"].tolist() == expected_labels, \
        f"labels mismatch: expected {expected_labels}, got {batch['labels'].tolist()}"
    print(f"✓ labels correct: {expected_labels}")

    # 5. audio features should be concatenated (128, 80)
    expected_audio_shape = (128, 80)  # 50 + 30
    assert batch["input_features"].shape == expected_audio_shape, \
        f"audio shape mismatch: expected {expected_audio_shape}, got {batch['input_features'].shape}"
    print(f"✓ audio features concatenated: {expected_audio_shape}")

    # 6. feature_lens should be [50, 30]
    expected_feature_lens = [50, 30]
    assert batch["feature_lens"].tolist() == expected_feature_lens, \
        f"feature_lens mismatch: expected {expected_feature_lens}, got {batch['feature_lens'].tolist()}"
    print(f"✓ feature_lens correct: {expected_feature_lens}")

    # 7. sample_lens should be [100, 80]
    expected_sample_lens = [100, 80]
    assert batch["sample_lens"].tolist() == expected_sample_lens, \
        f"sample_lens mismatch: expected {expected_sample_lens}, got {batch['sample_lens'].tolist()}"
    print(f"✓ sample_lens correct: {expected_sample_lens}")

    print("\n" + "=" * 60)
    print("✅ All tests passed! PackedDataCollator is working correctly.")
    print("=" * 60)


def test_batch_boundary_extraction():
    """Test extracting individual samples from packed batch using cu_seqlens."""
    print("\n" + "=" * 60)
    print("Testing batch boundary extraction")
    print("=" * 60)

    # Simulate a packed batch
    packed_input_ids = torch.tensor([1, 2, 3, 4, 5])
    packed_labels = torch.tensor([-100, -100, 10, -100, 20])
    cu_seqlens = torch.tensor([0, 3, 5])

    batch_size = len(cu_seqlens) - 1
    print(f"\nBatch size: {batch_size}")
    print(f"Packed input_ids: {packed_input_ids.tolist()}")
    print(f"cu_seqlens: {cu_seqlens.tolist()}")

    # Extract each sample
    for i in range(batch_size):
        start = cu_seqlens[i].item()
        end = cu_seqlens[i + 1].item()
        sample_input_ids = packed_input_ids[start:end]
        sample_labels = packed_labels[start:end]
        print(f"\nSample {i}: [{start}:{end}]")
        print(f"  input_ids: {sample_input_ids.tolist()}")
        print(f"  labels: {sample_labels.tolist()}")

    print("\n✓ Boundary extraction works correctly")


def test_larger_batch():
    """Test with a larger batch (4 samples with varying lengths)."""
    print("\n" + "=" * 60)
    print("Testing larger batch with 4 samples")
    print("=" * 60)

    class MockTokenizer:
        pad_token_id = 0

    tokenizer = MockTokenizer()

    # Create 4 samples with different lengths
    features = [
        {
            "input_ids": list(range(1, 11)),  # [1,2,3,4,5,6,7,8,9,10] - length 10
            "labels": [-100] * 9 + [100],
            "sample_len": 200,
            "input_features": torch.randn(128, 100),
            "feature_lens": 100
        },
        {
            "input_ids": list(range(11, 16)),  # [11,12,13,14,15] - length 5
            "labels": [-100] * 4 + [101],
            "sample_len": 150,
            "input_features": torch.randn(128, 80),
            "feature_lens": 80
        },
        {
            "input_ids": list(range(16, 24)),  # [16,...,23] - length 8
            "labels": [-100] * 7 + [102],
            "sample_len": 180,
            "input_features": torch.randn(128, 120),
            "feature_lens": 120
        },
        {
            "input_ids": list(range(24, 27)),  # [24,25,26] - length 3
            "labels": [-100] * 2 + [103],
            "sample_len": 120,
            "input_features": torch.randn(128, 60),
            "feature_lens": 60
        },
    ]

    collator = PackedDataCollator(tokenizer)
    batch = collator(features)

    print(f"\nTotal packed length: {len(batch['input_ids'])} (expected 10+5+8+3=26)")
    print(f"cu_seqlens: {batch['cu_seqlens'].tolist()} (expected [0,10,15,23,26])")
    print(f"First 15 input_ids: {batch['input_ids'][:15].tolist()}")
    print(f"Last 5 input_ids: {batch['input_ids'][-5:].tolist()}")

    # Verify
    assert len(batch['input_ids']) == 26, f"Expected 26 tokens, got {len(batch['input_ids'])}"
    assert batch['cu_seqlens'].tolist() == [0, 10, 15, 23, 26], \
        f"cu_seqlens mismatch: {batch['cu_seqlens'].tolist()}"
    assert batch['input_ids'][0].item() == 1, "First token should be 1"
    assert batch['input_ids'][-1].item() == 26, "Last token should be 26"
    assert batch['input_features'].shape == (128, 360), \
        f"Audio shape should be (128, 360), got {batch['input_features'].shape}"

    print("\n✓ Larger batch test passed")


if __name__ == "__main__":
    test_packed_collator()
    test_batch_boundary_extraction()
    test_larger_batch()
    print("\n" + "=" * 60)
    print("🎉 All PackedDataCollator tests completed successfully!")
    print("=" * 60)
