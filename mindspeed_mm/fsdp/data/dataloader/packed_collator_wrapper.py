"""Packed format collator wrapper for MindSpeed audio training.

Wraps MultiModalDataCollatorForSeq2Seq to convert padded batches to pack format,
eliminating padding waste. The wrapper:
  1. Calls the base collator (pad format output)
  2. Strips padding tokens
  3. Concatenates samples into a single sequence
  4. Generates cu_seqlens (cumulative lengths) and position_ids (per-sample restarts)
"""

import torch
from typing import Dict, List, Any


class PackedCollatorWrapper:
    """Wrapper that converts pad-format collator output to pack format."""

    def __init__(self, base_collator, tokenizer):
        """
        Args:
            base_collator: Original MultiModalDataCollatorForSeq2Seq instance
            tokenizer: Tokenizer with pad_token_id
        """
        self.base_collator = base_collator
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Convert features to pack format.

        Args:
            features: List of dicts with input_ids, labels, images, audios, etc.

        Returns:
            Pack format batch:
              - input_ids: (total_len,) concatenated sequence
              - position_ids: (total_len,) per-sample position counters
              - cu_seqlens: (batch_size+1,) cumulative lengths [0, len1, len1+len2, ...]
              - labels: (total_len,) concatenated
              - input_features: audio features (unchanged from base collator)
              - feature_attention_mask: audio attention mask (unchanged)
        """
        # Step 1: Get pad-format batch from base collator
        padded_batch = self.base_collator(features)

        # Step 2: Convert to pack format
        return self._pad_to_pack(padded_batch)

    def _pad_to_pack(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Convert padded batch to pack format.

        Note on position_ids shape:
          The base collator (MultiModalDataCollatorForSeq2Seq) emits position_ids of
          shape (3, batch_size, seq_len) for Qwen3.5/qwen3_omni MRoPE — three axes for
          Time/Height/Width. The collator validation in collator.py:220-222 enforces
          dim()==3. For audio-only training (no images/videos), all three axes hold
          identical position values, but the 3D shape contract still applies.
        """
        input_ids = batch["input_ids"]                 # (batch_size, max_len)
        labels = batch.get("labels")                   # (batch_size, max_len) or None
        position_ids = batch.get("position_ids")       # (3, batch_size, max_len) for MRoPE

        batch_size, max_len = input_ids.shape
        device = input_ids.device

        # Find actual lengths (before padding) for each sample
        if self.pad_token_id is not None:
            non_pad_mask = input_ids != self.pad_token_id
            sample_lens = non_pad_mask.long().sum(dim=1)  # (batch_size,)
        else:
            sample_lens = torch.full((batch_size,), max_len, device=device, dtype=torch.long)

        # Validate position_ids shape (must be 3D for MRoPE Qwen3.5 family)
        is_mrope_3d = position_ids is not None and position_ids.dim() == 3
        if position_ids is not None and not is_mrope_3d:
            # Treat 2D as fallback (e.g. non-MRoPE models): expand to 3D so we have one code path.
            if position_ids.dim() == 2:
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).contiguous()
                is_mrope_3d = True

        # Pack sequences: strip padding, concatenate, restart position_ids per sample.
        packed_input_ids = []
        packed_labels = []
        # For MRoPE we accumulate per-axis position lists separately and cat at the end.
        packed_pos_per_axis = [[], [], []]
        cu_seqlens = [0]

        for i in range(batch_size):
            seq_len = sample_lens[i].item()
            if seq_len == 0:
                continue  # Skip empty (shouldn't happen)

            packed_input_ids.append(input_ids[i, :seq_len])
            if labels is not None:
                packed_labels.append(labels[i, :seq_len])

            # Per-sample position_ids restarting at 0 — REQUIRED for transformers to
            # detect packing via _is_packed_sequence and route to FA2 varlen.
            # We override base collator's monotonically-increasing positions because
            # those don't reset across samples and FA2 won't recognize boundaries.
            sample_positions = torch.arange(seq_len, device=device, dtype=torch.long)
            for ax in range(3):
                packed_pos_per_axis[ax].append(sample_positions)

            cu_seqlens.append(cu_seqlens[-1] + seq_len)

        # Stack: input_ids/labels get leading batch dim (1, total_len) so framework
        # loss_func operates on dim=1 as it expects for 2D labels.
        # position_ids gets shape (3, 1, total_len) per MRoPE 3D contract.
        packed_input_ids_t = torch.cat(packed_input_ids, dim=0).unsqueeze(0)         # (1, total_len)
        per_axis_concat = [torch.cat(packed_pos_per_axis[ax], dim=0) for ax in range(3)]
        position_ids_3d = torch.stack(per_axis_concat, dim=0).unsqueeze(1)            # (3, 1, total_len)

        result = {
            "input_ids": packed_input_ids_t,
            "position_ids": position_ids_3d,
            "cu_seqlens": torch.tensor(cu_seqlens, device=device, dtype=torch.long),  # (batch_size+1,)
        }

        if labels is not None and packed_labels:
            result["labels"] = torch.cat(packed_labels, dim=0).unsqueeze(0)            # (1, total_len)

        # Copy over audio-related fields unchanged (already in correct format from base collator)
        for key in ["input_features", "feature_attention_mask"]:
            if key in batch:
                result[key] = batch[key]

        # Copy over any other fields that don't need packing conversion
        for key in batch:
            if key not in result and key not in ["input_ids", "labels", "position_ids", "attention_mask"]:
                # Skip attention_mask for pack format (must be None to trigger FA2 varlen)
                result[key] = batch[key]

        return result
