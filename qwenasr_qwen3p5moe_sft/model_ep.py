#!/usr/bin/env python3
"""EP (Expert Parallel) model for Qwen3.5-35B-A3B MoE.

Key insight: Qwen3_5MoeExperts stores experts as fused 3D tensors:
  gate_up_proj: (num_experts, 2*intermediate, hidden)
  down_proj: (num_experts, hidden, intermediate)

EP slices these tensors along dim=0 so each rank holds only local experts.
All-to-all communication dispatches tokens to the rank owning the target expert.
"""
import os
import re
import json
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from safetensors.torch import load_file
from safetensors import safe_open
from transformers import AutoConfig
from transformers.activations import ACT2FN
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoder,
)
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoderConfig,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """LoRA adapter wrapping a frozen Linear layer.

    output = frozen_linear(x) + (B @ A)(x) * (alpha / rank)
    """

    def __init__(self, original_linear, rank, alpha=16, dropout=0.0):
        super().__init__()
        self.original = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # Freeze original weights
        for param in self.original.parameters():
            param.requires_grad = False

        # LoRA low-rank matrices
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        base_out = self.original(x)
        lora_out = F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B) * self.scaling
        return base_out + lora_out


def apply_lora_to_model(model, rank, alpha=16, dropout=0.0):
    """Inject LoRA adapters into attention layers of the LLM.

    Targets:
      - self_attn: q_proj, k_proj, v_proj, o_proj
      - linear_attn: in_proj_qkv, out_proj, in_proj_z

    Returns list of (name, LoRALinear) for logging.
    """
    # Target module names (last component of the dotted path)
    self_attn_targets = {'q_proj', 'k_proj', 'v_proj', 'o_proj'}
    linear_attn_targets = {'in_proj_qkv', 'out_proj', 'in_proj_z'}

    lora_modules = []

    for layer in model.llm.model.layers:
        # self_attn layers
        if hasattr(layer, 'self_attn'):
            attn = layer.self_attn
            for target_name in self_attn_targets:
                if hasattr(attn, target_name):
                    original = getattr(attn, target_name)
                    if isinstance(original, nn.Linear):
                        lora_layer = LoRALinear(original, rank, alpha, dropout)
                        lora_layer = lora_layer.to(dtype=original.weight.dtype, device=original.weight.device)
                        setattr(attn, target_name, lora_layer)
                        lora_modules.append((f'self_attn.{target_name}', lora_layer))

        # linear_attn layers
        if hasattr(layer, 'linear_attn'):
            attn = layer.linear_attn
            for target_name in linear_attn_targets:
                if hasattr(attn, target_name):
                    original = getattr(attn, target_name)
                    if isinstance(original, nn.Linear):
                        lora_layer = LoRALinear(original, rank, alpha, dropout)
                        lora_layer = lora_layer.to(dtype=original.weight.dtype, device=original.weight.device)
                        setattr(attn, target_name, lora_layer)
                        lora_modules.append((f'linear_attn.{target_name}', lora_layer))

    # Freeze all non-LoRA LLM parameters
    for name, param in model.llm.named_parameters():
        if 'lora_A' not in name and 'lora_B' not in name:
            param.requires_grad = False

    lora_param_count = sum(
        p.numel() for p in model.llm.parameters() if p.requires_grad
    )
    logger.info(f"LoRA applied: {len(lora_modules)} adapters, {lora_param_count/1e6:.2f}M trainable params (rank={rank}, alpha={alpha})")

    return lora_modules


# ---------------------------------------------------------------------------
# EP Experts: holds only local shard of the fused 3D expert tensors
# ---------------------------------------------------------------------------
class EPExperts(nn.Module):
    """Local expert shard. Same forward logic as Qwen3_5MoeExperts but
    operates on a slice of (num_local_experts, ...) tensors."""

    def __init__(self, gate_up_proj, down_proj, act_fn, num_experts, num_local_experts, ep_rank):
        super().__init__()
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.ep_rank = ep_rank
        self.expert_start = ep_rank * num_local_experts
        # These are already sliced to (num_local_experts, ...)
        self.gate_up_proj = nn.Parameter(gate_up_proj)
        self.down_proj = nn.Parameter(down_proj)
        self.act_fn = act_fn

    def forward(self, hidden_states, top_k_index, top_k_weights):
        """
        hidden_states: (num_tokens, hidden_dim)
        top_k_index: (num_tokens, top_k) - LOCAL expert indices (0..num_local_experts-1)
        top_k_weights: (num_tokens, top_k)
        """
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_local_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx >= self.num_local_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = F.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


# ---------------------------------------------------------------------------
# EP Sparse MoE Block: replaces Qwen3_5MoeSparseMoeBlock
# ---------------------------------------------------------------------------
class EPSparseMoeBlock(nn.Module):
    """MoE block with Expert Parallelism via all-to-all."""

    def __init__(self, gate, shared_expert, shared_expert_gate, local_experts,
                 ep_group, ep_size, ep_rank, num_experts, num_local_experts):
        super().__init__()
        self.gate = gate
        self.shared_expert = shared_expert
        self.shared_expert_gate = shared_expert_gate
        self.local_experts = local_experts
        self.ep_group = ep_group
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts

    def forward(self, hidden_states):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        h = hidden_states.reshape(-1, hidden_dim)
        num_tokens = h.shape[0]

        # 1. Shared expert (all ranks compute, no communication needed)
        shared_out = F.sigmoid(self.shared_expert_gate(h)) * self.shared_expert(h)

        # 2. Router (all ranks compute full routing)
        _, routing_weights, selected_experts = self.gate(h)
        # routing_weights: (num_tokens, top_k)
        # selected_experts: (num_tokens, top_k) - global expert indices 0..255

        # 3. EP dispatch + local compute + combine
        expert_out = self._ep_forward(h, selected_experts, routing_weights)

        # 4. Combine
        output = expert_out + shared_out
        return output.reshape(batch_size, seq_len, hidden_dim)

    def _ep_forward(self, hidden_states, selected_experts, routing_weights):
        """Optimized EP forward: vectorized dispatch, merged all-to-all, batched experts."""
        num_tokens, hidden_dim = hidden_states.shape
        top_k = selected_experts.shape[1]
        device = hidden_states.device

        # --- Flatten (num_tokens, top_k) -> (N,) ---
        flat_experts = selected_experts.reshape(-1)
        flat_weights = routing_weights.reshape(-1)
        flat_token_idx = torch.arange(num_tokens, device=device).unsqueeze(1).expand(-1, top_k).reshape(-1)

        expert_to_rank = flat_experts // self.num_local_experts
        local_expert_idx = flat_experts % self.num_local_experts
        N = flat_experts.shape[0]

        # --- Vectorized dispatch: sort by target rank (float32 for AiCore) ---
        sort_idx = torch.argsort(expert_to_rank.float(), stable=True)
        send_counts = torch.bincount(expert_to_rank, minlength=self.ep_size)
        send_splits = send_counts.tolist()

        send_h = hidden_states[flat_token_idx[sort_idx]]
        send_eidx = local_expert_idx[sort_idx]
        send_w = flat_weights[sort_idx]
        send_tidx = flat_token_idx[sort_idx]

        # --- Exchange counts ---
        recv_counts = torch.zeros_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts, group=self.ep_group)
        recv_splits = recv_counts.tolist()
        total_recv = recv_counts.sum().item()

        # --- Merged all-to-all: pack (hidden + expert_idx + weight) into one buffer ---
        pack_dim = hidden_dim + 2
        send_packed = torch.empty(N, pack_dim, dtype=hidden_states.dtype, device=device)
        send_packed[:, :hidden_dim] = send_h
        send_packed[:, hidden_dim] = send_eidx.to(hidden_states.dtype)
        send_packed[:, hidden_dim + 1] = send_w

        recv_packed = torch.empty(total_recv, pack_dim, dtype=hidden_states.dtype, device=device)

        if N > 0 or total_recv > 0:
            dist.all_to_all_single(recv_packed, send_packed,
                                   output_split_sizes=recv_splits,
                                   input_split_sizes=send_splits,
                                   group=self.ep_group)

        # --- Unpack ---
        recv_h = recv_packed[:, :hidden_dim]
        recv_eidx = recv_packed[:, hidden_dim].long()
        recv_w = recv_packed[:, hidden_dim + 1]

        # --- Local expert computation ---
        if total_recv > 0:
            local_out = self._local_expert_forward(recv_h, recv_eidx, recv_w)
        else:
            local_out = recv_h.new_empty(0, hidden_dim)

        # --- All-to-all back ---
        result_recv = torch.empty(N, hidden_dim, dtype=hidden_states.dtype, device=device)
        if N > 0 or total_recv > 0:
            dist.all_to_all_single(result_recv, local_out,
                                   output_split_sizes=send_splits,
                                   input_split_sizes=recv_splits,
                                   group=self.ep_group)

        # --- Vectorized scatter back ---
        final_output = torch.zeros(num_tokens, hidden_dim, dtype=hidden_states.dtype, device=device)
        final_output.index_add_(0, send_tidx, result_recv)

        return final_output

    def _local_expert_forward(self, hidden_states, expert_indices, weights):
        """Fully batched expert computation using torch.bmm — no Python for-loop."""
        num_tokens, hidden_dim = hidden_states.shape
        E = self.num_local_experts

        # Sort by expert (float32 argsort for AiCore)
        sort_idx = torch.argsort(expert_indices.float())
        sorted_h = hidden_states[sort_idx]
        sorted_w = weights[sort_idx]
        sorted_e = expert_indices[sort_idx]

        counts = torch.bincount(sorted_e, minlength=E)
        max_count = counts.max().item()

        if max_count == 0:
            return torch.zeros_like(hidden_states)

        # Pad each expert's tokens to max_count and stack into (E, max_count, hidden_dim)
        padded = sorted_h.new_zeros(E, max_count, hidden_dim)
        padded_w = sorted_h.new_zeros(E, max_count, 1)
        offset = 0
        for i in range(E):
            c = counts[i].item()
            if c > 0:
                padded[i, :c] = sorted_h[offset:offset + c]
                padded_w[i, :c, 0] = sorted_w[offset:offset + c]
                offset += c

        # Batched gate_up: (E, max_count, hidden) @ (E, hidden, 2*inter) -> (E, max_count, 2*inter)
        gate_up = torch.bmm(padded, self.local_experts.gate_up_proj.transpose(1, 2))
        gate, up = gate_up.chunk(2, dim=-1)
        h = self.local_experts.act_fn(gate) * up

        # Batched down: (E, max_count, inter) @ (E, inter, hidden) -> (E, max_count, hidden)
        down = torch.bmm(h, self.local_experts.down_proj.transpose(1, 2))

        # Apply weights
        down = down * padded_w

        # Unpad and unsort
        output = torch.zeros_like(hidden_states)
        offset = 0
        for i in range(E):
            c = counts[i].item()
            if c > 0:
                output[sort_idx[offset:offset + c]] = down[i, :c]
                offset += c

        return output


# ---------------------------------------------------------------------------
# Weight loading with EP sharding
# ---------------------------------------------------------------------------
def _set_module_tensor(model, key, tensor):
    """Set a tensor in the model by dotted key, handling numeric indices.
    Works with meta-device models by replacing the parameter in-place."""
    parts = key.split('.')
    module = model
    for part in parts[:-1]:
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    param_name = parts[-1]
    old = getattr(module, param_name, None)
    new_param = nn.Parameter(tensor, requires_grad=old.requires_grad if isinstance(old, nn.Parameter) else True)
    setattr(module, param_name, new_param)


def _materialize_meta_tensors(model, device, dtype=torch.bfloat16):
    """Replace any remaining meta tensors with real empty tensors on device."""
    for name, param in list(model.named_parameters()):
        if param.device == torch.device('meta'):
            parts = name.split('.')
            module = model
            for part in parts[:-1]:
                if part.isdigit():
                    module = module[int(part)]
                else:
                    module = getattr(module, part)
            setattr(module, parts[-1],
                    nn.Parameter(torch.zeros(param.shape, dtype=dtype, device=device),
                                 requires_grad=param.requires_grad))

    for name, buf in list(model.named_buffers()):
        if buf.device == torch.device('meta'):
            parts = name.split('.')
            module = model
            for part in parts[:-1]:
                if part.isdigit():
                    module = module[int(part)]
                else:
                    module = getattr(module, part)
            setattr(module, parts[-1],
                    torch.zeros(buf.shape, dtype=dtype, device=device))


def load_llm_with_ep(model_path, ep_size, ep_rank, device):
    """Load LLM with EP sharding: only local experts loaded.

    Strategy: create model on meta device, load each safetensor KEY individually
    (not full shard) to avoid full expert tensors staying in memory.
    """
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.use_cache = False

    num_experts = config.num_experts
    num_local_experts = num_experts // ep_size
    expert_start = ep_rank * num_local_experts
    expert_end = expert_start + num_local_experts

    logger.info(f"Rank {ep_rank}: Loading LLM with EP, local experts [{expert_start}, {expert_end})")

    # Create model on meta device
    from transformers import Qwen3_5MoeForCausalLM
    with torch.device('meta'):
        model = Qwen3_5MoeForCausalLM(config)

    # Build a map: key -> shard_file
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)

    # Group keys by shard file
    shard_to_keys = {}
    for key, shard_file in index["weight_map"].items():
        shard_to_keys.setdefault(shard_file, []).append(key)

    # Get unique shard files
    shard_files = sorted(set(index["weight_map"].values()))

    loaded_count = 0
    for shard_file in shard_files:
        shard_path = os.path.join(model_path, shard_file)

        # Use safe_open to read actual keys from file and load one tensor at a time
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            actual_keys = list(f.keys())
            logger.info(f"Rank {ep_rank}: Loading shard {shard_file} ({len(actual_keys)} tensors)")

            for key in actual_keys:
                tensor = f.get_tensor(key)

                # Remap key: safetensors has "model.language_model.X" but model expects "model.X"
                model_key = key.replace("model.language_model.", "model.")

                # Skip keys that don't exist in our model (visual, mtp, etc.)
                try:
                    if "experts.gate_up_proj" in key:
                        sliced = tensor[expert_start:expert_end].contiguous().clone()
                        _set_module_tensor(model, model_key, sliced.to(dtype=torch.bfloat16, device=device))
                        del sliced, tensor
                    elif "experts.down_proj" in key:
                        sliced = tensor[expert_start:expert_end].contiguous().clone()
                        _set_module_tensor(model, model_key, sliced.to(dtype=torch.bfloat16, device=device))
                        del sliced, tensor
                    else:
                        _set_module_tensor(model, model_key, tensor.to(dtype=torch.bfloat16, device=device))
                        del tensor
                    loaded_count += 1
                except (AttributeError, IndexError) as e:
                    # Skip weights for components not in our model (visual, mtp, etc.)
                    del tensor
                    continue

        import gc; gc.collect()

    # Materialize remaining meta tensors as small zeros (visual/mtp/unused)
    meta_count = 0
    for name, param in list(model.named_parameters()):
        if param.device == torch.device('meta'):
            meta_count += 1
            parts = name.split('.')
            module = model
            for part in parts[:-1]:
                if part.isdigit():
                    module = module[int(part)]
                else:
                    module = getattr(module, part)
            setattr(module, parts[-1],
                    nn.Parameter(torch.zeros(param.shape, dtype=torch.bfloat16, device=device),
                                 requires_grad=False))

    for name, buf in list(model.named_buffers()):
        if buf.device == torch.device('meta'):
            parts = name.split('.')
            module = model
            for part in parts[:-1]:
                if part.isdigit():
                    module = module[int(part)]
                else:
                    module = getattr(module, part)
            setattr(module, parts[-1],
                    torch.zeros(buf.shape, dtype=torch.bfloat16, device=device))

    logger.info(f"Rank {ep_rank}: Loaded {loaded_count} keys, materialized {meta_count} remaining meta params")

    # Ensure all params are bf16 consistently
    for name, param in model.named_parameters():
        if param.dtype != torch.bfloat16 and param.is_floating_point():
            parts = name.split('.')
            module = model
            for part in parts[:-1]:
                if part.isdigit():
                    module = module[int(part)]
                else:
                    module = getattr(module, part)
            setattr(module, parts[-1],
                    nn.Parameter(param.data.to(torch.bfloat16), requires_grad=param.requires_grad))

    for name, buf in model.named_buffers():
        if buf.dtype != torch.bfloat16 and buf.is_floating_point():
            parts = name.split('.')
            module = model
            for part in parts[:-1]:
                if part.isdigit():
                    module = module[int(part)]
                else:
                    module = getattr(module, part)
            setattr(module, parts[-1], buf.to(torch.bfloat16))

    # Log actual memory usage
    if hasattr(torch.npu, 'memory_allocated'):
        mem_gb = torch.npu.memory_allocated(device) / (1024**3)
        logger.info(f"Rank {ep_rank}: NPU memory allocated: {mem_gb:.2f} GB")
    elif hasattr(torch.cuda, 'memory_allocated'):
        mem_gb = torch.cuda.memory_allocated(device) / (1024**3)
        logger.info(f"Rank {ep_rank}: GPU memory allocated: {mem_gb:.2f} GB")

    # Update config to reflect local expert count
    model._ep_num_local_experts = num_local_experts
    model._ep_expert_start = expert_start

    return model, config


# ---------------------------------------------------------------------------
# EPSpeechTranslationModel
# ---------------------------------------------------------------------------
class EPSpeechTranslationModel(nn.Module):
    """Speech Translation with EP for the LLM's MoE layers."""

    supports_gradient_checkpointing = True

    def __init__(self, llm_path, asr_path, audio_token_id, ep_size, ep_rank, ep_group, device):
        super().__init__()
        self.audio_token_id = audio_token_id
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.ep_group = ep_group

        # ---- Audio encoder (full, all ranks identical) --------------------
        asr_config_path = os.path.join(asr_path, "config.json")
        with open(asr_config_path) as f:
            asr_config = json.load(f)

        audio_config = Qwen3OmniMoeAudioEncoderConfig(
            **asr_config["thinker_config"]["audio_config"]
        )
        audio_config._attn_implementation = 'flash_attention_2'  # NPU flash attention with cu_seqlens varlen support
        self.audio_encoder = Qwen3OmniMoeAudioEncoder(audio_config)
        self._load_audio_weights(asr_path)
        self._freeze_audio_encoder()
        self.audio_encoder = self.audio_encoder.to(dtype=torch.bfloat16, device=device)
        logger.info(
            f"Rank {ep_rank}: Audio encoder attention implementation = "
            f"{self.audio_encoder.config._attn_implementation}"
        )

        # ---- LLM with EP -------------------------------------------------
        logger.info(f"Rank {ep_rank}: Loading LLM with EP...")
        self.llm, self.llm_config = load_llm_with_ep(llm_path, ep_size, ep_rank, device)

        num_local_experts = self.llm_config.num_experts // ep_size
        act_fn = ACT2FN[self.llm_config.hidden_act]

        # Replace each MoE block with EP version
        for layer_idx in range(self.llm_config.num_hidden_layers):
            layer = self.llm.model.layers[layer_idx]
            old_block = layer.mlp

            local_experts = EPExperts(
                gate_up_proj=old_block.experts.gate_up_proj.data,
                down_proj=old_block.experts.down_proj.data,
                act_fn=act_fn,
                num_experts=self.llm_config.num_experts,
                num_local_experts=num_local_experts,
                ep_rank=ep_rank,
            )

            ep_block = EPSparseMoeBlock(
                gate=old_block.gate,
                shared_expert=old_block.shared_expert,
                shared_expert_gate=old_block.shared_expert_gate,
                local_experts=local_experts,
                ep_group=ep_group,
                ep_size=ep_size,
                ep_rank=ep_rank,
                num_experts=self.llm_config.num_experts,
                num_local_experts=num_local_experts,
            )

            layer.mlp = ep_block

        # Clean up old expert references
        del old_block

        self.config = self.llm_config
        logger.info(f"Rank {ep_rank}: Model initialized with EP={ep_size}")

    def _load_audio_weights(self, asr_path):
        index_path = os.path.join(asr_path, "model.safetensors.index.json")
        with open(index_path) as f:
            index = json.load(f)

        audio_state = {}
        for shard_file in set(index["weight_map"].values()):
            shard_path = os.path.join(asr_path, shard_file)
            shard_state = load_file(shard_path)
            for k, v in shard_state.items():
                if k.startswith("thinker.audio_tower."):
                    new_key = k.replace("thinker.audio_tower.", "")
                    audio_state[new_key] = v
        self.audio_encoder.load_state_dict(audio_state, strict=True)
        logger.info(f"Loaded {len(audio_state)} audio encoder parameters")

    def _freeze_audio_encoder(self):
        for name, param in self.audio_encoder.named_parameters():
            if any(x in name for x in ["conv2d", "conv_out", "layers.", "positional_embedding"]):
                param.requires_grad = False

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def forward(self, input_ids, labels, input_features, feature_lens,
                position_ids=None, cu_seqlens=None, attention_mask=None, **kwargs):
        """
        Forward pass supporting both Pack and Pad formats.

        Pack format (recommended):
          - input_ids: (total_len,) packed sequence
          - position_ids: (total_len,) independent position IDs per sample
          - cu_seqlens: (batch_size+1,) cumulative sequence lengths
          - attention_mask: None (implicitly defined by cu_seqlens)

        Pad format (legacy):
          - input_ids: (batch_size, max_len) padded sequence
          - attention_mask: (batch_size, max_len)
          - position_ids: None (auto-generated)
          - cu_seqlens: None
        """
        device = input_ids.device

        # Auto-detect format
        is_packed = cu_seqlens is not None

        if is_packed:
            # Pack format: input_ids is 1D
            assert input_ids.dim() == 1, f"Pack format requires 1D input_ids, got {input_ids.dim()}D"
            batch_size = len(cu_seqlens) - 1
            total_len = input_ids.shape[0]
        else:
            # Pad format: input_ids is 2D
            assert input_ids.dim() == 2, f"Pad format requires 2D input_ids, got {input_ids.dim()}D"
            assert attention_mask is not None, "Pad format requires attention_mask"
            batch_size = input_ids.shape[0]

        # 1. Audio encode (format-agnostic, already using varlen)
        input_features = input_features.to(device=device, dtype=torch.bfloat16)
        feature_lens = feature_lens.to(device)
        total_feature_len = int(feature_lens.sum().item())

        if input_features.shape[1] != total_feature_len:
            raise ValueError(
                f"Concatenated audio length mismatch: input_features has {input_features.shape[1]} frames, "
                f"but feature_lens sum to {total_feature_len}"
            )

        if self.audio_encoder.config._attn_implementation == 'flash_attention_2':
            # FA2: concat真实长度, cu_seqlens保证样本间无信息泄露
            # input_features shape: (128, total_len), feature_lens: (batch_size,)
            audio_outputs = self.audio_encoder(input_features, feature_lens=feature_lens)
            audio_embeds = audio_outputs.last_hidden_state.to(dtype=torch.bfloat16)
        else:
            # SDPA/eager: cu_seqlens不生效, 必须逐样本编码避免信息泄露
            # input_features shape: (128, total_len), 需要按feature_lens拆分
            audio_embeds_list = []
            offset = 0
            for b in range(batch_size):
                sample_len = feature_lens[b].item()
                sample_features = input_features[:, offset:offset + sample_len]  # (128, sample_len)
                sample_outputs = self.audio_encoder(
                    sample_features,
                    feature_lens=feature_lens[b:b+1],
                )
                audio_embeds_list.append(sample_outputs.last_hidden_state.to(dtype=torch.bfloat16))
                offset += sample_len
            audio_embeds = torch.cat(audio_embeds_list, dim=0)

        # 2. Text embeddings
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        # Pack: (total_len, hidden_dim), Pad: (batch, seq_len, hidden_dim)

        # Check dimension mismatch
        if audio_embeds.shape[-1] != inputs_embeds.shape[-1]:
            raise ValueError(f"Dimension mismatch: audio_embeds {audio_embeds.shape[-1]} vs text_embeds {inputs_embeds.shape[-1]}")

        # 3. Replace audio tokens with audio embeddings
        if is_packed:
            inputs_embeds = self._replace_audio_tokens_packed(
                inputs_embeds, audio_embeds, input_ids, cu_seqlens
            )
        else:
            inputs_embeds = self._replace_audio_tokens_padded(
                inputs_embeds, audio_embeds, input_ids, batch_size
            )

        # Ensure bf16
        inputs_embeds = inputs_embeds.to(dtype=torch.bfloat16)

        # 4. LLM forward (with EP MoE blocks)
        # Note: For now, we don't patch Qwen3 attention to use cu_seqlens directly.
        # Instead, when in pack format, we generate an attention_mask from cu_seqlens.
        # This still saves memory (no padding in tokens) and most compute (tighter packing),
        # but attention still uses regular mask-based FA2 instead of varlen mode.
        # TODO: Implement varlen attention patch for full optimization.

        if is_packed and attention_mask is None:
            # Generate causal attention mask for pack format
            # Shape: (batch_size, 1, max_seq_len, max_seq_len) or simplified (total_len, total_len)
            # For simplicity, we create a block-diagonal causal mask
            attention_mask = self._create_packed_attention_mask(cu_seqlens, total_len, device)

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        return outputs

    def _create_packed_attention_mask(self, cu_seqlens, total_len, device):
        """Create block-diagonal causal attention mask for packed sequences.

        Each sample can only attend to tokens within its own sequence (cu_seqlens boundaries),
        and must respect causal masking (can't attend to future tokens).

        Returns:
            attention_mask: (total_len, total_len) or (1, 1, total_len, total_len)
            where mask[i,j] = 0 means position i can attend to position j
        """
        batch_size = len(cu_seqlens) - 1
        # Create a full mask (total_len, total_len), initially all -inf (cannot attend)
        mask = torch.full((total_len, total_len), float('-inf'), device=device, dtype=torch.bfloat16)

        # Fill in each sample's block with causal mask
        for b in range(batch_size):
            start = cu_seqlens[b].item()
            end = cu_seqlens[b+1].item()
            seq_len = end - start

            # Create causal mask for this sample: lower triangular (can attend to past)
            # torch.tril creates a lower triangular matrix of 1s
            causal_block = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bfloat16))
            # Convert 1 -> 0 (can attend), 0 -> -inf (cannot attend)
            causal_block = (1.0 - causal_block) * float('-inf')

            # Place into the full mask at positions [start:end, start:end]
            mask[start:end, start:end] = causal_block

        # Transformers expects 4D mask: (batch_size, 1, seq_len, seq_len)
        # But with packed format, we have (total_len,) not (batch_size, seq_len)
        # So we use (1, 1, total_len, total_len) as a broadcast-compatible shape
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, total_len, total_len)

        return mask

    def _replace_audio_tokens_packed(self, inputs_embeds, audio_embeds, input_ids, cu_seqlens):
        """Replace audio tokens in pack format (按cu_seqlens边界逐样本替换)."""
        batch_size = len(cu_seqlens) - 1
        audio_offset = 0

        for b in range(batch_size):
            start_idx = cu_seqlens[b].item()
            end_idx = cu_seqlens[b+1].item()
            sample_ids = input_ids[start_idx:end_idx]

            # Find audio token positions within this sample (relative to start_idx)
            audio_positions = (sample_ids == self.audio_token_id).nonzero(as_tuple=True)[0]

            if len(audio_positions) > 0:
                audio_len = len(audio_positions)
                sample_audio_embeds = audio_embeds[audio_offset:audio_offset + audio_len]

                # Verify shape
                if sample_audio_embeds.shape[0] != audio_len:
                    raise ValueError(f"Audio embedding count mismatch: got {sample_audio_embeds.shape[0]}, expected {audio_len}")

                # Global index = start_idx + relative index
                global_positions = start_idx + audio_positions
                inputs_embeds[global_positions] = sample_audio_embeds

                audio_offset += audio_len

        # Verify all audio embeddings consumed
        if audio_offset != audio_embeds.shape[0]:
            raise ValueError(
                f"Unused audio embeddings detected: consumed {audio_offset}, "
                f"encoder produced {audio_embeds.shape[0]}"
            )

        return inputs_embeds

    def _replace_audio_tokens_padded(self, inputs_embeds, audio_embeds, input_ids, batch_size):
        """Replace audio tokens in pad format (原逻辑)."""
        audio_offset = 0

        for b in range(batch_size):
            # Find audio token positions in this sample
            audio_positions = (input_ids[b] == self.audio_token_id).nonzero(as_tuple=True)[0]

            if len(audio_positions) > 0:
                # Get audio embeddings for this sample
                audio_len = len(audio_positions)
                sample_audio_embeds = audio_embeds[audio_offset:audio_offset + audio_len]

                # Verify shapes match
                if sample_audio_embeds.shape[0] != audio_len:
                    raise ValueError(f"Audio embedding count mismatch: got {sample_audio_embeds.shape[0]}, expected {audio_len}")

                # Replace audio tokens
                inputs_embeds[b, audio_positions] = sample_audio_embeds

                audio_offset += audio_len

        if audio_offset != audio_embeds.shape[0]:
            raise ValueError(
                f"Unused audio embeddings detected: consumed {audio_offset}, "
                f"encoder produced {audio_embeds.shape[0]}"
            )

        return inputs_embeds


# ---------------------------------------------------------------------------
# Checkpoint save/load helpers
# ---------------------------------------------------------------------------
def save_ep_checkpoint(model, rank, expert_replica_rank, ep_rank, ep_size, save_dir, tokenizer=None, lora_only=False):
    """Save sharded checkpoint: each rank saves its own expert shard.
    No all_gather needed — avoids OOM on rank0.
    Use merge_ep_checkpoint() offline to reconstruct full model.

    Args:
        lora_only: If True, only save LoRA parameters + audio encoder trainable params.
                   Since LoRA params are identical across EP ranks, only rank0 saves.
    """
    os.makedirs(save_dir, exist_ok=True)

    if lora_only:
        # LoRA mode: only save trainable params (LoRA + audio encoder projections).
        # After DP sync, a single global rank can write the shared state.
        if rank == 0:
            lora_state = {}
            for name, param in model.named_parameters():
                if param.requires_grad:
                    lora_state[name] = param.data.cpu()
            lora_path = os.path.join(save_dir, "lora_weights.pt")
            torch.save(lora_state, lora_path)
            logger.info(f"LoRA checkpoint saved to {lora_path} ({len(lora_state)} params)")
    else:
        # Full mode: only one replica writes each EP shard.
        if expert_replica_rank == 0:
            shard = {}
            for name, param in model.named_parameters():
                shard[name] = param.data.cpu()

            shard_path = os.path.join(save_dir, f"shard_rank{ep_rank}.pt")
            torch.save(shard, shard_path)

    # Global rank 0 also saves tokenizer and metadata
    if rank == 0:
        meta = {
            "ep_size": ep_size,
            "num_shards": ep_size,
            "lora_only": lora_only,
            "world_size": dist.get_world_size() if dist.is_initialized() else 1,
        }
        with open(os.path.join(save_dir, "ep_metadata.json"), "w") as f:
            json.dump(meta, f)
        if tokenizer:
            tokenizer.save_pretrained(save_dir)

    dist.barrier()
    if rank == 0:
        mode = "LoRA" if lora_only else f"full ({ep_size} shards)"
        logger.info(f"Checkpoint saved to {save_dir} [{mode}]")


def load_ep_checkpoint(model, checkpoint_dir, ep_rank):
    """Load EP checkpoint for inference. Each rank loads its own shard."""
    shard_path = os.path.join(checkpoint_dir, f"shard_rank{ep_rank}.pt")

    if not os.path.exists(shard_path):
        raise FileNotFoundError(f"Checkpoint not found: {shard_path}")

    shard = torch.load(shard_path, map_location="cpu")

    # Load weights into model
    model_state = model.state_dict()
    loaded_keys = []
    missing_keys = []

    for name, param in shard.items():
        if name in model_state:
            model_state[name].copy_(param)
            loaded_keys.append(name)
        else:
            missing_keys.append(name)

    logger.info(f"Rank {ep_rank}: Loaded {len(loaded_keys)} parameters from {shard_path}")
    if missing_keys and ep_rank == 0:
        logger.warning(f"Missing keys in checkpoint: {missing_keys[:5]}...")

    return model


def load_lora_checkpoint(model, checkpoint_dir, ep_rank):
    """Load LoRA checkpoint. All ranks load the same LoRA weights.

    Args:
        model: EPSpeechTranslationModel with LoRA adapters already injected
        checkpoint_dir: Directory containing lora_weights.pt
        ep_rank: Current rank (for logging)
    """
    lora_path = os.path.join(checkpoint_dir, "lora_weights.pt")
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRA checkpoint not found: {lora_path}")

    lora_state = torch.load(lora_path, map_location="cpu")

    # Load LoRA weights
    model_state = model.state_dict()
    loaded_keys = []
    missing_keys = []

    for name, param in lora_state.items():
        if name in model_state:
            model_state[name].copy_(param)
            loaded_keys.append(name)
        else:
            missing_keys.append(name)

    logger.info(f"Rank {ep_rank}: Loaded {len(loaded_keys)} LoRA parameters from {lora_path}")
    if missing_keys and ep_rank == 0:
        logger.warning(f"Missing keys in LoRA checkpoint: {missing_keys[:5]}...")

    return model


def merge_ep_checkpoint(save_dir, output_path=None):
    """Offline merge: reconstruct full model from EP shards. Run on CPU.
    Usage: python -c "from model_ep import merge_ep_checkpoint; merge_ep_checkpoint('output_ep')" """
    import json
    with open(os.path.join(save_dir, "ep_metadata.json")) as f:
        meta = json.load(f)
    ep_size = meta["ep_size"]

    merged = {}
    for rank in range(ep_size):
        shard = torch.load(os.path.join(save_dir, f"shard_rank{rank}.pt"), map_location="cpu")
        for name, param in shard.items():
            if 'local_experts.gate_up_proj' in name or 'local_experts.down_proj' in name:
                clean_name = name.replace('local_experts.', 'experts.')
                if clean_name not in merged:
                    merged[clean_name] = []
                merged[clean_name].append(param)
            else:
                # Non-expert params are identical across ranks, take from rank 0
                if rank == 0:
                    merged[name] = param
        del shard

    # Concatenate expert shards along dim=0
    for name in list(merged.keys()):
        if isinstance(merged[name], list):
            merged[name] = torch.cat(merged[name], dim=0)

    out = output_path or os.path.join(save_dir, "model_merged.pt")
    torch.save(merged, out)
    print(f"Merged checkpoint saved to {out} ({len(merged)} keys)")
