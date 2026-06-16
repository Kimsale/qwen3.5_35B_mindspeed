# Copyright 2025 Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Manual EP loader for Qwen3.5 audio training.

This path keeps MindSpeed-MM's existing FSDP2/EP forward path, but bypasses
DCP loading for the Qwen MoE experts. Expert tensors are read from HF
safetensors one tensor at a time and sliced on dim=0 before being copied to
the already-sharded DTensor local storage.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
import torch.distributed as dist
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn
from torch.distributed.tensor import DTensor
from transformers.models.whisper.configuration_whisper import WhisperConfig

from mindspeed.fsdp.utils.log import print_rank
from mindspeed_mm.fsdp.utils.register import model_register

# The local register decorator does not return the decorated class. Import the
# module for registration side effects, then fetch the real class from registry.
from . import modeling_qwen3_5_audio  # noqa: F401
from .projector import AudioProjector

logger = logging.getLogger(__name__)
Qwen3_5AudioForConditionalGeneration = model_register.get("qwen3_5_audio")

_EXPERT_RE = re.compile(
    r"^model\.language_model\.layers\.\d+\.mlp\.experts\."
    r"(gate_up_proj|down_proj)$"
)
_VISUAL_PREFIX = "model.visual."
_MTP_PREFIX = "mtp."
_TIE_MAPPING = {"lm_head.weight": "model.language_model.embed_tokens.weight"}


def _get_cfg_value(cfg, name: str, default=None):
    if cfg is None:
        return default
    return getattr(cfg, name, default)


def _rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def _world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def _is_expert_key(key: str) -> bool:
    return _EXPERT_RE.match(key) is not None


def _target_param_names(hf_key: str) -> Iterable[str]:
    """Yield possible in-model parameter names for a HF safetensors key."""
    yield hf_key
    if hf_key.endswith(".weight"):
        yield hf_key[:-len(".weight")] + ".base_layer.weight"
    if hf_key.endswith(".bias"):
        yield hf_key[:-len(".bias")] + ".base_layer.bias"


def _find_param(param_by_name: Dict[str, nn.Parameter], hf_key: str) -> Optional[nn.Parameter]:
    for name in _target_param_names(hf_key):
        param = param_by_name.get(name)
        if param is not None:
            return param
    return None


def _dtensor_local_slice(full_tensor: torch.Tensor, dtensor: DTensor) -> torch.Tensor:
    """Return the local shard from a full tensor for the DTensor placement."""
    shard = full_tensor
    mesh = dtensor.device_mesh
    coord = mesh.get_coordinate()
    if coord is None:
        coord = [0] * mesh.ndim

    for mesh_dim, placement in enumerate(dtensor.placements):
        if not placement.is_shard():
            continue
        shard_dim = placement.dim
        mesh_size = int(mesh.mesh.shape[mesh_dim])
        mesh_rank = int(coord[mesh_dim])
        shard_size, shard_offset = placement._local_shard_size_on_dim(
            full_tensor.shape[shard_dim],
            mesh_size,
            mesh_rank,
            return_offset=True,
        )
        shard = shard.narrow(shard_dim, shard_offset, shard_size)

    local_shape = tuple(dtensor.to_local().shape)
    if tuple(shard.shape) == local_shape:
        return shard.contiguous()

    padded = torch.zeros(local_shape, dtype=shard.dtype)
    common = tuple(slice(0, min(a, b)) for a, b in zip(local_shape, shard.shape))
    padded[common].copy_(shard[common])
    return padded


def _copy_tensor_to_param(param: nn.Parameter, tensor: torch.Tensor, *, local_tensor: bool = False) -> None:
    """Copy a CPU tensor into a Tensor or DTensor parameter."""
    with torch.no_grad():
        data = param.data
        if isinstance(data, DTensor):
            dst = data.to_local()
            src = tensor.contiguous() if local_tensor else _dtensor_local_slice(tensor, data)
            if tuple(src.shape) != tuple(dst.shape):
                raise ValueError(
                    f"DTensor local shape mismatch: source {tuple(src.shape)} vs target {tuple(dst.shape)}"
                )
            dst.copy_(src.to(device=dst.device, dtype=dst.dtype, non_blocking=True))
        else:
            if tuple(tensor.shape) != tuple(data.shape):
                raise ValueError(
                    f"Tensor shape mismatch: source {tuple(tensor.shape)} vs target {tuple(data.shape)}"
                )
            data.copy_(tensor.to(device=data.device, dtype=data.dtype, non_blocking=True))


def _prepare_expert_tensor(hf_key: str, tensor: torch.Tensor, start: int, end: int) -> torch.Tensor:
    local = tensor[start:end].contiguous().clone()
    # HF stores Qwen3.5 experts as (E, 2I, H) / (E, H, I). MindSpeed grouped
    # matmul expects (E, H, 2I) / (E, I, H).
    if hf_key.endswith("gate_up_proj") or hf_key.endswith("down_proj"):
        local = local.permute(0, 2, 1).contiguous()
    return local


@model_register.register("qwen3_5_audio_manual_ep")
class Qwen3_5AudioManualEPForConditionalGeneration(Qwen3_5AudioForConditionalGeneration):
    """Qwen3.5 audio model with safe_open + dim0 expert slicing loader."""

    def load_manual_ep_weights(self, args) -> None:
        cfg = getattr(args.training, "manual_ep_hf_load", None)
        qwen_hf_dir = _get_cfg_value(cfg, "qwen_hf_dir", None) or args.model.model_name_or_path
        whisper_hf_dir = _get_cfg_value(cfg, "whisper_hf_dir", None) or getattr(args.model, "whisper_path", None)
        load_visual = bool(_get_cfg_value(cfg, "load_visual", False))
        load_audio = bool(_get_cfg_value(cfg, "load_audio", True))
        init_audio_projector = bool(_get_cfg_value(cfg, "init_audio_projector", True))

        ep_size = int(args.parallel.expert_parallel_size)
        ep_rank = _rank()
        if dist.is_available() and dist.is_initialized():
            try:
                from mindspeed_mm.fsdp.distributed.parallel_state import get_parallel_state

                ps = get_parallel_state()
                ep_rank = ps.get_ep_rank()
                ep_size = ps.get_ep_group_size()
            except Exception:
                ep_rank = _rank()
                ep_size = int(args.parallel.expert_parallel_size)

        num_experts = int(self.config.text_config.num_experts)
        if num_experts % ep_size != 0:
            raise ValueError(f"num_experts={num_experts} is not divisible by ep_size={ep_size}")
        local_experts = num_experts // ep_size
        expert_start = ep_rank * local_experts
        expert_end = expert_start + local_experts

        print_rank(
            logger.info,
            f"[manual_ep] loading Qwen from {qwen_hf_dir}; EP rank {ep_rank}/{ep_size}, "
            f"experts [{expert_start}, {expert_end})",
        )
        self._load_qwen_safetensors(qwen_hf_dir, expert_start, expert_end, load_visual=load_visual)

        if load_audio:
            self._load_whisper_safetensors(whisper_hf_dir)
        if init_audio_projector:
            self._init_audio_projector_from_default(whisper_hf_dir)

        self._manual_ep_loaded = True
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        print_rank(logger.info, "[manual_ep] weight loading complete")

    def _load_qwen_safetensors(self, qwen_hf_dir: str, expert_start: int, expert_end: int, *, load_visual: bool) -> None:
        qwen_dir = Path(qwen_hf_dir)
        index_path = qwen_dir / "model.safetensors.index.json"
        with index_path.open("r") as f:
            index = json.load(f)

        shard_to_keys = defaultdict(list)
        for key, shard_file in index["weight_map"].items():
            shard_to_keys[shard_file].append(key)

        param_by_name = dict(self.named_parameters())
        loaded = skipped = missing = 0
        tied_loaded = 0

        for shard_file in sorted(shard_to_keys):
            shard_path = qwen_dir / shard_file
            with safe_open(str(shard_path), framework="pt", device="cpu") as f:
                for hf_key in f.keys():
                    if hf_key.startswith(_MTP_PREFIX):
                        skipped += 1
                        continue
                    if hf_key.startswith(_VISUAL_PREFIX) and not load_visual:
                        skipped += 1
                        continue

                    param = _find_param(param_by_name, hf_key)
                    if param is None:
                        skipped += 1
                        continue

                    tensor = f.get_tensor(hf_key)
                    if _is_expert_key(hf_key):
                        tensor = _prepare_expert_tensor(hf_key, tensor, expert_start, expert_end)
                        _copy_tensor_to_param(param, tensor, local_tensor=True)
                    else:
                        _copy_tensor_to_param(param, tensor)
                        for tied_key, source_key in _TIE_MAPPING.items():
                            if source_key == hf_key:
                                tied_param = _find_param(param_by_name, tied_key)
                                if tied_param is None:
                                    missing += 1
                                else:
                                    _copy_tensor_to_param(tied_param, tensor)
                                    tied_loaded += 1
                    loaded += 1
                    del tensor

            gc.collect()
            if hasattr(torch, "npu"):
                try:
                    torch.npu.empty_cache()
                except Exception:
                    pass

        print_rank(
            logger.info,
            f"[manual_ep] qwen tensors loaded={loaded}, tied_loaded={tied_loaded}, "
            f"skipped={skipped}, missing_tied={missing}",
        )

    def _load_whisper_safetensors(self, whisper_hf_dir: str) -> None:
        if whisper_hf_dir is None:
            raise ValueError("manual_ep_hf_load.whisper_hf_dir or model.whisper_path must be set")

        param_by_name = dict(self.named_parameters())
        loaded = skipped = 0
        for shard_path in sorted(Path(whisper_hf_dir).glob("*.safetensors")):
            state = load_file(str(shard_path), device="cpu")
            for key, tensor in state.items():
                if key.startswith("model.encoder."):
                    target_key = "audio_tower.encoder." + key[len("model.encoder."):]
                elif key.startswith("encoder."):
                    target_key = "audio_tower.encoder." + key[len("encoder."):]
                else:
                    skipped += 1
                    continue
                param = _find_param(param_by_name, target_key)
                if param is None:
                    skipped += 1
                    continue
                _copy_tensor_to_param(param, tensor)
                loaded += 1
            del state
            gc.collect()

        print_rank(logger.info, f"[manual_ep] whisper encoder tensors loaded={loaded}, skipped={skipped}")

    def _init_audio_projector_from_default(self, whisper_hf_dir: str) -> None:
        whisper_config = WhisperConfig.from_pretrained(whisper_hf_dir)
        projector = AudioProjector(
            audio_hidden_size=whisper_config.d_model,
            llm_hidden_size=self.config.text_config.hidden_size,
            projector_hidden_act=getattr(self.config, "audio_projector_act", "gelu"),
        )
        param_by_name = dict(self.named_parameters())
        loaded = 0
        for key, tensor in projector.state_dict().items():
            target_key = f"audio_projector.{key}"
            param = _find_param(param_by_name, target_key)
            if param is None:
                continue
            _copy_tensor_to_param(param, tensor)
            loaded += 1
        print_rank(logger.info, f"[manual_ep] audio projector tensors initialized={loaded}")
