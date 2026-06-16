import os
import time
from collections import defaultdict
from typing import List, Optional

import torch
import torch.distributed as dist

from mindspeed.fsdp.distributed.dist_ops import all_to_all as _all_to_all
from mindspeed_mm.fsdp.ops.moe_ops.gemm import grouped_matmul
from mindspeed_mm.fsdp.ops.moe_ops.permute import permute
from mindspeed_mm.fsdp.ops.moe_ops.unpermute import unpermute
from mindspeed_mm.fsdp.ops.swiglu import swiglu


_MOE_PHASE_STATE = {
    "calls": 0,
    "window_calls": 0,
    "lines": 0,
    "phase_ms": defaultdict(float),
    "scalar": defaultdict(float),
    "last_input_splits": None,
    "last_output_splits": None,
}


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _phase_sync(enabled: bool):
    if enabled and hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.synchronize()


def _format_int_list(values: List[int]) -> str:
    return "[" + ",".join(str(value) for value in values) + "]"


def _profile_rank_enabled() -> bool:
    ranks = os.getenv("MOE_PHASE_RANKS", "0")
    if not dist.is_available() or not dist.is_initialized():
        rank = 0
    else:
        rank = dist.get_rank()
    return str(rank) in {item.strip() for item in ranks.split(",") if item.strip()}


class _MoePhaseProfiler:
    def __init__(self, fused: bool):
        self.enabled = _env_bool("MOE_PHASE_TIMING", False) and _profile_rank_enabled()
        self.sync = _env_bool("MOE_PHASE_TIMING_SYNC", False)
        self.fused = fused
        self.phase_ms = {}
        self._last = None
        if self.enabled:
            _phase_sync(self.sync)
            self._last = time.perf_counter()

    def mark(self, name: str):
        if not self.enabled:
            return
        _phase_sync(self.sync)
        now = time.perf_counter()
        self.phase_ms[name] = self.phase_ms.get(name, 0.0) + (now - self._last) * 1000.0
        self._last = now

    def finish(
        self,
        input_splits: List,
        output_splits: List,
        num_global_sum_tokens_per_local_expert: torch.Tensor,
    ):
        if not self.enabled:
            return

        state = _MOE_PHASE_STATE
        state["calls"] += 1
        start_call = _env_int("MOE_PHASE_START_CALL", 0)
        if state["calls"] <= start_call:
            return

        state["window_calls"] += 1
        for name, value in self.phase_ms.items():
            state["phase_ms"][name] += value

        counts = num_global_sum_tokens_per_local_expert.detach().float()
        if counts.numel() > 0:
            state["scalar"]["expert_counts_mean"] += counts.mean().item()
            state["scalar"]["expert_counts_max"] += counts.max().item()
            state["scalar"]["expert_counts_std"] += counts.std(unbiased=False).item()
            nonzero = torch.count_nonzero(counts).item()
            state["scalar"]["expert_counts_nonzero"] += float(nonzero)

        state["last_input_splits"] = [int(x) for x in input_splits]
        state["last_output_splits"] = [int(x) for x in output_splits]

        log_every = max(1, _env_int("MOE_PHASE_LOG_EVERY", 80))
        max_lines = _env_int("MOE_PHASE_MAX_LINES", 40)
        if state["window_calls"] < log_every:
            return
        if max_lines >= 0 and state["lines"] >= max_lines:
            state["phase_ms"].clear()
            state["scalar"].clear()
            state["window_calls"] = 0
            return

        calls = state["window_calls"]
        phase_parts = [
            f"{name}_ms={state['phase_ms'][name] / calls:.3f}"
            for name in (
                "dispatch_preprocess",
                "permute_pre_a2a",
                "alltoall_dispatch",
                "permute_post_a2a",
                "gmm_fc1",
                "swiglu",
                "gmm_fc2",
                "unpermute_pre_combine",
                "alltoall_combine",
                "unpermute_final",
            )
            if name in state["phase_ms"]
        ]
        scalar_parts = [
            f"{name}={state['scalar'][name] / calls:.3f}"
            for name in (
                "expert_counts_mean",
                "expert_counts_max",
                "expert_counts_std",
                "expert_counts_nonzero",
            )
            if name in state["scalar"]
        ]
        print(
            "[moe_phase] "
            f"rank={dist.get_rank() if dist.is_available() and dist.is_initialized() else 0} "
            f"call={state['calls']} window_calls={calls} fused={int(self.fused)} "
            + " ".join(phase_parts + scalar_parts)
            + f" input_splits={_format_int_list(state['last_input_splits'])} "
            + f"output_splits={_format_int_list(state['last_output_splits'])}",
            flush=True,
        )
        state["lines"] += 1
        state["phase_ms"].clear()
        state["scalar"].clear()
        state["window_calls"] = 0


def all_to_all(
    input_: torch.Tensor,
    process_group: dist.ProcessGroup,
    scatter_dim: int = 2,
    gather_dim: int = 1,
    scatter_sizes: List = None,
    gather_sizes: List = None
):
    return _all_to_all(process_group, input_, gather_sizes, scatter_sizes)


def ep_forward(
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
    ep_group: Optional[dist.ProcessGroup] = None,
    fused: bool = True,
) -> torch.Tensor:
    if routing_weights.size() != selected_experts.size():
        routing_weights = routing_weights.gather(1, selected_experts)

    profiler = _MoePhaseProfiler(fused=fused)
    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    input_splits, output_splits, num_global_tokens_per_local_expert, num_global_sum_tokens_per_local_expert = (
        dispatch_preprocess(selected_experts, num_experts, ep_group)
    )
    profiler.mark("dispatch_preprocess")
    hidden_states, unpermute_indices, post_dispatch_unpermute_indices = alltoall_dispatch(
        hidden_states,
        selected_experts,
        input_splits,
        output_splits,
        num_experts,
        num_global_tokens_per_local_expert,
        ep_group,
        fused=fused,
        profiler=profiler,
    )

    # If no tokens are assigned to the expert in the current EP shard, no computation is performed
    if hidden_states.shape[0] > 0:
        intermediate_hidden_states = grouped_matmul(hidden_states, fc1_weight, num_global_sum_tokens_per_local_expert, fused=fused)
        profiler.mark("gmm_fc1")
        intermediate_activations = swiglu(intermediate_hidden_states, dim=-1, fused=fused)
        profiler.mark("swiglu")
        hidden_states = grouped_matmul(
            intermediate_activations, fc2_weight, num_global_sum_tokens_per_local_expert, fused=fused
        )
        profiler.mark("gmm_fc2")
    else:
        # empty operation to avoid no grads for experts' weights
        intermediate_hidden_states = hidden_states @ fc1_weight.sum(0)
        gate_output, down_output = torch.chunk(intermediate_hidden_states, 2, dim=-1)
        hidden_states = (gate_output + down_output) @ fc2_weight.sum(0) * 0.
        profiler.mark("gmm_fc1")
        profiler.mark("swiglu")
        profiler.mark("gmm_fc2")

    hidden_states = alltoall_combine(
        hidden_states,
        routing_weights,
        post_dispatch_unpermute_indices,
        unpermute_indices,
        input_splits,
        output_splits,
        num_experts,
        num_global_tokens_per_local_expert,
        ep_group,
        profiler=profiler,
    )
    profiler.finish(input_splits, output_splits, num_global_sum_tokens_per_local_expert)
    return hidden_states


def dispatch_preprocess(
    selected_experts: torch.Tensor,
    num_global_experts: int,
    ep_group: Optional[dist.ProcessGroup] = None,
):
    if ep_group is None:
        ep_size = 1
        ep_rank = 0
    else:
        ep_size = dist.get_world_size(ep_group)
        ep_rank = dist.get_rank(ep_group)
    if num_global_experts % ep_size != 0:
        raise ValueError(
            f"Number of experts ({num_global_experts}) must be divisible by expert parallel size ({ep_size})."
    )
    num_local_experts = num_global_experts // ep_size

    num_local_tokens_per_expert = torch.bincount(selected_experts.view(-1), minlength=num_global_experts)

    if ep_group is None or ep_size <= 1:
        num_global_tokens_per_expert = num_local_tokens_per_expert.view(1, -1)
    else:
        num_global_tokens_per_expert = torch.zeros(
            ep_size,
            num_global_experts,
            dtype=num_local_tokens_per_expert.dtype,
            device=num_local_tokens_per_expert.device,
        )
        dist.all_gather_into_tensor(num_global_tokens_per_expert, num_local_tokens_per_expert, group=ep_group)

    start_idx, end_idx = ep_rank * num_local_experts, (ep_rank + 1) * num_local_experts
    num_global_tokens_per_local_expert = num_global_tokens_per_expert[:, start_idx:end_idx].contiguous()

    input_splits = num_local_tokens_per_expert.reshape(ep_size, num_local_experts).sum(dim=1).tolist()
    output_splits = num_global_tokens_per_local_expert.sum(dim=1).tolist()

    num_global_sum_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(dim=0)
    return input_splits, output_splits, num_global_tokens_per_local_expert, num_global_sum_tokens_per_local_expert


def alltoall_dispatch(
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    input_splits: List,
    output_splits: List,
    num_global_experts: int,
    num_global_tokens_per_local_expert: torch.Tensor,
    ep_group: Optional[dist.ProcessGroup] = None,
    fused: bool = True,
    profiler: Optional[_MoePhaseProfiler] = None,
):
    hidden_states, unpermute_indices = permute(hidden_states, selected_experts.to(torch.int32), fused=fused)
    if profiler is not None:
        profiler.mark("permute_pre_a2a")
    hidden_states = all_to_all(hidden_states, ep_group, scatter_sizes=input_splits, gather_sizes=output_splits)
    if profiler is not None:
        profiler.mark("alltoall_dispatch")

    # No tokens have been assigned to the expert in the current EP shard
    if hidden_states.shape[0] == 0:
        return hidden_states, unpermute_indices, None

    ep_size = 1 if ep_group is None else dist.get_world_size(ep_group)
    num_local_experts = num_global_experts // ep_size
    if num_global_experts % ep_size != 0:
        raise ValueError(
            f"Number of experts ({num_global_experts}) must be divisible by expert parallel size ({ep_size})."
    )

    _expert_ids_per_ep_rank = torch.arange(num_global_experts, dtype=torch.int32, device=hidden_states.device) % num_local_experts
    global_input_tokens_local_experts_indices = torch.repeat_interleave(_expert_ids_per_ep_rank, num_global_tokens_per_local_expert.ravel())
    hidden_states, post_dispatch_unpermute_indices = permute(hidden_states, global_input_tokens_local_experts_indices, fused=fused)
    if profiler is not None:
        profiler.mark("permute_post_a2a")

    return hidden_states, unpermute_indices, post_dispatch_unpermute_indices


def alltoall_combine(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    post_dispatch_unpermute_indices: torch.Tensor,
    unpermute_indices: torch.Tensor,
    input_splits: List,
    output_splits: List,
    num_global_experts: int,
    num_global_tokens_per_local_expert: torch.Tensor,
    ep_group: Optional[dist.ProcessGroup] = None,
    fused: bool = True,
    profiler: Optional[_MoePhaseProfiler] = None,
):
    # If no tokens are assigned to the expert in the current EP shard, no computation is performed
    if hidden_states.shape[0] > 0:
        ep_size = 1 if ep_group is None else dist.get_world_size(ep_group)
        if num_global_experts % ep_size != 0:
            raise ValueError(
                f"Number of experts ({num_global_experts}) must be divisible by expert parallel size ({ep_size})."
        )

        hidden_states = unpermute(hidden_states, post_dispatch_unpermute_indices, fused=fused)
    if profiler is not None:
        profiler.mark("unpermute_pre_combine")

    hidden_states = all_to_all(hidden_states, ep_group, scatter_sizes=output_splits, gather_sizes=input_splits)
    if profiler is not None:
        profiler.mark("alltoall_combine")
    hidden_states = unpermute(hidden_states.to(routing_weights.dtype), unpermute_indices,
                                                      probs=routing_weights, fused=fused)
    if profiler is not None:
        profiler.mark("unpermute_final")
    return hidden_states
