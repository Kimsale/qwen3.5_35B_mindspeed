import os
import time
from collections import defaultdict
from dataclasses import dataclass
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
_ASYNC_A2A_HANDLES: List[object] = []
_PIPELINE_COMM_STREAM = None


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

    def add_elapsed(self, name: str, start_time: float):
        if not self.enabled:
            return
        _phase_sync(self.sync)
        self.phase_ms[name] = self.phase_ms.get(name, 0.0) + (time.perf_counter() - start_time) * 1000.0

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
                "pipeline_wait_dispatch",
                "pipeline_compute",
                "pipeline_start_combine",
                "pipeline_wait_combine",
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


class _AsyncAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, inputs, output_split_sizes, input_split_sizes, async_op):
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes

        world_size = dist.get_world_size(group=group)
        if world_size == 1:
            return inputs

        inputs = inputs.contiguous()
        output = inputs.new_empty(size=[sum(output_split_sizes)] + list(inputs.size()[1:]))
        handle = dist.all_to_all_single(
            output,
            inputs,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
            async_op=async_op,
        )
        if async_op:
            _ASYNC_A2A_HANDLES.append(handle)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = _all_to_all(ctx.group, grad_output.contiguous(), ctx.input_split_sizes, ctx.output_split_sizes)
        return None, grad_input, None, None, None


@dataclass
class _AsyncA2AWork:
    tensor: torch.Tensor
    input_tensor: Optional[torch.Tensor] = None
    handle: Optional[object] = None
    stream: Optional[object] = None

    def wait(self):
        if self.handle is not None:
            self.handle.wait()
        if self.stream is not None and hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.current_stream().wait_stream(self.stream)
        return self.tensor


@dataclass
class _PipelineChunkSpec:
    start: int
    end: int
    send_counts: List[int]
    recv_counts: List[int]
    local_expert_counts: torch.Tensor


def _get_pipeline_comm_stream():
    global _PIPELINE_COMM_STREAM
    if _PIPELINE_COMM_STREAM is None:
        _PIPELINE_COMM_STREAM = torch.npu.Stream(device=torch.npu.current_device())
    return _PIPELINE_COMM_STREAM


def _pop_async_a2a_handle() -> Optional[object]:
    if not _ASYNC_A2A_HANDLES:
        return None
    return _ASYNC_A2A_HANDLES.pop()


def _record_stream(tensor: torch.Tensor, stream):
    if stream is not None and hasattr(tensor, "record_stream"):
        tensor.record_stream(stream)


def _start_all_to_all(
    input_: torch.Tensor,
    process_group: dist.ProcessGroup,
    scatter_sizes: List[int],
    gather_sizes: List[int],
    multi_stream: bool = True,
) -> _AsyncA2AWork:
    world_size = dist.get_world_size(group=process_group)
    if world_size == 1:
        return _AsyncA2AWork(input_, input_tensor=input_)

    input_ = input_.contiguous()
    use_stream = multi_stream and hasattr(torch, "npu") and torch.npu.is_available()
    if use_stream:
        stream = _get_pipeline_comm_stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            output = _AsyncAllToAll.apply(process_group, input_, gather_sizes, scatter_sizes, True)
        _record_stream(input_, stream)
        _record_stream(output, stream)
        return _AsyncA2AWork(output, input_tensor=input_, handle=_pop_async_a2a_handle(), stream=stream)

    output = _AsyncAllToAll.apply(process_group, input_, gather_sizes, scatter_sizes, True)
    return _AsyncA2AWork(output, input_tensor=input_, handle=_pop_async_a2a_handle(), stream=None)


def _build_pipeline_ranges(
    num_global_tokens_per_expert: torch.Tensor,
    ep_size: int,
    num_local_experts: int,
    pipeline_chunks: int,
    min_tokens_per_chunk: int,
):
    num_chunks = max(1, min(int(pipeline_chunks), num_local_experts))
    chunk_size = (num_local_experts + num_chunks - 1) // num_chunks
    ranges = [
        (start, min(start + chunk_size, num_local_experts))
        for start in range(0, num_local_experts, chunk_size)
    ]
    if min_tokens_per_chunk <= 0 or len(ranges) <= 1:
        return ranges

    merged_ranges = []
    current_start = None
    current_end = None
    current_tokens = 0
    for start, end in ranges:
        if current_start is None:
            current_start = start
        current_end = end
        for target_rank in range(ep_size):
            global_start = target_rank * num_local_experts + start
            global_end = target_rank * num_local_experts + end
            current_tokens += int(num_global_tokens_per_expert[:, global_start:global_end].sum().item())
        if current_tokens >= min_tokens_per_chunk:
            merged_ranges.append((current_start, current_end))
            current_start = None
            current_end = None
            current_tokens = 0
    if current_start is not None:
        if merged_ranges:
            prev_start, _ = merged_ranges.pop()
            merged_ranges.append((prev_start, current_end))
        else:
            merged_ranges.append((current_start, current_end))
    return merged_ranges


def _build_pipeline_specs(
    num_global_tokens_per_expert: torch.Tensor,
    ep_rank: int,
    ep_size: int,
    num_local_experts: int,
    pipeline_chunks: int,
    min_tokens_per_chunk: int,
) -> List[_PipelineChunkSpec]:
    local_counts = num_global_tokens_per_expert[ep_rank]
    specs = []
    for start, end in _build_pipeline_ranges(
        num_global_tokens_per_expert,
        ep_size,
        num_local_experts,
        pipeline_chunks,
        min_tokens_per_chunk,
    ):
        send_counts = []
        for target_rank in range(ep_size):
            global_start = target_rank * num_local_experts + start
            global_end = target_rank * num_local_experts + end
            send_counts.append(int(local_counts[global_start:global_end].sum().item()))

        local_start = ep_rank * num_local_experts + start
        local_end = ep_rank * num_local_experts + end
        recv_counts = [
            int(num_global_tokens_per_expert[source_rank, local_start:local_end].sum().item())
            for source_rank in range(ep_size)
        ]
        local_expert_counts = num_global_tokens_per_expert[:, local_start:local_end].sum(dim=0).contiguous()
        specs.append(_PipelineChunkSpec(start, end, send_counts, recv_counts, local_expert_counts))
    return specs


def _expert_offsets(counts: torch.Tensor) -> List[int]:
    offsets = [0]
    offsets.extend(torch.cumsum(counts, dim=0).tolist())
    return [int(value) for value in offsets]


def _slice_pipeline_dispatch_input(
    permuted_hidden_states: torch.Tensor,
    local_expert_offsets: List[int],
    spec: _PipelineChunkSpec,
    ep_size: int,
    num_local_experts: int,
):
    pieces = []
    for target_rank in range(ep_size):
        global_start = target_rank * num_local_experts + spec.start
        global_end = target_rank * num_local_experts + spec.end
        pieces.append(permuted_hidden_states[local_expert_offsets[global_start]:local_expert_offsets[global_end]])
    if pieces:
        return torch.cat(pieces, dim=0)
    return permuted_hidden_states.new_empty((0, permuted_hidden_states.shape[-1]))


def _local_expert_indices_for_chunk(
    spec: _PipelineChunkSpec,
    num_global_tokens_per_expert: torch.Tensor,
    ep_rank: int,
    ep_size: int,
    num_local_experts: int,
):
    local_ids = torch.arange(
        spec.end - spec.start,
        dtype=torch.int32,
        device=num_global_tokens_per_expert.device,
    )
    pieces = []
    local_start = ep_rank * num_local_experts + spec.start
    local_end = ep_rank * num_local_experts + spec.end
    for source_rank in range(ep_size):
        counts = num_global_tokens_per_expert[source_rank, local_start:local_end]
        pieces.append(torch.repeat_interleave(local_ids, counts))
    if pieces:
        return torch.cat(pieces, dim=0)
    return local_ids.new_empty((0,), dtype=torch.int32)


def _empty_expert_output(hidden_states: torch.Tensor, fc1_weight: torch.Tensor, fc2_weight: torch.Tensor):
    intermediate_hidden_states = hidden_states @ fc1_weight.sum(0)
    gate_output, down_output = torch.chunk(intermediate_hidden_states, 2, dim=-1)
    return (gate_output + down_output) @ fc2_weight.sum(0) * 0.


def ep_forward(
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
    ep_group: Optional[dist.ProcessGroup] = None,
    fused: bool = True,
    pipeline_chunks: int = 1,
    pipeline_multi_stream: bool = True,
    pipeline_min_tokens_per_chunk: int = 0,
) -> torch.Tensor:
    if routing_weights.size() != selected_experts.size():
        routing_weights = routing_weights.gather(1, selected_experts)

    ep_size = 1 if ep_group is None else dist.get_world_size(ep_group)
    if pipeline_chunks > 1 and ep_size > 1:
        return ep_forward_pipeline(
            num_experts,
            routing_weights,
            selected_experts,
            hidden_states,
            fc1_weight,
            fc2_weight,
            ep_group=ep_group,
            fused=fused,
            pipeline_chunks=pipeline_chunks,
            pipeline_multi_stream=pipeline_multi_stream,
            pipeline_min_tokens_per_chunk=pipeline_min_tokens_per_chunk,
        )

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


def ep_forward_pipeline(
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
    ep_group: dist.ProcessGroup,
    fused: bool = True,
    pipeline_chunks: int = 2,
    pipeline_multi_stream: bool = True,
    pipeline_min_tokens_per_chunk: int = 0,
) -> torch.Tensor:
    profiler = _MoePhaseProfiler(fused=fused)
    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    (
        input_splits,
        output_splits,
        num_global_tokens_per_local_expert,
        num_global_sum_tokens_per_local_expert,
        num_global_tokens_per_expert,
    ) = dispatch_preprocess(selected_experts, num_experts, ep_group, return_global_tokens=True)
    profiler.mark("dispatch_preprocess")

    ep_size = dist.get_world_size(ep_group)
    ep_rank = dist.get_rank(ep_group)
    if num_experts % ep_size != 0:
        raise ValueError(f"Number of experts ({num_experts}) must be divisible by expert parallel size ({ep_size}).")
    num_local_experts = num_experts // ep_size

    permuted_hidden_states, unpermute_indices = permute(hidden_states, selected_experts.to(torch.int32), fused=fused)
    profiler.mark("permute_pre_a2a")
    local_expert_offsets = _expert_offsets(num_global_tokens_per_expert[ep_rank])
    specs = _build_pipeline_specs(
        num_global_tokens_per_expert,
        ep_rank,
        ep_size,
        num_local_experts,
        pipeline_chunks,
        pipeline_min_tokens_per_chunk,
    )

    rank_parts: List[List[torch.Tensor]] = [[] for _ in range(ep_size)]
    dispatch_work = None
    combine_work = None
    combine_spec = None

    def start_dispatch(spec: _PipelineChunkSpec):
        chunk_input = _slice_pipeline_dispatch_input(
            permuted_hidden_states,
            local_expert_offsets,
            spec,
            ep_size,
            num_local_experts,
        )
        return _start_all_to_all(
            chunk_input,
            ep_group,
            scatter_sizes=spec.send_counts,
            gather_sizes=spec.recv_counts,
            multi_stream=pipeline_multi_stream,
        )

    def finish_combine(work: _AsyncA2AWork, spec: _PipelineChunkSpec):
        combined = work.wait()
        profiler.mark("pipeline_wait_combine")
        for rank, part in enumerate(torch.split(combined, spec.send_counts, dim=0)):
            rank_parts[rank].append(part)

    if specs:
        dispatch_work = start_dispatch(specs[0])

    for idx, spec in enumerate(specs):
        next_dispatch_work = start_dispatch(specs[idx + 1]) if idx + 1 < len(specs) else None

        dispatched_hidden_states = dispatch_work.wait()
        profiler.mark("pipeline_wait_dispatch")
        compute_start = time.perf_counter()
        if dispatched_hidden_states.shape[0] > 0:
            local_expert_indices = _local_expert_indices_for_chunk(
                spec,
                num_global_tokens_per_expert,
                ep_rank,
                ep_size,
                num_local_experts,
            )
            dispatched_hidden_states, post_dispatch_unpermute_indices = permute(
                dispatched_hidden_states,
                local_expert_indices,
                fused=fused,
            )
            profiler.mark("permute_post_a2a")
            intermediate_hidden_states = grouped_matmul(
                dispatched_hidden_states,
                fc1_weight[spec.start:spec.end],
                spec.local_expert_counts,
                fused=fused,
            )
            profiler.mark("gmm_fc1")
            intermediate_activations = swiglu(intermediate_hidden_states, dim=-1, fused=fused)
            profiler.mark("swiglu")
            chunk_output = grouped_matmul(
                intermediate_activations,
                fc2_weight[spec.start:spec.end],
                spec.local_expert_counts,
                fused=fused,
            )
            profiler.mark("gmm_fc2")
            chunk_output = unpermute(chunk_output, post_dispatch_unpermute_indices, fused=fused)
            profiler.mark("unpermute_pre_combine")
        else:
            chunk_output = _empty_expert_output(
                dispatched_hidden_states,
                fc1_weight[spec.start:spec.end],
                fc2_weight[spec.start:spec.end],
            )
            profiler.mark("gmm_fc1")
            profiler.mark("swiglu")
            profiler.mark("gmm_fc2")
            profiler.mark("unpermute_pre_combine")
        profiler.add_elapsed("pipeline_compute", compute_start)

        if combine_work is not None:
            finish_combine(combine_work, combine_spec)

        current_combine_work = _start_all_to_all(
            chunk_output,
            ep_group,
            scatter_sizes=spec.recv_counts,
            gather_sizes=spec.send_counts,
            multi_stream=pipeline_multi_stream,
        )
        profiler.mark("pipeline_start_combine")

        dispatch_work = next_dispatch_work
        combine_work = current_combine_work
        combine_spec = spec

    if combine_work is not None:
        finish_combine(combine_work, combine_spec)

    rank_outputs = []
    for parts in rank_parts:
        if parts:
            rank_outputs.append(torch.cat(parts, dim=0))
    if rank_outputs:
        hidden_states = torch.cat(rank_outputs, dim=0)
    else:
        hidden_states = hidden_states.new_empty((0, hidden_states.shape[-1]))
    hidden_states = unpermute(hidden_states.to(routing_weights.dtype), unpermute_indices, probs=routing_weights, fused=fused)
    profiler.mark("unpermute_final")
    profiler.finish(input_splits, output_splits, num_global_sum_tokens_per_local_expert)
    return hidden_states


def dispatch_preprocess(
    selected_experts: torch.Tensor,
    num_global_experts: int,
    ep_group: Optional[dist.ProcessGroup] = None,
    return_global_tokens: bool = False,
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
    if return_global_tokens:
        return (
            input_splits,
            output_splits,
            num_global_tokens_per_local_expert,
            num_global_sum_tokens_per_local_expert,
            num_global_tokens_per_expert,
        )
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
