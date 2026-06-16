#!/usr/bin/env python3
"""Verify MC2 dispatcher mathematical equivalence to fused dispatcher.

Runs a minimal forward pass through Qwen3.5 MoE experts with EP=8 on identical
inputs using both 'fused' and 'mc2' dispatchers. Asserts outputs are allclose
within bf16 numerical tolerance. This is a hard gate before any training runs.

Usage:
  torchrun --nproc_per_node=8 verify_mc2_equivalence.py
"""
import argparse
import os
import sys
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.distributed.device_mesh import init_device_mesh

# Minimal import to avoid pulling full training stack
sys.path.insert(0, "/data/sejin/third_party/mindspeed-mm-26.0.0")


def init_ep_group(world_size: int) -> ProcessGroup:
    """Initialize expert-parallel process group spanning all ranks."""
    assert world_size == 8, "This test requires exactly 8 NPUs for EP=8"
    return dist.new_group(list(range(world_size)))


def create_mock_expert_module(config_dict: dict, ep_group: ProcessGroup):
    """Create a Qwen3_5MoeExperts module with DTensor-sharded weights on EP mesh."""
    import importlib.util
    from torch.distributed.tensor import DTensor, Shard, distribute_tensor

    # Direct import to avoid mindspeed_mm.__init__ pulling megatron
    spec = importlib.util.spec_from_file_location(
        "qwen_moe_mod",
        "/data/sejin/third_party/mindspeed-mm-26.0.0/mindspeed_mm/fsdp/models/qwen3_5_moe/modeling_qwen3_5_moe.py"
    )
    qwen_moe_mod = importlib.util.module_from_spec(spec)
    sys.modules["qwen_moe_mod"] = qwen_moe_mod
    spec.loader.exec_module(qwen_moe_mod)
    Qwen3_5MoeExperts = qwen_moe_mod.Qwen3_5MoeExperts

    from transformers.models.qwen2_moe import Qwen2MoeConfig

    config = Qwen2MoeConfig(**config_dict)
    module = Qwen3_5MoeExperts(config).to("npu").to(torch.bfloat16)

    # Shard expert weights on dim=0 (expert dimension) across EP ranks
    ep_mesh = init_device_mesh("npu", (8,), mesh_dim_names=("ep",))
    module.gate_up_proj = torch.nn.Parameter(
        distribute_tensor(module.gate_up_proj, ep_mesh, [Shard(0)])
    )
    module.down_proj = torch.nn.Parameter(
        distribute_tensor(module.down_proj, ep_mesh, [Shard(0)])
    )

    return module


def run_forward(module, hidden_states, top_k_index, top_k_weights, ep_group, dispatcher: str):
    """Run a single forward pass through ep_forward with specified dispatcher."""
    with torch.no_grad():
        output = module.ep_forward(
            hidden_states.clone(),
            top_k_index.clone(),
            top_k_weights.clone(),
            ep_group=ep_group,
            dispatcher=dispatcher
        )
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--atol", type=float, default=1e-3, help="Absolute tolerance for allclose (BF16 precision)")
    parser.add_argument("--rtol", type=float, default=1e-2, help="Relative tolerance for allclose")
    args = parser.parse_args()

    # Initialize distributed
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl")
    ep_group = init_ep_group(world_size)

    # Minimal Qwen3.5-35B-A3B config (only MoE params needed)
    config_dict = {
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "hidden_size": 5120,
        "moe_intermediate_size": 6912,
        "hidden_act": "silu",
        "use_grouped_expert_matmul": True,
    }

    if rank == 0:
        print(f"[Rank {rank}] Creating Qwen3_5MoeExperts module with EP={world_size}")
    module = create_mock_expert_module(config_dict, ep_group)

    # Synthetic input: batch=2, seq=16, hidden=5120, top-8 routing
    torch.manual_seed(42 + rank)  # Different seed per rank to simulate real data
    batch_size, seq_len = 2, 16
    hidden_states = torch.randn(batch_size, seq_len, 5120, dtype=torch.bfloat16, device="npu")
    top_k_index = torch.randint(0, 128, (batch_size, seq_len, 8), dtype=torch.long, device="npu")
    top_k_weights = torch.rand(batch_size, seq_len, 8, dtype=torch.bfloat16, device="npu")
    # Normalize routing weights
    top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

    if rank == 0:
        print(f"[Rank {rank}] Running fused dispatcher...")
    output_fused = run_forward(module, hidden_states, top_k_index, top_k_weights, ep_group, "fused")

    if rank == 0:
        print(f"[Rank {rank}] Running mc2 dispatcher...")
    output_mc2 = run_forward(module, hidden_states, top_k_index, top_k_weights, ep_group, "mc2")

    # Verify numerical equivalence
    if rank == 0:
        print(f"[Rank {rank}] Comparing outputs (atol={args.atol}, rtol={args.rtol})...")

    is_close = torch.allclose(output_fused, output_mc2, atol=args.atol, rtol=args.rtol)
    max_abs_diff = (output_fused - output_mc2).abs().max().item()
    mean_abs_diff = (output_fused - output_mc2).abs().mean().item()

    # Gather results from all ranks
    all_close_tensor = torch.tensor([1.0 if is_close else 0.0], device="npu")
    max_diff_tensor = torch.tensor([max_abs_diff], device="npu")
    mean_diff_tensor = torch.tensor([mean_abs_diff], device="npu")

    dist.all_reduce(all_close_tensor, op=dist.ReduceOp.MIN)
    dist.all_reduce(max_diff_tensor, op=dist.ReduceOp.MAX)
    dist.all_reduce(mean_diff_tensor, op=dist.ReduceOp.SUM)
    mean_diff_tensor /= world_size

    if rank == 0:
        all_ranks_pass = (all_close_tensor.item() == 1.0)
        print(f"\n{'='*60}")
        print(f"MC2 Equivalence Verification Result")
        print(f"{'='*60}")
        print(f"All ranks pass:     {all_ranks_pass}")
        print(f"Max abs diff:       {max_diff_tensor.item():.6e}")
        print(f"Mean abs diff:      {mean_diff_tensor.item():.6e}")
        print(f"Tolerance (atol):   {args.atol:.6e}")
        print(f"Tolerance (rtol):   {args.rtol:.6e}")
        print(f"{'='*60}")

        if all_ranks_pass:
            print("✓ MC2 dispatcher is mathematically equivalent to fused dispatcher.")
            print("  Safe to proceed with performance evaluation.")
        else:
            print("✗ MC2 dispatcher output DIFFERS from fused dispatcher.")
            print("  DO NOT proceed to training. Investigate weight layout or op mismatch.")
            sys.exit(1)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
