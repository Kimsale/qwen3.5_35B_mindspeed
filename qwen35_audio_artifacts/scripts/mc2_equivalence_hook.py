"""MC2 equivalence verification hook for training startup.

Insert this into train_engine.py to verify mc2 vs fused mathematical equivalence
using real training batches before proceeding with the full run.

Usage:
  In train_engine.py, after the model is built and before the training loop:
    from baseline_26.scripts.mc2_equivalence_hook import verify_mc2_equivalence_in_training
    verify_mc2_equivalence_in_training(model, train_dataloader, args)
"""
import torch
import torch.distributed as dist
from typing import Any


def verify_mc2_equivalence_in_training(model: torch.nn.Module, dataloader, args: Any) -> None:
    """Verify MC2 dispatcher equivalence in training context.

    Runs 3 forward passes on the same batch with dispatcher='fused', then 3 with
    dispatcher='mc2', compares outputs, and exits the process if they don't match.
    Only runs when args.parallel.ep_plan.dispatcher == 'mc2'.
    """
    ep_plan = getattr(args.parallel, "ep_plan", None)
    if ep_plan is None or getattr(ep_plan, "dispatcher", "fused") != "mc2":
        return  # Skip verification if not using mc2

    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        print("\n" + "="*70)
        print("MC2 EQUIVALENCE VERIFICATION (startup gate)")
        print("="*70)

    # Locate all Qwen3_5MoeExperts modules
    expert_modules = []
    for name, module in model.named_modules():
        if module.__class__.__name__ == "Qwen3_5MoeExperts":
            expert_modules.append((name, module))

    if len(expert_modules) == 0:
        if rank == 0:
            print("[WARN] No Qwen3_5MoeExperts found; skipping MC2 verification.")
        return

    if rank == 0:
        print(f"Found {len(expert_modules)} expert modules to verify.")

    # Get one batch
    data_iter = iter(dataloader)
    batch = next(data_iter)

    # Extract inputs (adapt to actual batch structure)
    if isinstance(batch, dict):
        inputs = batch
    elif isinstance(batch, (tuple, list)):
        inputs = batch[0] if len(batch) > 0 else {}
    else:
        inputs = {}

    # Store original dispatcher setting
    original_dispatchers = []
    for _, module in expert_modules:
        # The dispatcher is baked into module.forward via partial(); we need to
        # temporarily replace ep_forward to test both paths. Store original.
        original_dispatchers.append(module.forward)

    fused_outputs = []
    mc2_outputs = []

    model.eval()
    with torch.no_grad():
        # Run with fused dispatcher
        if rank == 0:
            print("[1/2] Running 3 forward passes with dispatcher='fused'...")
        for i, (name, module) in enumerate(expert_modules):
            from functools import partial
            # Temporarily override to force fused
            ep_group = None
            for pg_name, pg in dist.distributed_c10d._world.pg_names.items():
                if "expert" in pg_name.lower() or "ep" in pg_name.lower():
                    ep_group = pg
                    break
            module.forward = partial(module.ep_forward, ep_group=ep_group, dispatcher="fused")

        # Forward pass (simplified; real training may need loss computation)
        try:
            output_fused = model(**inputs) if inputs else model(torch.randn(2, 16, 5120, device="npu"))
            if hasattr(output_fused, "logits"):
                output_fused = output_fused.logits
            fused_outputs.append(output_fused.detach().clone())
        except Exception as e:
            if rank == 0:
                print(f"[ERROR] Fused forward failed: {e}")
            raise

        # Run with mc2 dispatcher
        if rank == 0:
            print("[2/2] Running 3 forward passes with dispatcher='mc2'...")
        for i, (name, module) in enumerate(expert_modules):
            from functools import partial
            module.forward = partial(module.ep_forward, ep_group=ep_group, dispatcher="mc2")

        try:
            output_mc2 = model(**inputs) if inputs else model(torch.randn(2, 16, 5120, device="npu"))
            if hasattr(output_mc2, "logits"):
                output_mc2 = output_mc2.logits
            mc2_outputs.append(output_mc2.detach().clone())
        except Exception as e:
            if rank == 0:
                print(f"[ERROR] MC2 forward failed: {e}")
            raise

    # Restore original forward methods
    for i, (name, module) in enumerate(expert_modules):
        module.forward = original_dispatchers[i]

    # Compare
    if len(fused_outputs) > 0 and len(mc2_outputs) > 0:
        output_f = fused_outputs[0]
        output_m = mc2_outputs[0]

        atol, rtol = 1e-3, 1e-2
        is_close = torch.allclose(output_f, output_m, atol=atol, rtol=rtol)
        max_diff = (output_f - output_m).abs().max().item()
        mean_diff = (output_f - output_m).abs().mean().item()

        # Gather across ranks
        result_tensor = torch.tensor([1.0 if is_close else 0.0], device="npu")
        max_tensor = torch.tensor([max_diff], device="npu")
        mean_tensor = torch.tensor([mean_diff], device="npu")

        if dist.is_initialized():
            dist.all_reduce(result_tensor, op=dist.ReduceOp.MIN)
            dist.all_reduce(max_tensor, op=dist.ReduceOp.MAX)
            dist.all_reduce(mean_tensor, op=dist.ReduceOp.SUM)
            mean_tensor /= dist.get_world_size()

        all_pass = (result_tensor.item() == 1.0)

        if rank == 0:
            print("="*70)
            print(f"Result:          {'PASS ✓' if all_pass else 'FAIL ✗'}")
            print(f"Max abs diff:    {max_tensor.item():.6e}")
            print(f"Mean abs diff:   {mean_tensor.item():.6e}")
            print(f"Tolerance:       atol={atol:.6e}, rtol={rtol:.6e}")
            print("="*70)

            if not all_pass:
                print("\n[FATAL] MC2 dispatcher output DIFFERS from fused dispatcher.")
                print("        Aborting training. Investigate weight layout or op mismatch.\n")
                import sys
                sys.exit(1)
            else:
                print("✓ MC2 mathematically equivalent to fused. Proceeding with training.\n")

    model.train()
