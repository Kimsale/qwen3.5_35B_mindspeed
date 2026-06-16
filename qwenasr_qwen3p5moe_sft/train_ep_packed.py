#!/usr/bin/env python3
"""EP Training Script with Sequence Packing - Pack format eliminates padding waste.

Key changes from train_ep.py:
1. PackedDataCollator: concatenates sequences without padding, uses cu_seqlens for boundaries
2. Model forward: supports packed input (1D input_ids + cu_seqlens + position_ids)
3. Compatible with existing EP architecture and audio encoder varlen mode
"""

# Import everything from original train_ep.py
import sys
import os

# Add current directory to path so we can import from train_ep
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_ep import *  # Import all existing functions and classes

# Override DataCollator with packed version
@dataclass
class PackedDataCollator:
    """Sequence packing collator - eliminates padding waste.

    Converts batch of variable-length sequences into a single packed sequence:
      - input_ids: concatenated 1D tensor (total_len,)
      - position_ids: per-sample position indices (total_len,)
      - cu_seqlens: cumulative sequence lengths [0, len1, len1+len2, ...] (batch_size+1,)
      - labels: concatenated with -100 mask (total_len,)

    Audio features remain unchanged (already packed via concat).
    """
    tokenizer: AutoTokenizer

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # ========== Text sequences: Pack format (no padding) ==========
        packed_input_ids = []
        packed_labels = []
        cu_seqlens = [0]  # Cumulative sequence lengths, starts at 0
        position_ids = []
        sample_lens = []

        for f in features:
            seq_len = len(f["input_ids"])
            # Concatenate sequences directly
            packed_input_ids.extend(f["input_ids"])
            packed_labels.extend(f["labels"])
            cu_seqlens.append(cu_seqlens[-1] + seq_len)
            # Position IDs: independent counting per sample (0, 1, 2, ... for each sample)
            position_ids.extend(list(range(seq_len)))
            sample_lens.append(f["sample_len"])

        # ========== Audio features: keep existing concat approach (already optimal) ==========
        audio_list = []
        feature_lens = []
        for f in features:
            audio_list.append(f["input_features"])   # each is (128, real_len)
            feature_lens.append(f["feature_lens"])

        batch = {
            "input_ids": torch.tensor(packed_input_ids, dtype=torch.long),      # (total_len,) - 1D packed
            "position_ids": torch.tensor(position_ids, dtype=torch.long),       # (total_len,) - per-sample positions
            "cu_seqlens": torch.tensor(cu_seqlens, dtype=torch.int32),          # (batch_size+1,) - sample boundaries
            "labels": torch.tensor(packed_labels, dtype=torch.long),            # (total_len,) - 1D packed
            "sample_lens": torch.tensor(sample_lens, dtype=torch.long),         # (batch_size,) - original lengths
            "input_features": torch.cat(audio_list, dim=1),                     # (128, total_audio_len)
            "feature_lens": torch.tensor(feature_lens, dtype=torch.long),       # (batch_size,)
        }

        # Note: attention_mask is removed - cu_seqlens implicitly defines masking in FA2 varlen mode
        return batch


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_path", type=str, default="/repo1/yjjiang11/work/vibecoding/Qwen3.5/qwen3.5_MOE_AST_sft/models/Qwen/Qwen3_5_CausalLM")
    parser.add_argument("--asr_path", type=str, default="/repo1/yjjiang11/work/vibecoding/Qwen3.5/qwen3.5_MOE_AST_sft/models/Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--tokenizer_path", type=str, default="/repo1/yjjiang11/work/vibecoding/Qwen3.5/qwen3.5_MOE_AST_sft/models/Qwen/Qwen3___5-35B-A3B")
    parser.add_argument("--train_data_path", type=str, default="/repo1/yjjiang11/work/vibecoding/Qwen3.5/qwenasr_qwen3p5moe_sft_lmdb/lmdbdata/train.json")
    parser.add_argument("--eval_data_path", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="output_ep_packed")
    parser.add_argument("--batch_tokens", type=int, default=1024, help="Max total tokens per batch (sum of lengths)")
    parser.add_argument("--max_batch_size", type=int, default=None, help="Max number of samples per batch")
    parser.add_argument("--shuffle_chunk_entries", type=int, default=8)
    parser.add_argument("--shuffle_block_samples", type=int, default=5000)
    parser.add_argument("--sortish_window_size", type=int, default=1024)
    parser.add_argument("--sampler_mode", type=str, default="legacy", choices=["global_random", "legacy", "balanced"])
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--lr_decay_iters", type=int, default=0)
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_tokens_persample", type=int, default=None)
    parser.add_argument("--asr_acc_threshold", type=float, default=2.0)
    parser.add_argument("--lmdb_tokenizer_path", type=str, default="/repo1/yjjiang11/work/vibecoding/Qwen3.5/qwenasr_qwen3p5moe_sft_lmdb/lmdbdata/res")
    parser.add_argument("--loss_bucket_edges", type=str, default="640,1280,1920,2600")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--ep_size", type=int, default=8)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    args = parser.parse_args()

    # Setup distributed
    rank, world_size, local_rank, device = setup_distributed()

    ep_size = args.ep_size if args.ep_size > 0 else world_size
    if world_size % ep_size != 0:
        raise ValueError(f"world_size={world_size} must be divisible by ep_size={ep_size}")

    num_expert_replicas = world_size // ep_size
    expert_replica_rank = rank // ep_size
    ep_rank = rank % ep_size

    # Create output dir and logging
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO if rank == 0 else logging.WARN)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(os.path.join(log_dir, f"rank{rank}.log"), mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    if rank == 0:
        logger.info(f"[PACKED FORMAT] Training with sequence packing enabled")
        logger.info(f"Training: world_size={world_size}, ep_size={ep_size}, expert_replicas={num_expert_replicas}")
        logger.info(f"Args: {args}")

    # Create EP groups
    ep_groups = []
    for i in range(num_expert_replicas):
        ep_group_ranks = list(range(i * ep_size, (i + 1) * ep_size))
        ep_groups.append(dist.new_group(ep_group_ranks))

    ep_group = ep_groups[expert_replica_rank]
    ep_group_ranks = list(range(expert_replica_rank * ep_size, (expert_replica_rank + 1) * ep_size))

    expert_replica_groups = []
    for i in range(ep_size):
        replica_group_ranks = list(range(i, world_size, ep_size))
        expert_replica_groups.append(dist.new_group(replica_group_ranks))

    expert_replica_group = expert_replica_groups[ep_rank]

    # Load tokenizer and feature extractor
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.asr_path, trust_remote_code=True)
    audio_token_id = tokenizer.convert_tokens_to_ids("<|audio_pad|>")

    if rank == 0:
        logger.info(f"Audio token ID: {audio_token_id}")

    # Load model with EP
    model = EPSpeechTranslationModel(
        llm_path=args.llm_path,
        asr_path=args.asr_path,
        audio_token_id=audio_token_id,
        ep_size=ep_size,
        ep_rank=ep_rank,
        ep_group=ep_group,
        device=device,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # LoRA or full finetuning
    if args.use_lora:
        apply_lora_to_model(model, args.lora_rank, args.lora_alpha, args.lora_dropout)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if rank == 0:
            t_count = sum(p.numel() for p in trainable_params)
            logger.info(f"LoRA mode: {t_count/1e6:.2f}M trainable params")

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
    else:
        non_expert_params, expert_params = get_non_expert_params(model)
        if rank == 0:
            ne_count = sum(p.numel() for p in non_expert_params)
            e_count = sum(p.numel() for p in expert_params)
            logger.info(f"Full finetuning: Non-expert {ne_count/1e6:.1f}M, Expert (local) {e_count/1e6:.1f}M")

        optimizer = torch.optim.AdamW(
            [
                {"params": non_expert_params, "lr": args.learning_rate, "weight_decay": args.weight_decay},
                {"params": expert_params, "lr": args.learning_rate, "weight_decay": args.weight_decay},
            ],
            betas=(0.9, 0.999),
            eps=1e-8,
        )

    # Dataset: use PackedDataCollator
    train_dataset = LmdbSpeechTranslationDataset(
        args.train_data_path, tokenizer, feature_extractor,
        args.max_tokens_persample, args.asr_acc_threshold,
        lmdb_tokenizer_path=args.lmdb_tokenizer_path,
    )
    batch_sampler = DynamicBatchSampler(
        lengths=train_dataset.all_lengths,
        batch_tokens=args.batch_tokens,
        world_size=world_size,
        rank=rank,
        shuffle=True,
        seed=0,
        max_batch_size=args.max_batch_size,
        cumulative_sizes=train_dataset.filtered_cumulative_sizes,
        shuffle_chunk_entries=args.shuffle_chunk_entries,
        shuffle_block_samples=args.shuffle_block_samples,
        sortish_window_size=args.sortish_window_size,
        sampler_mode=args.sampler_mode,
    )

    # *** KEY CHANGE: Use PackedDataCollator ***
    data_collator = PackedDataCollator(tokenizer=tokenizer)

    train_loader = DataLoader(
        train_dataset, batch_sampler=batch_sampler,
        collate_fn=data_collator, num_workers=4, pin_memory=True,
        persistent_workers=True,
    )

    if rank == 0:
        logger.info(f"[PACKED FORMAT] Using PackedDataCollator - eliminates padding waste")

    logger.info(
        "Parallel topology | "
        f"rank={rank} local_rank={local_rank} "
        f"ep_rank={ep_rank} expert_replica_rank={expert_replica_rank} "
        f"ep_group={ep_group_ranks} expert_replica_group={list(range(ep_rank, world_size, ep_size))} "
        f"sampler_rank={rank}/{world_size}"
    )

    # LR scheduler
    steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
    total_steps = steps_per_epoch * args.num_epochs
    lr_decay_iters = args.lr_decay_iters if args.lr_decay_iters > 0 else total_steps - args.warmup_steps
    scheduler = get_cosine_annealing_schedule(
        optimizer, args.warmup_steps, lr_decay_iters, args.min_lr, args.learning_rate
    )

    loss_bucket_edges = _parse_loss_bucket_edges(args.loss_bucket_edges)
    loss_bucket_labels = [
        _format_loss_bucket_label(loss_bucket_edges, idx)
        for idx in range(len(loss_bucket_edges) + 1)
    ]
    loss_bucket_edges_tensor = torch.tensor(loss_bucket_edges, dtype=torch.long, device=device)
    num_loss_buckets = len(loss_bucket_labels)

    if rank == 0:
        logger.info(
            f"Dataset stats | raw_total={train_dataset.raw_total_samples} "
            f"filtered={train_dataset.filtered_samples} ({train_dataset.filtered_ratio:.2%}) "
            f"active={len(train_dataset)} ({train_dataset.kept_ratio:.2%})"
        )
        logger.info(
            f"Training: {steps_per_epoch} steps/epoch, {total_steps} total steps"
        )

    # Resume from checkpoint
    resume_epoch = 0
    resume_global_step = 0
    resume_step_in_epoch = 0

    if args.resume_from_checkpoint:
        ckpt_dir = args.resume_from_checkpoint
        if args.use_lora:
            load_lora_checkpoint(model, ckpt_dir, ep_rank)
        else:
            load_ep_checkpoint(model, ckpt_dir, ep_rank)

        ckpt_world_size = _infer_checkpoint_world_size(ckpt_dir)
        can_resume_train_state = ckpt_world_size == world_size

        if can_resume_train_state:
            state_path = os.path.join(ckpt_dir, f"train_state_rank{rank}.pt")
            if os.path.exists(state_path):
                train_state = torch.load(state_path, map_location="cpu")
                optimizer.load_state_dict(train_state["optimizer"])
                scheduler.load_state_dict(train_state["scheduler"])
                resume_global_step = train_state["global_step"]
                resume_epoch = train_state["epoch"]
                resume_step_in_epoch = resume_global_step * args.gradient_accumulation_steps - \
                    resume_epoch * steps_per_epoch * args.gradient_accumulation_steps
                if rank == 0:
                    logger.info(f"Resumed from {ckpt_dir}: epoch={resume_epoch}, global_step={resume_global_step}")

    # Training loop
    model.train()
    global_step = resume_global_step
    accumulated_loss = 0.0
    local_step_loss_sum = 0.0
    local_step_valid_tokens = 0
    local_window_loss_sum = 0.0
    local_window_valid_tokens = 0
    local_bucket_loss_sums = torch.zeros(num_loss_buckets, dtype=torch.float32, device=device)
    local_bucket_token_counts = torch.zeros(num_loss_buckets, dtype=torch.float32, device=device)
    local_bucket_sample_counts = torch.zeros(num_loss_buckets, dtype=torch.float32, device=device)

    prof = {"data": 0.0, "forward": 0.0, "backward": 0.0, "sync": 0.0, "optim": 0.0}
    prof_steps = 0
    prof_t = time.time()

    for epoch in range(resume_epoch, args.num_epochs):
        batch_sampler.set_epoch(epoch)
        epoch_start = time.time()

        for step, batch in enumerate(train_loader):
            if epoch == resume_epoch and step < resume_step_in_epoch:
                continue

            if rank == 0:
                torch.npu.synchronize() if hasattr(torch, 'npu') else None
                t0 = time.time()
                prof["data"] += t0 - prof_t

            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward with packed format
            if rank == 0:
                torch.npu.synchronize() if hasattr(torch, 'npu') else None
                t1 = time.time()

            # *** KEY CHANGE: Pass packed format to model ***
            outputs = model(
                input_ids=batch["input_ids"],           # (total_len,) - 1D packed
                position_ids=batch["position_ids"],     # (total_len,) - per-sample positions
                cu_seqlens=batch["cu_seqlens"],         # (batch_size+1,) - boundaries
                labels=batch["labels"],                 # (total_len,) - 1D packed
                input_features=batch["input_features"], # (128, total_audio_len)
                feature_lens=batch["feature_lens"],     # (batch_size,)
            )

            valid_label_tokens = int((batch["labels"] != -100).sum().item())
            local_step_valid_tokens += valid_label_tokens
            local_step_loss_sum += float(outputs.loss.detach().float().item() * valid_label_tokens)

            # Compute per-sample loss for bucketing
            with torch.no_grad():
                batch_size = len(batch["cu_seqlens"]) - 1
                shift_labels = F.pad(batch["labels"], (0, 1), value=-100)[..., 1:].contiguous()
                per_token_loss = F.cross_entropy(
                    outputs.logits.detach().float().reshape(-1, outputs.logits.shape[-1]),
                    shift_labels.reshape(-1),
                    ignore_index=-100,
                    reduction="none",
                )

                # Aggregate loss per sample using cu_seqlens
                for b in range(batch_size):
                    start_idx = batch["cu_seqlens"][b].item()
                    end_idx = batch["cu_seqlens"][b+1].item()
                    sample_loss = per_token_loss[start_idx:end_idx]
                    sample_mask = shift_labels[start_idx:end_idx] != -100
                    sample_loss_sum = (sample_loss * sample_mask.to(sample_loss.dtype)).sum().float()
                    sample_token_count = sample_mask.sum().float()

                    # Bucket assignment
                    if len(loss_bucket_edges) > 0:
                        bucket_idx = torch.bucketize(batch["sample_lens"][b], loss_bucket_edges_tensor)
                    else:
                        bucket_idx = torch.tensor(0, device=device)

                    local_bucket_loss_sums[bucket_idx] += sample_loss_sum
                    local_bucket_token_counts[bucket_idx] += sample_token_count
                    local_bucket_sample_counts[bucket_idx] += 1.0

            loss = outputs.loss / args.gradient_accumulation_steps
            accumulated_loss += loss.item()
            if rank == 0:
                torch.npu.synchronize() if hasattr(torch, 'npu') else None
                prof["forward"] += time.time() - t1

            # Backward
            if rank == 0:
                torch.npu.synchronize() if hasattr(torch, 'npu') else None
                t2 = time.time()
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if rank == 0:
                    torch.npu.synchronize() if hasattr(torch, 'npu') else None
                    prof["backward"] += time.time() - t2
                    t3 = time.time()

                if args.use_lora:
                    sync_gradients(trainable_params, world_size)
                    torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                else:
                    sync_gradients(non_expert_params, world_size)
                    sync_gradients(expert_params, num_expert_replicas, expert_replica_group)
                    all_params = list(non_expert_params) + list(expert_params)
                    torch.nn.utils.clip_grad_norm_(all_params, args.max_grad_norm)

                if rank == 0:
                    torch.npu.synchronize() if hasattr(torch, 'npu') else None
                    prof["sync"] += time.time() - t3
                    t4 = time.time()

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                local_window_loss_sum += local_step_loss_sum
                local_window_valid_tokens += local_step_valid_tokens
                rank0_step_loss = accumulated_loss
                local_step_loss_sum = 0.0
                local_step_valid_tokens = 0

                if rank == 0:
                    torch.npu.synchronize() if hasattr(torch, 'npu') else None
                    prof["optim"] += time.time() - t4
                    prof_steps += 1

                if global_step % args.logging_steps == 0:
                    loss_stats = torch.tensor(
                        [local_window_loss_sum, float(local_window_valid_tokens)],
                        dtype=torch.float32,
                        device=device,
                    )
                    if dist.is_initialized():
                        dist.all_reduce(loss_stats, op=dist.ReduceOp.SUM)
                    global_window_loss_sum = float(loss_stats[0].item())
                    global_window_valid_tokens = int(loss_stats[1].item())

                    if rank == 0:
                        lr = scheduler.get_last_lr()[0]
                        global_loss = (
                            global_window_loss_sum / global_window_valid_tokens
                            if global_window_valid_tokens > 0 else float("nan")
                        )
                        logger.info(
                            f"Epoch {epoch+1} | Step {global_step}/{total_steps} | "
                            f"global_loss={global_loss:.4f} | "
                            f"rank0_step_loss={rank0_step_loss:.4f} | "
                            f"window_valid_tokens={global_window_valid_tokens} | "
                            f"LR: {lr:.2e}"
                        )

                        # Bucket stats
                        bucket_stats = torch.cat(
                            [local_bucket_loss_sums, local_bucket_token_counts, local_bucket_sample_counts]
                        )
                    else:
                        bucket_stats = torch.cat(
                            [local_bucket_loss_sums, local_bucket_token_counts, local_bucket_sample_counts]
                        )
                    if dist.is_initialized():
                        dist.all_reduce(bucket_stats, op=dist.ReduceOp.SUM)
                    global_bucket_loss_sums = bucket_stats[:num_loss_buckets]
                    global_bucket_token_counts = bucket_stats[num_loss_buckets:2 * num_loss_buckets]
                    global_bucket_sample_counts = bucket_stats[2 * num_loss_buckets:]
                    if rank == 0:
                        bucket_parts = []
                        for idx, label in enumerate(loss_bucket_labels):
                            token_count = int(global_bucket_token_counts[idx].item())
                            sample_count = int(global_bucket_sample_counts[idx].item())
                            bucket_loss = (
                                global_bucket_loss_sums[idx].item() / token_count
                                if token_count > 0 else float("nan")
                            )
                            bucket_parts.append(
                                f"{label}: loss={bucket_loss:.4f}, samples={sample_count}, tokens={token_count}"
                            )
                        logger.info("[LOSS_BUCKETS] " + " | ".join(bucket_parts))

                    # Profile report
                    if rank == 0 and prof_steps > 0:
                        total_t = sum(prof.values())
                        logger.info(
                            f"[PROF] avg over {prof_steps} optim-steps | "
                            f"data={prof['data']/prof_steps*1000:.1f}ms  "
                            f"forward={prof['forward']/prof_steps*1000:.1f}ms  "
                            f"backward={prof['backward']/prof_steps*1000:.1f}ms  "
                            f"sync={prof['sync']/prof_steps*1000:.1f}ms  "
                            f"optim={prof['optim']/prof_steps*1000:.1f}ms  "
                            f"total={total_t/prof_steps*1000:.1f}ms"
                        )
                        for k in prof: prof[k] = 0.0
                        prof_steps = 0

                    local_window_loss_sum = 0.0
                    local_window_valid_tokens = 0
                    local_bucket_loss_sums.zero_()
                    local_bucket_token_counts.zero_()
                    local_bucket_sample_counts.zero_()
                accumulated_loss = 0.0

                # Save checkpoint
                if global_step % args.save_steps == 0:
                    save_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    save_ep_checkpoint(
                        model, rank, expert_replica_rank, ep_rank, ep_size, save_dir, tokenizer, lora_only=args.use_lora
                    )
                    train_state = {
                        "global_step": global_step,
                        "epoch": epoch,
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    }
                    torch.save(train_state, os.path.join(save_dir, f"train_state_rank{rank}.pt"))
                    if rank == 0:
                        logger.info(f"Checkpoint saved at step {global_step}")

            if rank == 0:
                torch.npu.synchronize() if hasattr(torch, 'npu') else None
                prof_t = time.time()

        epoch_time = time.time() - epoch_start
        if rank == 0:
            logger.info(f"Epoch {epoch+1} completed in {epoch_time:.1f}s")

    # Final save
    save_ep_checkpoint(
        model, rank, expert_replica_rank, ep_rank, ep_size, args.output_dir, tokenizer, lora_only=args.use_lora
    )
    if rank == 0:
        logger.info("[PACKED FORMAT] Training completed!")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
