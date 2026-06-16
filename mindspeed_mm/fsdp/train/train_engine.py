import logging
import time
from contextlib import nullcontext
from datetime import datetime

import torch

from mindspeed.fsdp.utils.log import print_rank

from mindspeed_mm.fsdp.utils.dtype import get_dtype
from mindspeed_mm.fsdp.distributed.fully_shard_parallel import pregather_fsdp_params
from mindspeed_mm.fsdp.distributed.parallel_state import get_parallel_state
from mindspeed_mm.fsdp.utils.utils import move_to_device, get_time
from mindspeed_mm.fsdp.data.data_utils.utils import build_iterations
from mindspeed_mm.fsdp.optimizer.clip_grad_norm import clip_grad_norm
from mindspeed_mm.fsdp.tools.profiler import Profiler
from mindspeed_mm.fsdp.tools.memory_profiler import memory_profiler
from mindspeed_mm.fsdp.loss.loss_func import build_loss_func
from mindspeed_mm.fsdp.params.argument import Arguments


logger = logging.getLogger(__name__)


class TrainEngine:
    """Training engine that manages the main training loop and operations."""
    def __init__(self, args: Arguments, train_dataloader, model, optimizer, scheduler, checkpointer, lora_weight_manager=None, **kwargs):
        self.args = args

        self.model = model
        self.train_dataloader = train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = scheduler
        self.checkpointer = checkpointer
        self.lora_weight_manager = lora_weight_manager

        # Training state tracking
        self.iteration, self.consumed_train_samples = 0, 0

        # Load checkpoint if specified
        if args.training.load:
            self.iteration, self.consumed_train_samples = self.load()

        self.profiler = Profiler(args.tools.profile)
        self.profiler.start()

    def _perf_timing_enabled(self):
        return bool(getattr(getattr(self.args.training, "perf_timing", None), "enable", False))

    def _perf_timing_cfg(self):
        return getattr(self.args.training, "perf_timing", None)

    def _perf_sync(self):
        timing_cfg = self._perf_timing_cfg()
        if not bool(getattr(timing_cfg, "enable", False)) or not bool(getattr(timing_cfg, "sync", True)):
            return
        if hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.synchronize()

    def _diagnostic_sync(self, phase):
        timing_cfg = self._perf_timing_cfg()
        phases = getattr(timing_cfg, "diagnostic_sync_phases", []) or []
        if phase not in phases:
            return
        if hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.synchronize()

    def _log_micro_phase(self, iteration, accum_step, accum_steps, phase, event):
        timing_cfg = self._perf_timing_cfg()
        if not bool(getattr(timing_cfg, "log_micro_steps", False)):
            return
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return
        print_rank(
            logger.info,
            f"[perf_micro] iter={iteration + 1} accum={accum_step + 1}/{accum_steps} phase={phase} event={event}",
        )

    def _perf_now(self):
        self._perf_sync()
        return time.perf_counter()

    def _perf_elapsed(self, start):
        self._perf_sync()
        return time.perf_counter() - start

    def _collect_token_counts(self, batch_data):
        timing_cfg = self._perf_timing_cfg()
        if not bool(getattr(timing_cfg, "log_tokens", True)):
            return {"input_tokens": 0, "label_tokens": 0, "audio_tokens": 0}

        device = None
        input_tokens = torch.tensor(0, dtype=torch.long)
        label_tokens = torch.tensor(0, dtype=torch.long)
        audio_tokens = torch.tensor(0, dtype=torch.long)

        input_ids = batch_data.get("input_ids")
        if isinstance(input_ids, torch.Tensor):
            device = input_ids.device
            if "attention_mask" in batch_data and isinstance(batch_data["attention_mask"], torch.Tensor):
                input_tokens = batch_data["attention_mask"].sum(dtype=torch.long)
            else:
                input_tokens = torch.tensor(input_ids.numel(), device=device, dtype=torch.long)
            audio_token_id = getattr(self.args.model, "audio_token_id", None)
            if audio_token_id is not None:
                audio_tokens = (input_ids == int(audio_token_id)).sum(dtype=torch.long)

        labels = batch_data.get("labels")
        if isinstance(labels, torch.Tensor):
            if device is None:
                device = labels.device
            label_tokens = (labels != -100).sum(dtype=torch.long)

        if device is None:
            return {"input_tokens": 0, "label_tokens": 0, "audio_tokens": 0}

        stats = torch.stack([
            input_tokens.to(device=device, dtype=torch.long),
            label_tokens.to(device=device, dtype=torch.long),
            audio_tokens.to(device=device, dtype=torch.long),
        ])
        if torch.distributed.is_initialized():
            ps = get_parallel_state()
            torch.distributed.all_reduce(stats, group=ps.get_dp_group())

        return {
            "input_tokens": int(stats[0].item()),
            "label_tokens": int(stats[1].item()),
            "audio_tokens": int(stats[2].item()),
        }

    def average_losses_across_data_parallel_group(self, losses):
        """Reduce a tensor of losses across all GPUs."""
        ps = get_parallel_state()
        averaged_losses = torch.cat(
            [loss.clone().detach().view(1) for loss in losses])
        torch.distributed.all_reduce(averaged_losses,
                                    group=ps.get_dp_group())
        averaged_losses = averaged_losses / \
            torch.distributed.get_world_size(group=ps.get_dp_group())

        return averaged_losses

    def get_batch(self, data_iterator):
        """Generate a batch."""
        if data_iterator is not None:
            batch = next(data_iterator)
        else:
            raise ValueError("Data iterator is None. Unable to retrieve batch.")
        return batch

    def set_loss_func(self, batch_data):
        args = self.args
        if args.model.loss_cfg.loss_type == "raw":
            return

        if args.model.enable_chunk_loss:
            loss_func, loss_mask = build_loss_func(args.model.loss_cfg.loss_type,
                                                 chunk_size=args.model.chunkloss_plan.chunk_size, **batch_data)
        else:
            loss_func, loss_mask = build_loss_func(args.model.loss_cfg.loss_type, chunk_size=None, **batch_data)

        if hasattr(self.model, "loss_function"):
            self.model.loss_function = loss_func
        else:
            setattr(self.model, "loss_function", loss_func)

        batch_data.update(output_router_logits=args.model.loss_cfg.router_aux_loss_coef > 0.0)

    def train_step(self, train_dataloader_iter):
        """Perform a single training step with gradient accumulation."""
        args = self.args
        total_loss = 0
        perf_enabled = self._perf_timing_enabled()
        perf = {
            "get_batch_s": 0.0,
            "move_s": 0.0,
            "loss_setup_s": 0.0,
            "forward_s": 0.0,
            "backward_s": 0.0,
            "input_tokens": 0,
            "label_tokens": 0,
            "audio_tokens": 0,
        }
        # Gradient accumulation
        accum_steps = args.training.gradient_accumulation_steps
        use_no_sync = bool(getattr(args.training, "gradient_accumulation_no_sync", False))
        for accum_step in range(accum_steps):
            # Get current batch data
            self._log_micro_phase(self.iteration, accum_step, accum_steps, "get_batch", "start")
            t0 = self._perf_now() if perf_enabled else None
            batch_data = self.get_batch(train_dataloader_iter)
            self._diagnostic_sync("get_batch")
            if perf_enabled:
                perf["get_batch_s"] += self._perf_elapsed(t0)
            self._log_micro_phase(self.iteration, accum_step, accum_steps, "get_batch", "end")

            # Move input to device and cast precision
            self._log_micro_phase(self.iteration, accum_step, accum_steps, "move", "start")
            t0 = self._perf_now() if perf_enabled else None
            batch_data = move_to_device(batch_data, get_dtype(args.parallel.fsdp_plan.param_dtype) if args.parallel.fsdp_plan.param_dtype else None)
            self._diagnostic_sync("move")
            if perf_enabled:
                perf["move_s"] += self._perf_elapsed(t0)
                token_counts = self._collect_token_counts(batch_data)
                for key, value in token_counts.items():
                    perf[key] += value
            self._log_micro_phase(self.iteration, accum_step, accum_steps, "move", "end")

            # setup loss ctx
            self._log_micro_phase(self.iteration, accum_step, accum_steps, "loss_setup", "start")
            t0 = self._perf_now() if perf_enabled else None
            self.set_loss_func(batch_data)
            self._diagnostic_sync("loss_setup")
            if perf_enabled:
                perf["loss_setup_s"] += self._perf_elapsed(t0)
            self._log_micro_phase(self.iteration, accum_step, accum_steps, "loss_setup", "end")

            sync_context = nullcontext
            if use_no_sync and accum_step < accum_steps - 1 and hasattr(self.model, "no_sync"):
                sync_context = self.model.no_sync

            with sync_context():
                # forward step
                self._log_micro_phase(self.iteration, accum_step, accum_steps, "forward", "start")
                t0 = self._perf_now() if perf_enabled else None
                output = self.model(**batch_data)
                loss = output.loss / accum_steps
                self._diagnostic_sync("forward")
                if perf_enabled:
                    perf["forward_s"] += self._perf_elapsed(t0)
                self._log_micro_phase(self.iteration, accum_step, accum_steps, "forward", "end")

                # Backward
                self._log_micro_phase(self.iteration, accum_step, accum_steps, "backward", "start")
                t0 = self._perf_now() if perf_enabled else None
                loss.backward()
                self._diagnostic_sync("backward")
                if perf_enabled:
                    perf["backward_s"] += self._perf_elapsed(t0)
                self._log_micro_phase(self.iteration, accum_step, accum_steps, "backward", "end")

            total_loss += loss

        # Average loss across data parallel group
        total_loss = self.average_losses_across_data_parallel_group([total_loss])

        if perf_enabled:
            return total_loss, perf
        return total_loss, None

    def train(self):
        """Main training loop."""
        args = self.args

        # Get data iterator
        train_dataloader_iter, _, _ = build_iterations(self.train_dataloader)
        self.model.train()

        # --- Train Loop ---
        curr_step_lr = self.lr_scheduler.get_last_lr()[0]
        while self.iteration < args.training.train_iters:
            # Record memory usage if enabled
            memory_profiler.step()
            start_time = get_time(barrier=True)

            perf_enabled = self._perf_timing_enabled()
            pregather_s = 0.0
            if self.args.parallel.fsdp_plan.pregather:
                t0 = self._perf_now() if perf_enabled else None
                pregather_fsdp_params(self.model)
                if perf_enabled:
                    pregather_s = self._perf_elapsed(t0)

            loss, perf_info = self.train_step(train_dataloader_iter)
            if perf_info is not None:
                perf_info["pregather_s"] = pregather_s

            # Clip gradients when clip_grad>0 and get total grad_norm
            self._log_micro_phase(self.iteration, 0, 1, "clip", "start")
            t0 = self._perf_now() if perf_enabled else None
            grad_norm = clip_grad_norm(self.model, max_norm=args.training.clip_grad, foreach=args.training.clip_grad_foreach)
            self._diagnostic_sync("clip")
            if perf_info is not None:
                perf_info["clip_s"] = self._perf_elapsed(t0)
            self._log_micro_phase(self.iteration, 0, 1, "clip", "end")

            # Update parameters
            self._log_micro_phase(self.iteration, 0, 1, "optimizer", "start")
            t0 = self._perf_now() if perf_enabled else None
            self.optimizer.step()
            self._diagnostic_sync("optimizer")
            if perf_info is not None:
                perf_info["optimizer_s"] = self._perf_elapsed(t0)
            self._log_micro_phase(self.iteration, 0, 1, "optimizer", "end")
            t0 = self._perf_now() if perf_enabled else None
            self.lr_scheduler.step()
            if perf_info is not None:
                perf_info["lr_scheduler_s"] = self._perf_elapsed(t0)
            t0 = self._perf_now() if perf_enabled else None
            self.optimizer.zero_grad()
            if perf_info is not None:
                perf_info["zero_grad_s"] = self._perf_elapsed(t0)
            empty_cache_interval = int(getattr(args.training, "empty_cache_interval", 0) or 0)
            if empty_cache_interval > 0 and (self.iteration + 1) % empty_cache_interval == 0:
                t0 = self._perf_now() if perf_enabled else None
                if hasattr(torch, "npu") and torch.npu.is_available():
                    torch.npu.empty_cache()
                if perf_info is not None:
                    perf_info["empty_cache_s"] = self._perf_elapsed(t0)

            # Stop profiling if enabled
            t0 = self._perf_now() if perf_enabled else None
            self.profiler.step()
            if perf_info is not None:
                perf_info["profiler_s"] = self._perf_elapsed(t0)

            # Update training state
            self.consumed_train_samples += args.training.global_batch_size
            self.iteration += 1

            # Calculate iteration time
            elapsed_time_per_iteration = get_time(barrier=True) - start_time

            # Logging
            if self.iteration % args.training.log_interval == 0:
                self.training_log(
                    self.iteration,
                    elapsed_time_per_iteration,
                    curr_step_lr,
                    self.consumed_train_samples,
                    loss,
                    grad_norm,
                    perf_info
                )

            curr_step_lr = self.lr_scheduler.get_last_lr()[0]

            # Save checkpoint at specified intervals
            if args.training.save and args.training.save_interval > 0 and self.iteration % args.training.save_interval == 0:
                self.save(self.iteration, self.consumed_train_samples)

        # Stop profiling if enabled
        self.profiler.stop()
        memory_profiler.stop()
        # Final save after training completes
        if args.training.save:
            self.save(self.iteration, self.consumed_train_samples)

    def training_log(
        self,
        iteration,
        elapsed_time_per_iteration,
        curr_step_lr,
        consumed_train_samples,
        loss,
        grad_norm,
        perf_info=None,
    ):
        args = self.args
        log_string = f" [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
        log_string += ' iteration {:8d}/{:8d} |'.format(
            iteration, args.training.train_iters)
        log_string += ' consumed samples: {:12d} |'.format(
            consumed_train_samples)
        log_string += ' elapsed time per iteration (ms): {:.1f} |'.format(
            elapsed_time_per_iteration * 1000.0)
        log_string += ' learning rate: {:.6E} |'.format(curr_step_lr)
        log_string += ' global batch size: {:5d} |'.format(args.training.global_batch_size)
        log_string += ' loss: {:.6E} |'.format(loss.item())
        if grad_norm is not None:
            log_string += ' grad norm: {:.3f} |'.format(grad_norm)
        if perf_info is not None:
            total_s = elapsed_time_per_iteration if elapsed_time_per_iteration is not None else 0.0
            input_tokens = perf_info.get("input_tokens", 0)
            label_tokens = perf_info.get("label_tokens", 0)
            audio_tokens = perf_info.get("audio_tokens", 0)
            input_wps = input_tokens / total_s if total_s > 0 else 0.0
            label_wps = label_tokens / total_s if total_s > 0 else 0.0
            audio_wps = audio_tokens / total_s if total_s > 0 else 0.0
            log_string += (
                " perf timing ms: "
                f"pregather={perf_info.get('pregather_s', 0.0) * 1000.0:.1f}, "
                f"get_batch={perf_info.get('get_batch_s', 0.0) * 1000.0:.1f}, "
                f"move={perf_info.get('move_s', 0.0) * 1000.0:.1f}, "
                f"loss_setup={perf_info.get('loss_setup_s', 0.0) * 1000.0:.1f}, "
                f"forward={perf_info.get('forward_s', 0.0) * 1000.0:.1f}, "
                f"backward={perf_info.get('backward_s', 0.0) * 1000.0:.1f}, "
                f"clip={perf_info.get('clip_s', 0.0) * 1000.0:.1f}, "
                f"optimizer={perf_info.get('optimizer_s', 0.0) * 1000.0:.1f}, "
                f"lr_scheduler={perf_info.get('lr_scheduler_s', 0.0) * 1000.0:.1f}, "
                f"zero_grad={perf_info.get('zero_grad_s', 0.0) * 1000.0:.1f}, "
                f"empty_cache={perf_info.get('empty_cache_s', 0.0) * 1000.0:.1f}, "
                f"profiler={perf_info.get('profiler_s', 0.0) * 1000.0:.1f} | "
                f"tokens: input={input_tokens}, label={label_tokens}, audio={audio_tokens} | "
                f"wps: input={input_wps:.1f}, label={label_wps:.1f}, audio={audio_wps:.1f} |"
            )

        print_rank(logger.info, log_string)

    def load(self):
        """Load checkpoint and restore training state."""
        args = self.args
        iteration, consumed_train_samples = 0, 0

        state = {"model": self.model, "extra_state": {}}  # cannot be None
        if not args.training.no_load_optim:
            state["optimizer"] = self.optimizer

        release = self.checkpointer.load(
            path=args.training.load,
            state=state,
            load_rank0_and_broadcast=args.training.load_rank0_and_broadcast,
            load_strict=args.training.load_strict,
        )

        if not release:
            iteration = state["extra_state"]["iteration"]
            consumed_train_samples = state["extra_state"]["consumed_train_samples"]

            self.lr_scheduler.load_state_dict(state["extra_state"]["lr_scheduler"])
            self.train_dataloader.load_state_dict(state["extra_state"]["train_dataloader"])
            if not args.training.no_load_rng:
                if "torch_rng_state" not in state["extra_state"]:
                    print_rank(logger.warning, f"No RNG state found in checkpoint, skipping RNG loading")
                else:
                    torch.set_rng_state(state["extra_state"]["torch_rng_state"])

        # Synchronize all processes after loading
        torch.distributed.barrier()

        return iteration, consumed_train_samples

    def save(self, iteration, consumed_train_samples):
        """Save checkpoint with model, optimizer, and training state."""
        args = self.args
        
        # Handle LoRA save modes
        if args.training.lora.enable:
            if args.training.lora.save_mode == "lora_only":
                # Save only LoRA adapter weights
                if self.lora_weight_manager is not None:
                    self.lora_weight_manager.save_lora_only(
                        save_path=args.training.save,
                        iteration=iteration,
                    )
                return
            elif args.training.lora.save_mode == "full_model":
                # Save full model with LoRA (default behavior)
                pass
        
        # Default save behavior (full model)
        state = {
            "model": self.model,
            "extra_state": {
                "iteration": iteration,
                "consumed_train_samples": consumed_train_samples,
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "train_dataloader": self.train_dataloader.state_dict()
            },
        }
        if not args.training.no_save_optim:
            state["optimizer"] = self.optimizer
        if not args.training.no_save_rng:
            state["extra_state"]["torch_rng_state"] = torch.get_rng_state()
        self.checkpointer.save(args.training.save, state=state, iteration=iteration)

        # Synchronize all processes after saving
        torch.distributed.barrier()
