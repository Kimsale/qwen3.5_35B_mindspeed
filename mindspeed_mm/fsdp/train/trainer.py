from dataclasses import MISSING, asdict, dataclass, field, fields
import faulthandler
import signal
from typing import Any, Callable, Dict, List, Literal, Optional, TypeVar, Union, get_type_hints
import logging
import os
from functools import partial

import torch

from mindspeed.fsdp.utils.log import print_rank, set_log_level
from mindspeed.fsdp.utils.random import set_seed

from mindspeed_mm.fsdp.utils.device import (
    get_dist_comm_backend,
    get_torch_device,
    get_device_type,
    set_accelerator_compatible,
    set_allow_hf32
)
from mindspeed_mm.fsdp.distributed.parallel_state import init_parallel_state, get_parallel_state
from mindspeed_mm.fsdp.models.modelhub import ModelHub
from mindspeed_mm.fsdp.distributed.torch_parallelize import ParallelApplier
from mindspeed_mm.fsdp.features.apply_features import FeaturesApplier
from mindspeed_mm.fsdp.utils.utils import to_empty_if_needed, init_model_weights
from mindspeed_mm.fsdp.data import build_mm_dataloader, build_mm_dataset
from mindspeed_mm.fsdp.data.dataloader.dataloader import PrefetchGradAccDataLoader
from mindspeed_mm.fsdp.optimizer.optimizer import build_optimizer
from mindspeed_mm.fsdp.optimizer.lr_scheduler import build_lr_scheduler
from mindspeed_mm.fsdp.checkpoint.dcp_checkpointer import DistributedCheckpointer
from mindspeed_mm.fsdp.utils.register import import_plugin
from mindspeed_mm.fsdp.params.argument import Arguments, parse_args
from mindspeed_mm.fsdp.tools.memory_profiler import memory_profiler
from mindspeed_mm.fsdp.train.train_engine import TrainEngine
from mindspeed_mm.fsdp.utils.lora_utils import (
    add_lora_to_model,
    freeze_parameters,
    match_target_modules,
    validate_lora_config,
    get_lora_trainable_params,
    print_lora_config,
)
from mindspeed_mm.fsdp.utils.lora_weight_manager import LoraWeightManager


logger = logging.getLogger(__name__)



class Trainer():
    def __init__(self, args: Arguments, model_provider: Optional[Callable] = None, dataloader_provider: Optional[Callable] = None):
        """
        Initialize the trainer with configuration and optional custom providers.

        Args:
            args: Training configuration arguments
            model_provider: Optional custom function to provide the model
            dataloader_provider: Optional custom function to provide the dataloader
        """
        self.args = args

        self.initialize()

        # Initialize model parallelization and feature application
        self.model_parallel_applier = ParallelApplier(args.parallel, args.training)
        self.model_features_applier = FeaturesApplier(args.model)

        # Reset memory profiler
        memory_profiler.reset(args.tools.memory_profile)
        self.lora_weight_manager = None
        # Build core training components
        self.model = self.get_model(model_provider)
        self.optimizer = self.get_optimizer()
        self.lr_scheduler = self.get_scheduler()
        self.train_dataloader = self.get_dataloader() if dataloader_provider is None else dataloader_provider(args)
        self.checkpointer = self.get_checkpointer()
        
        
        # Validate and calculate training iterations
        self._validate_and_set_train_iters(args)

        # Create the training engine
        self.trainer = TrainEngine(
            args, self.train_dataloader, self.model, self.optimizer, self.lr_scheduler, self.checkpointer,
            lora_weight_manager=self.lora_weight_manager
        )

    def _validate_and_set_train_iters(self, args: Arguments):
        # Calculate total training iterations based on epochs if specified
        if args.training.train_epochs is not None:
            if not hasattr(self.train_dataloader, "__len__"):
                raise ValueError(
                    f"Cannot calculate train_iters from epochs because the dataloader "
                    f"(type: {type(self.train_dataloader).__name__}) does not have __len__ attribute. "
                    f"This typically happens when using IterableDataset or streaming data. "
                    f"Please either:\n"
                    f"1. Specify train_iters directly instead of epochs, or\n"
                    f"2. Use a dataloader with a determinable length (regular Dataset), or\n"
                    f"3. Provide a custom dataloader_provider that returns a dataloader with __len__"
                )
            elif len(self.train_dataloader) == 0:
                raise ValueError(
                    f"Cannot calculate train_iters from epochs because the dataloader "
                    f"(type: {type(self.train_dataloader).__name__}) has zero length. "
                    f"This indicates an empty dataset or invalid dataloader configuration. "
                    f"Please check your dataset or dataloader setup."
                )
            else:
                args.training.train_iters = args.training.train_epochs * len(self.train_dataloader)

    def initialize(self):
        """Initialize training environment: logging, random seeds, distributed groups."""
        args: Arguments = self.args
        try:
            faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
        except (RuntimeError, ValueError):
            pass
        print_rank(logger.info, f"Start initializing training environment!!!")

        # Set allow_hf32
        set_allow_hf32(args.training.allow_hf32)

        # Set accelerator compatibility and logging level
        set_accelerator_compatible(get_torch_device())
        set_log_level()
        # Set device index for current process
        torch.accelerator.set_device_index(int(os.environ['LOCAL_RANK']))
        # Set random seeds for reproducibility
        set_seed(args.training.seed, set_deterministic=args.training.use_deter_comp)

        # import plugin and trigger register
        import_plugin(getattr(args.training, "plugin", []))

        # Initialize process group for distributed training
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend=get_dist_comm_backend())

        # Initialize parallel communication groups and mesh
        init_parallel_state(**asdict(args.parallel))

    def get_foundation_model(self):
        """Load the foundation model from the model hub."""
        args: Arguments = self.args
        model = ModelHub.build(args.model, args.training)
        return model

    def get_model(self, model_provider: Optional[Callable] = None):
        """
        Build and prepare the model for training.
        Args:
            model_provider: Optional custom function to provide the model

        Returns:
            Prepared model with parallelization and features applied
        """
        args = self.args
        model = self.get_foundation_model() if model_provider is None else model_provider()

        # Apply LoRA adapters before FSDP2 sharding (if enabled)
        if args.training.lora.enable:
            model = self.enable_lora(model)

        # Apply parallelization strategy and model features
        model = self.model_parallel_applier(model)
        self.model_features_applier(model)

        # Initialize weights on meta device if specified (for memory efficiency)
        if args.training.init_model_with_meta_device:
            manual_ep_cfg = getattr(args.training, "manual_ep_hf_load", None)
            manual_ep_enabled = bool(getattr(manual_ep_cfg, "enable", False))
            if manual_ep_enabled:
                to_empty_if_needed(model, device=get_device_type())
                manual_loader = getattr(model, "load_manual_ep_weights", None)
                if not callable(manual_loader):
                    raise ValueError(
                        "training.manual_ep_hf_load.enable is true, but the selected model "
                        "does not implement load_manual_ep_weights()."
                    )
                manual_loader(args)
                if args.training.lora.enable:
                    self._enforce_lora_only_trainable(model)
                    self._reset_lora_params(model)
            elif args.training.load is None and args.training.load_rank0_and_broadcast:
                raise ValueError("Must set `training.load` when `training.load_rank0_and_broadcast` is True, otherwise the model will be initialized with meta device but no weights will be loaded.")
            elif args.training.load is None and not args.training.load_rank0_and_broadcast:
                to_empty_if_needed(model, device=get_device_type())
                init_model_weights(model)
            elif not args.training.load_rank0_and_broadcast:
                to_empty_if_needed(model, device=get_device_type())
                # FIX: meta_device + DCP load 路径下 to_empty 会把 LoRA 参数初始化为 0
                # (DCP 里没 LoRA 权重,基模型由后续 DCP load 覆盖,但 LoRA 仍是全 0).
                # 标准 LoRA 初始化要求 lora_A 用 kaiming 随机,lora_B 保持 0;两者都 0 会
                # 让 LoRA 输出恒为 0、梯度断流 (grad norm=0,loss 横盘).
                if args.training.lora.enable:
                    self._reset_lora_params(model)

        return model

    @staticmethod
    def _enforce_lora_only_trainable(model: torch.nn.Module) -> None:
        """Keep only LoRA adapter tensors trainable after FSDP/EP wrapping."""
        trainable = 0
        total = 0
        for name, param in model.named_parameters():
            total += 1
            is_lora_param = "lora_" in name and "base_layer" not in name
            param.requires_grad_(is_lora_param)
            if is_lora_param:
                trainable += 1
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            logger.info(f"[LoRA fix] post-FSDP trainable tensors: {trainable}/{total} (LoRA only)")

    @staticmethod
    def _reset_lora_params(model: torch.nn.Module) -> None:
        """重置 LoRA 适配器参数: lora_A 用 kaiming, lora_B 用小随机值.

        注: 标准 LoRA 用 B=0,但在本框架 (FSDP2 composable + NPU bf16/fp32 混精度
        + activation checkpoint) 下,B=0 时 lora_out 恒为 0,反向传播链会因数值/计算图
        优化退化(grad norm=0,LoRA 完全不更新). 经实验验证,改用 small normal (std=0.01)
        既能让梯度链正常流动,对训练初期影响也极小(初始 |scaling*B*A*x|≈0.01,远小于
        base_layer 输出量级)."""
        import math
        from torch.distributed.tensor import DTensor
        n_a, n_b = 0, 0
        for name, param in model.named_parameters():
            if "lora_A" not in name and "lora_B" not in name:
                continue
            with torch.no_grad():
                # FSDP2 下 LoRA 参数是 DTensor,需要拿 local 张量初始化
                tgt = param.data.to_local() if isinstance(param.data, DTensor) else param.data
                if "lora_A" in name:
                    # PEFT 默认 a=sqrt(5) 的 kaiming_uniform_
                    torch.nn.init.kaiming_uniform_(tgt, a=math.sqrt(5))
                    n_a += 1
                else:  # lora_B
                    # 关键: B 用小随机值代替 0,避免 FSDP2+NPU 下梯度链退化
                    torch.nn.init.normal_(tgt, mean=0.0, std=0.01)
                    n_b += 1
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            logger.info(f"[LoRA fix] re-initialized lora_A={n_a} (kaiming), lora_B={n_b} (small_normal std=0.01)")

    def enable_lora(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Enable LoRA fine-tuning by injecting LoRA adapters into model.
        
        This method should be called before FSDP2 sharding to ensure
        LoRA parameters are properly distributed across GPUs.
        
        Args:
            model: The PyTorch model to inject LoRA adapters into.
            
        Returns:
            The model with LoRA adapters injected.
            
        Raises:
            ImportError: If PEFT library is not installed.
            ValueError: If LoRA configuration is invalid.
        """
        lora_config = self.args.training.lora
        
        print_rank(logger.info, "Enabling LoRA fine-tuning...")
        
        # Validate LoRA configuration
        try:
            validate_lora_config(
                rank=lora_config.rank,
                alpha=lora_config.alpha,
                target_modules=lora_config.target_modules,
                dropout=lora_config.dropout,
                init_lora_weights=lora_config.init_lora_weights,
            )
        except ValueError as e:
            raise ValueError(f"Invalid LoRA configuration: {e}") from e
        
        # Match target modules using wildcard patterns
        matched_modules = match_target_modules(model, lora_config.target_modules)
        
        if not matched_modules:
            raise ValueError(
                f"No modules matched target_modules: {lora_config.target_modules}. "
                f"Please check your model architecture and target_modules configuration."
            )
        
        print_rank(logger.info, f"Matched {len(matched_modules)} modules for LoRA:")
        for module_name in matched_modules[:5]:
            print_rank(logger.info, f"  - {module_name}")
        if len(matched_modules) > 5:
            print_rank(logger.info, f"  ... and {len(matched_modules) - 5} more")
        
        # Freeze base model parameters
        freeze_parameters(model)
        
        # Inject LoRA adapters
        model = add_lora_to_model(
            model=model,
            lora_rank=lora_config.rank,
            lora_alpha=lora_config.alpha,
            lora_target_modules=matched_modules,
            lora_dropout=lora_config.dropout,
            init_lora_weights=lora_config.init_lora_weights,
            pretrained_lora_path=lora_config.pretrained_lora_path,
            lora_target_modules_support=lora_config.lora_target_modules_support,
        )
        
        # Get LoRA parameter statistics
        trainable_params, total_params, stats_dict = get_lora_trainable_params(model)
        
        # Print LoRA configuration summary
        print_lora_config(
            rank=lora_config.rank,
            alpha=lora_config.alpha,
            target_modules=matched_modules,
            dropout=lora_config.dropout,
            init_lora_weights=lora_config.init_lora_weights,
            trainable_params=trainable_params,
            total_params=total_params,
        )
        self.lora_weight_manager = LoraWeightManager(model)
        self.lora_weight_manager.verify_lora_weights()
        
        print_rank(logger.info, "LoRA fine-tuning enabled successfully")
        
        return model

    def get_optimizer(self):
        args = self.args
        """Build optimizer for the model."""
        optimizer = build_optimizer(
            model=self.model,
            lr=args.training.lr,
            betas=(args.training.adam_beta1, args.training.adam_beta2),
            eps=args.training.adam_eps,
            weight_decay=args.training.weight_decay,
            fused=args.training.adam_fused,
            optimizer_type=args.training.optimizer,
        )
        return optimizer

    def get_scheduler(self):
        """Build learning rate scheduler."""
        args = self.args
        lr_scheduler = build_lr_scheduler(
            self.optimizer,
            train_steps=args.training.train_iters,
            lr=args.training.lr,
            lr_min=args.training.lr_min,
            lr_decay_style=args.training.lr_decay_style,
            lr_decay_ratio=args.training.lr_decay_ratio,
            lr_warmup_ratio=args.training.lr_warmup_ratio,
            lr_start=args.training.lr_start,
        )
        return lr_scheduler

    def get_dataloader(self):
        """Build training dataloader with proper parallel partitioning."""
        args = self.args
        print_rank(logger.info, "Prepare data")
        data_config = args.data
        ps = get_parallel_state()

        datasets = build_mm_dataset(data_config.dataset_param)
        dataloader_param = data_config.dataloader_param.to_dict()
        dataloader_param.update(
            {
                "batch_size": args.training.micro_batch_size,
                "seed": args.training.seed,
            }
        )
        build_dataloader = partial(
            build_mm_dataloader,
            dataloader_param=dataloader_param,
            process_group=ps.get_dp_group(),
            dataset_param=data_config.dataset_param,
            model=self.model,
        )
        train_dataloader = build_dataloader(datasets)

        if args.model.loss_cfg.loss_type == "per_token_loss":
            train_dataloader = PrefetchGradAccDataLoader(train_dataloader,
                                                         grad_acc_step=args.training.gradient_accumulation_steps)

        return train_dataloader

    def get_checkpointer(self):
        """Return checkpointing class (can be overridden for different checkpoint formats)."""
        return DistributedCheckpointer

    def train(self):
        """Start the training process."""
        self.trainer.train()


if __name__ == "__main__":
    # Entry point for training script
    args = parse_args(Arguments)
    trainer = Trainer(args=args)
    trainer.train()
