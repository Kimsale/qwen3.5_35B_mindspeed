#!/usr/bin/env python3
"""诊断 FSDP+LoRA 反向链断裂问题:检查 LoRA 是否在 forward 路径、requires_grad 状态"""
import sys
sys.path.insert(0, "/data/sejin/third_party/mindspeed-mm-26.0.0")
import os
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import torch
from mindspeed_mm.fsdp.train.trainer import Trainer
from mindspeed_mm.fsdp.params.args import Args

# 加载配置
args = Args.load("/data/sejin/third_party/mindspeed-mm-26.0.0/examples/qwen3_5_audio/dist100_config.yaml")
args.training.train_iters = 1  # 只跑1步诊断

# 关闭分布式(单进程诊断)
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"

# 初始化
torch.distributed.init_process_group(backend="gloo")

# 构建 trainer(会自动 enable_lora + freeze base)
trainer = Trainer(args)

print("\n=== 1. LoRA 参数 requires_grad 状态 ===")
lora_params = [(n, p) for n, p in trainer.model.named_parameters() if "lora" in n.lower()]
base_params = [(n, p) for n, p in trainer.model.named_parameters() if "lora" not in n.lower()]

print(f"LoRA 参数: {len(lora_params)} 个")
requires_grad_true = sum(1 for _, p in lora_params if p.requires_grad)
print(f"  requires_grad=True: {requires_grad_true}")
print(f"  requires_grad=False: {len(lora_params) - requires_grad_true}")
if requires_grad_true > 0:
    print("  ✅ LoRA 参数 requires_grad 正确")
else:
    print("  ❌ LoRA 参数全被冻结!")

print(f"\n基础模型参数(前10个抽样):")
for n, p in base_params[:10]:
    print(f"  {n[:60]:60s} requires_grad={p.requires_grad}")

print(f"\n=== 2. LoRA 模块是否在 forward 路径 ===")
# 找一个注入了 LoRA 的层
target_layer_name = "model.language_model.layers.11.self_attn.q_proj"
target = None
for name, module in trainer.model.named_modules():
    if name == target_layer_name:
        target = module
        break

if target is None:
    print(f"❌ 找不到 {target_layer_name}")
else:
    print(f"✅ 找到 {target_layer_name}: {type(target)}")
    print(f"   dir(module)[:20]: {dir(target)[:20]}")
    # 检查是否是 PEFT 包装的 LoRA 层
    if hasattr(target, "base_layer") and hasattr(target, "lora_A"):
        print(f"   ✅ 是 PEFT LoRA 层(有 base_layer 和 lora_A)")
        print(f"   lora_A shape: {target.lora_A['default'].weight.shape}")
        print(f"   lora_B shape: {target.lora_B['default'].weight.shape}")
    elif "Linear" in str(type(target)):
        print(f"   ❌ 是普通 Linear,LoRA 注入失败!")
    else:
        print(f"   ⚠️  未知类型")

print(f"\n=== 3. forward 一个 batch 检查梯度流 ===")
# 构建一个 dummy batch
trainer.model.train()
dummy_batch = next(iter(trainer.train_dataloader))
# 转到设备
from mindspeed_mm.fsdp.utils.device import get_device_type, move_to_device
dummy_batch = move_to_device(dummy_batch, None)

# forward
output = trainer.model(**dummy_batch)
loss = output.loss
print(f"✅ forward 成功, loss={loss.item():.4f}")

# backward
loss.backward()
print(f"✅ backward 成功")

# 检查 LoRA 参数是否有 .grad
lora_with_grad = [(n, p.grad.norm().item() if p.grad is not None else None)
                   for n, p in lora_params[:5]]
print(f"\nLoRA 参数 grad 抽样(前5个):")
for name, grad_norm in lora_with_grad:
    if grad_norm is None:
        print(f"  {name[:60]:60s} grad=None ❌")
    elif grad_norm == 0:
        print(f"  {name[:60]:60s} grad_norm=0.0000 (有grad但为0)")
    else:
        print(f"  {name[:60]:60s} grad_norm={grad_norm:.4e} ✅")

# 统计
total_with_grad = sum(1 for _, p in lora_params if p.grad is not None)
total_nonzero_grad = sum(1 for _, p in lora_params if p.grad is not None and p.grad.norm() > 1e-9)
print(f"\n全部 {len(lora_params)} 个 LoRA 参数:")
print(f"  有 .grad 的: {total_with_grad}")
print(f"  grad norm>1e-9 的: {total_nonzero_grad}")

if total_with_grad == 0:
    print("❌ 所有 LoRA 参数都没有 .grad → 反向链完全断了")
elif total_nonzero_grad == 0:
    print("⚠️  所有 LoRA 参数 grad=0 → 反向有、但全零(可能是数值问题或初始化问题)")
else:
    print("✅ LoRA 参数有非零梯度 → 反向链正常")
