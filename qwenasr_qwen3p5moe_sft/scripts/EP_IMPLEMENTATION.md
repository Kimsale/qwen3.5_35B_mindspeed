# Qwen3.5-35B-A3B MoE Expert Parallel (EP) 改造文档

## 1. 背景与目标

Qwen3.5-35B-A3B 是一个 256-expert MoE 模型（每 token 激活 8 个 expert），bf16 下总参数约 62 GB。在 8×Ascend 910B（64 GB/卡）上无法单卡承载全部参数。

目标：通过 Expert Parallel（EP=8）将 256 个 expert 均匀分配到 8 张卡上，每卡仅持有 32 个 expert，显存从 62 GB 降至约 12.7 GB/卡。

## 2. 模型结构分析

### 2.1 原始模型层次

```
Qwen3_5MoeForCausalLM
├── model (Qwen3_5MoeTextModel)
│   ├── embed_tokens
│   ├── layers × 40 (Qwen3_5MoeDecoderLayer)
│   │   ├── linear_attn (GatedDeltaNet) 或 self_attn (标准 Attention)
│   │   ├── mlp (Qwen3_5MoeSparseMoeBlock)
│   │   │   ├── gate (TopKRouter)  — 路由权重 (256, 2048)
│   │   │   ├── experts (Qwen3_5MoeExperts) — 融合 3D 张量
│   │   │   │   ├── gate_up_proj: (256, 1024, 2048)
│   │   │   │   └── down_proj: (256, 2048, 512)
│   │   │   ├── shared_expert (MLP) — 所有 token 共享
│   │   │   └── shared_expert_gate: Linear(2048, 1)
│   │   ├── input_layernorm
│   │   └── post_attention_layernorm
│   ├── norm
│   └── rotary_emb
└── lm_head
```

### 2.2 关键发现：Expert 存储方式

Expert 权重不是 `nn.ModuleList`，而是**融合的 3D 张量**：

```python
class Qwen3_5MoeExperts(nn.Module):
    gate_up_proj = nn.Parameter(torch.empty(256, 1024, 2048))  # (E, 2I, H)
    down_proj = nn.Parameter(torch.empty(256, 2048, 512))      # (E, H, I)
```

这意味着 EP 分片只需沿 dim=0 切片，不需要拆分 ModuleList。

### 2.3 显存分布

| 组件 | 大小 (bf16) | EP=8 后 |
|------|------------|---------|
| Expert 权重 (40层) | 60.00 GB | 7.50 GB/卡 |
| 非 Expert 权重 (attention, shared_expert, norm, embed) | 2.27 GB | 2.27 GB/卡 (全量复制) |
| **合计** | **62.27 GB** | **~9.77 GB/卡** |

实测加载后 NPU 显存占用约 12.66 GB/卡（含 audio encoder 和 buffer）。

## 3. EP 改造方案

### 3.1 核心思路

```
原始: 每卡持有全部 256 个 expert
EP=8: 每卡持有 32 个 expert (rank0: 0-31, rank1: 32-63, ..., rank7: 224-255)
```

每次 forward：
1. 所有 rank 计算完整路由（gate 权重很小，全量复制）
2. 根据路由结果，通过 all-to-all 将 token 发送到持有目标 expert 的 rank
3. 各 rank 在本地 expert 上计算
4. 通过 all-to-all 将结果发回原始 rank

### 3.2 改造的三个文件

| 文件 | 职责 |
|------|------|
| `model_ep.py` | EP 分片模型定义 + 权重加载 |
| `train_ep.py` | 自定义训练循环 |
| `run_ep.sh` | 8 卡启动脚本 |

## 4. model_ep.py 详解

### 4.1 EPExperts — 本地 Expert 计算

```python
class EPExperts(nn.Module):
    """持有切片后的 expert 3D 张量，forward 逻辑与原始一致。"""
    gate_up_proj: (32, 1024, 2048)  # 只有本地 32 个 expert
    down_proj: (32, 2048, 512)
```

### 4.2 EPSparseMoeBlock — All-to-All 通信

替换原始 `Qwen3_5MoeSparseMoeBlock`，核心流程：

```
Step 1: shared_expert 计算（所有 rank 独立完成，无通信）
Step 2: gate 路由（所有 rank 独立计算，gate 权重全量复制）
Step 3: EP dispatch
  3a. 统计每个 rank 需要发送/接收的 token 数量
  3b. all_to_all_single 交换 count
  3c. 构建 send_buf（按目标 rank 排列 token）
  3d. all_to_all_single 发送 hidden_states + expert_idx + weights
Step 4: 本地 expert 计算
Step 5: all_to_all_single 将结果发回
Step 6: scatter 回原始 token 位置
```

### 4.3 权重分片加载 — 解决 OOM

这是改造中最复杂的部分，经历了多次迭代：

**问题 1：meta device + `.to(device)` 报错**
- 原因：meta tensor 没有实际数据，不能直接 `.to(device)`
- 解决：逐参数设置，最后统一物化剩余 meta 参数

**问题 2：`load_file()` 导致 OOM**
- 原因：`load_file` 一次加载整个 shard 到内存，5GB 的 expert shard 即使切片后底层存储仍占满
- 解决：改用 `safe_open` 逐 tensor 读取

**问题 3：safetensors key 前缀不匹配**
- 原因：checkpoint 保存时用的是多模态模型结构 `model.language_model.layers.X`，但 `Qwen3_5MoeForCausalLM` 期望 `model.layers.X`
- 解决：加载时做 key 重映射 `model.language_model.` → `model.`

**问题 4：visual/mtp 等无关权重**
- 原因：safetensors 文件包含 visual encoder 和 MTP head 的权重，text-only 模型没有这些模块
- 解决：`try/except AttributeError` 跳过不存在的模块

**问题 5：dtype 不匹配**
- 原因：GatedDeltaNet 的 `dt_bias`、`A_log` 等参数初始化为 float32
- 解决：加载完成后统一将所有浮点参数转为 bf16

最终加载流程：

```python
# 1. meta device 创建模型（零显存）
with torch.device('meta'):
    model = Qwen3_5MoeForCausalLM(config)

# 2. 逐 shard、逐 tensor 加载
for shard_file in shard_files:
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            model_key = key.replace("model.language_model.", "model.")
            if "experts" in key:
                tensor = tensor[expert_start:expert_end]  # 切片！
            _set_module_tensor(model, model_key, tensor.to(bf16, device))

# 3. 物化剩余 meta 参数（visual/mtp → 零张量）
# 4. 统一 dtype 为 bf16
```

### 4.4 Checkpoint 保存

保存时通过 `dist.all_gather` 从所有 rank 收集 expert 分片，拼接回完整的 (256, ...) 张量：

```python
gathered = [torch.empty_like(param) for _ in range(ep_size)]
dist.all_gather(gathered, param.data, group=ep_group)
full_param = torch.cat(gathered, dim=0)  # (32*8, ...) = (256, ...)
```

## 5. train_ep.py 详解

### 5.1 梯度同步策略

```
Expert 参数 (gate_up_proj, down_proj):
  → 各 rank 独立更新，不需要 all-reduce
  → 因为每个 rank 只持有不同的 expert，梯度天然不同

非 Expert 参数 (attention, shared_expert, norm, embed, lm_head):
  → 需要 all-reduce 同步梯度
  → 确保所有 rank 的非 expert 参数保持一致
```

```python
def sync_non_expert_gradients(non_expert_params, world_size):
    for param in non_expert_params:
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= world_size
```

### 5.2 训练循环

```
for batch in dataloader:
    loss = model(**batch).loss / gradient_accumulation_steps
    loss.backward()

    if (step + 1) % gradient_accumulation_steps == 0:
        sync_non_expert_gradients(non_expert_params, world_size)
        clip_grad_norm_(all_params, max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```

### 5.3 优化器配置

- AdamW，lr=1e-5，weight_decay=0.01
- Cosine schedule with warmup (50 steps)
- Gradient clipping: max_norm=1.0

## 6. 运行方式

```bash
# 启动 8 卡 EP 训练
bash run_ep.sh

# 监控日志
tail -f logs/train_ep_*.log

# 监控显存
watch -n 1 'npu-smi info'
```

### 6.1 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| EP size | 8 | 每卡 32 个 expert |
| batch_size | 1 | per device |
| gradient_accumulation | 4 | 有效 batch = 1×4×8 = 32 |
| max_seq_length | 512 | |
| learning_rate | 1e-5 | cosine decay |
| warmup_steps | 50 | |
| gradient_checkpointing | 开启 | 节省激活值显存 |

## 7. 实测结果

| 指标 | 值 |
|------|-----|
| NPU 显存占用 | 12.66 GB/卡 |
| 非 Expert 参数量 | 2451.5M |
| 本地 Expert 参数量 | 4026.5M |
| 训练速度 | ~30s/step |
| Step 10 Loss | 13.9841 |
| Step 20 Loss | 13.2967 |
| Step 30 Loss | 11.7803 |

Loss 稳步下降，EP 通信和梯度同步工作正常。

## 8. 与之前方案的对比

| 方案 | 显存/卡 | 训练速度 | 实现复杂度 | 状态 |
|------|---------|---------|-----------|------|
| ZeRO-3 + CPU Offload | ~15-20 GB | 慢（CPU↔GPU传输） | 低 | 可用但慢 |
| DeepSpeed TP 配置 | N/A | N/A | 低 | 不兼容 HF 模型 |
| 手动 EP（本方案） | 12.66 GB | ~30s/step | 高 | **已验证** |

## 9. 已知限制与后续优化方向

1. **all-to-all 通信开销**：当前用 `all_to_all_single` 逐字段发送（hidden_states, expert_idx, weights），可以合并为单次通信
2. **Expert 计算效率**：当前逐 expert 循环计算，可以改为 grouped GEMM 批量计算
3. **数据并行**：当前 EP=world_size，没有额外的数据并行。如果扩展到 16 卡，可以 EP=8 + DP=2
4. **Checkpoint 格式**：当前保存为 PyTorch 格式，可以改为 safetensors 格式以兼容 HF 生态
