# Qwen3.5-35B-A3B MoE EP 改造全记录

> 从零实现 Expert Parallel 训练的完整踩坑、优化、加速记录

---

## 一、项目背景

| 项目 | 详情 |
|------|------|
| 模型 | Qwen3.5-35B-A3B (256 experts, top-8, 40层) |
| 任务 | 语音翻译 SFT (AudioEncoder + MoE LLM) |
| 硬件 | 8 × Ascend 910B (64 GB HBM/卡) |
| 模型大小 | ~62 GB (bf16), Expert占60GB, 非Expert占2.3GB |
| 核心挑战 | 单卡64GB无法容纳62GB模型 |

### 改造前基线 (ZeRO-3 + CPU Offload)

原始方案使用 DeepSpeed ZeRO-3 + CPU Offload（optimizer + param），通过 HF Trainer 驱动。
日志: `logs/train_20260325_145358.log`

| 配置 | 值 |
|------|-----|
| 框架 | HF Trainer + DeepSpeed ZeRO-3 |
| batch_size/卡 | 2 |
| gradient_accumulation | 4 |
| 有效batch_size | 2×4×8 = 64 |
| gradient_checkpointing | 开启 |
| learning_rate | 1e-5 |
| 平均速度 | **110.0 秒/step** |
| 总步数 | 69 |
| 总耗时 | **126.5 分钟** |

**Loss 曲线 (ZeRO-3)**:

| Step | Loss | 说明 |
|------|------|------|
| 10 | 446.1 | HF Trainer累积格式 |
| 20 | 368.5 | |
| 30 | 293.5 | |
| 40 | 217.8 | |
| 50 | 170.7 | |
| 60 | 132.5 | |

> 注: ZeRO-3的loss是HF Trainer格式（每10步累积的平均值），与EP的per-step loss尺度不同

---

## 二、踩坑记录

### 坑1: DeepSpeed TP 配置格式错误

**现象**: `pydantic ValidationError: Extra inputs are not permitted`

**原因**: DeepSpeed的`tensor_parallel`配置结构是嵌套的：
```json
// 错误写法
{"tensor_parallel": {"enabled": true, "tp_size": 2}}
{"tensor_parallel": {"tp_size": 2}}

// 正确写法
{"tensor_parallel": {"tp": {"tp_size": 2}}}
```
`TPTrainingConfig`的顶层字段是`tp`(TPConfig对象)、`autotp_size`等，不接受`tp_size`和`enabled`。

**解决**: 查看DeepSpeed源码`deepspeed/runtime/tensor_parallel/config.py`确认正确格式。

---

### 坑2: DeepSpeed TP 不支持 HuggingFace 模型训练

**现象**: 配置格式修正后仍然OOM（每卡60GB）

**原因**: DeepSpeed的TP功能（`TPTrainingConfig`）是为**推理**设计的AutoTP，不能自动分片HF模型的训练权重。TP在训练时需要模型原生支持（如Megatron），HF的`Qwen3_5MoeForCausalLM`没有TP分片逻辑。

**教训**: DeepSpeed config里写TP配置 ≠ 模型权重会被自动分片。

---

### 坑3: Expert 不可下标访问

**现象**: `TypeError: 'Qwen3_5MoeExperts' object is not subscriptable`

**原因**: Qwen3.5的expert权重**不是**`nn.ModuleList`，而是**融合的3D张量**：
```python
class Qwen3_5MoeExperts(nn.Module):
    gate_up_proj = nn.Parameter(torch.empty(256, 1024, 2048))  # (E, 2I, H)
    down_proj = nn.Parameter(torch.empty(256, 2048, 512))      # (E, H, I)
```
所以`experts[i]`会报错，必须用`experts.gate_up_proj[i]`来索引。

**解决**: EP分片时沿dim=0切片3D张量，而非拆分ModuleList。

---

### 坑4: 模型加载时OOM

**现象**: 每个rank都尝试加载完整62GB模型 → 64GB卡OOM

**原因**: `Qwen3_5MoeForCausalLM.from_pretrained()`会在每个进程中加载完整权重。

**解决**: 使用`torch.device('meta')`创建模型骨架（零显存），然后逐shard加载权重，expert张量只加载本地分片。

---

### 坑5: Meta tensor 不能 `.to(device)`

**现象**: `NotImplementedError: Cannot copy out of meta tensor; no data!`

**原因**: meta device上的参数没有实际数据，直接调用`model.to(device)`会尝试移动所有参数，包括未加载的meta参数。

**解决**: 不用`model.to(device)`，改为逐参数设置：
```python
_set_module_tensor(model, key, tensor.to(device))
```
最后用`_materialize_meta_tensors()`将剩余meta参数（visual/mtp等无关组件）初始化为零。

---

### 坑6: `load_file()` 导致显存泄漏

**现象**: 即使切片了expert，显存仍然60GB

**原因**: `safetensors.torch.load_file()`一次性加载整个shard文件到内存。即使对tensor做切片，底层storage仍持有完整数据，`.to(device)`时会拷贝完整存储到NPU。

**解决**: 改用`safetensors.safe_open()`逐tensor读取：
```python
with safe_open(shard_path, framework="pt", device="cpu") as f:
    for key in f.keys():
        tensor = f.get_tensor(key)
        sliced = tensor[start:end].contiguous().clone()  # 断开原始存储引用
        _set_module_tensor(model, key, sliced.to(device))
        del tensor  # 立即释放
```

---

### 坑7: Safetensors key 前缀不匹配

**现象**: `AttributeError: 'Qwen3_5MoeTextModel' object has no attribute 'language_model'`

**原因**: checkpoint保存自多模态模型，key前缀为`model.language_model.layers.X`，但`Qwen3_5MoeForCausalLM`的结构是`model.layers.X`（没有`language_model`层级）。

**解决**: 加载时做key重映射：
```python
model_key = key.replace("model.language_model.", "model.")
```

---

### 坑8: Visual/MTP 权重导致 AttributeError

**现象**: `AttributeError: 'Qwen3_5MoeTextModel' object has no attribute 'visual'`

**原因**: safetensors文件包含visual encoder和MTP head的权重（来自完整多模态模型），但text-only的CausalLM没有这些模块。

**解决**: `try/except AttributeError`跳过不存在的模块：
```python
try:
    _set_module_tensor(model, model_key, tensor.to(device))
except (AttributeError, IndexError):
    del tensor
    continue
```

---

### 坑9: dtype 不匹配

**现象**: `RuntimeError: Input type (float) and bias type (c10::BFloat16) should be the same`

**原因**: GatedDeltaNet的`dt_bias`和`A_log`等参数初始化为float32，而模型其他部分是bf16。

**解决**: 加载完权重后统一转换所有浮点参数为bf16：
```python
for name, param in model.named_parameters():
    if param.dtype != torch.bfloat16 and param.is_floating_point():
        setattr(module, parts[-1],
                nn.Parameter(param.data.to(torch.bfloat16), ...))
```

---

### 坑10: `_set_module_tensor` 不支持数字索引

**现象**: `AttributeError: 'Qwen3_5MoeTextModel' object has no attribute '0'`

**原因**: `nn.ModuleList`的子模块通过整数索引访问（`module[0]`），不能用`getattr(module, '0')`。

**解决**: 路径遍历时检查数字：
```python
for part in parts[:-1]:
    if part.isdigit():
        module = module[int(part)]
    else:
        module = getattr(module, part)
```

---

## 三、EP 实现方案

### 架构: EP=8 (纯 Expert Parallel)

```
8卡, 每卡持有 256/8 = 32 个 expert
非expert参数(attention, shared_expert, norm, embed): 全量复制
```

### 核心文件

| 文件 | 职责 |
|------|------|
| `model_ep.py` | EP分片模型 + 权重加载 |
| `train_ep.py` | 自定义训练循环 |
| `run_ep.sh` | 8卡启动脚本 |

### 关键类

- **`EPExperts`**: 持有切片后的expert 3D张量`(32, 1024, 2048)`，forward与原始一致
- **`EPSparseMoeBlock`**: 替换原始MoE块，实现all-to-all dispatch/combine
- **`EPSpeechTranslationModel`**: 组合AudioEncoder + EP LLM
- **`load_llm_with_ep()`**: meta device创建 + safe_open逐tensor加载 + key重映射

### EP Forward 流程

```
1. shared_expert计算 (所有rank独立, 无通信)
2. gate路由 (所有rank独立计算完整路由)
3. EP dispatch:
   3a. flatten (num_tokens, top_k) → (N,)
   3b. argsort按目标rank排序
   3c. bincount统计send_counts
   3d. pack hidden+expert_idx+weight为一个buffer
   3e. all_to_all_single发送
4. 本地expert计算 (torch.bmm批量)
5. all_to_all_single回传结果
6. index_add_散布回原位
7. expert_out + shared_out
```

### 梯度同步策略

```
Expert参数 (gate_up_proj, down_proj):
  → 各rank独立更新, 不需要all-reduce
  → 每个rank只持有不同的expert, 梯度天然不同

非Expert参数 (attention, shared_expert, norm, embed, lm_head):
  → flatten所有梯度为单buffer → 单次all-reduce → unflatten
```

---

## 四、性能优化记录

### 第一轮: 基础EP实现

| 项目 | 值 |
|------|-----|
| 速度 | 30.0 秒/step |
| 显存 | 12.66 GB/卡 |
| 状态 | 基准 |

**瓶颈**: 逐expert循环 + Python for-loop构建dispatch buffer + 逐参数all-reduce

---

### 第二轮: 向量化 + 合并通信

**优化内容**:
1. `_ep_forward`: Python for-loop → `torch.argsort` + `torch.bincount`向量化
2. all-to-all: 3次独立通信 → pack为1次
3. 梯度同步: 逐参数all-reduce → flatten单次all-reduce

| 项目 | 值 | 加速 |
|------|-----|------|
| 速度 | 22.5 秒/step | 1.33x |

---

### 第三轮: 关闭GC + 增大batch

**优化内容**:
1. **关闭gradient_checkpointing** → 不再重复计算forward, 约2x加速
2. **batch_size 1→2** → 更好的硬件利用率
3. **gradient_accumulation 4→2** → 减少forward/backward次数

| 项目 | 值 | 加速 |
|------|-----|------|
| 速度 | 5.9 秒/step | 5.1x |

---

### 第四轮: torch.bmm + argsort优化

**优化内容**:
1. `_local_expert_forward`: 逐expert `torch.mm`循环 → pad+stack后`torch.bmm`批量计算全部expert
2. `torch.argsort(int64)` → `torch.argsort(float32)` 避免AiCpu回退（Ascend NPU的ArgSort不支持int64在AiCore执行）

| 项目 | 值 | 加速 |
|------|-----|------|
| 速度 | 5.5 秒/step | 5.5x |
| 最快区间 | 5.2 秒/step | — |

---

### 第五轮: 数据加载优化

**优化内容**:
1. `librosa.load` → `soundfile.read` (快5-10倍)
2. `num_workers 2→4`, `prefetch_factor=4`, `persistent_workers=True`

**结论**: 数据加载不是当前瓶颈，速度无明显提升（5.8s vs 5.5s在误差范围内）。

---

### 第六轮: 全量缓存数据集 (已回退)

尝试将4398个样本全部预处理后缓存到内存。速度无提升，且不适用于大规模数据集，已回退。

---

## 五、最终性能数据

### ZeRO-3 vs EP 全面对比

| 指标 | ZeRO-3 + CPU Offload | EP=8 (最终) |
|------|----------------------|-------------|
| 框架 | HF Trainer + DeepSpeed | 自定义训练循环 |
| 平均速度 | 110.0 秒/step | **5.5 秒/step** |
| 加速比 | 1.0x | **19.4x** |
| 1 epoch 耗时 | 126.5 分钟 | **13.0 分钟** |
| 总步数/epoch | 69 | 137 |
| 有效batch_size | 64 | 32 |
| 显存占用 | ZeRO-3分片+CPU offload | 12.66 GB/卡 |
| gradient_checkpointing | 开启 | 关闭 |
| Loss (step 50) | 170.7 (累积) | 9.66 (per-step) |

> 两者有效batch_size不同(64 vs 32)，loss格式不同（HF Trainer累积 vs per-step），loss数值不直接可比。
> 速度对比为同硬件上的wall-clock time。

### EP Loss 曲线

| Step | Loss |
|------|------|
| 10 | 14.25 |
| 20 | 13.53 |
| 30 | 12.11 |
| 40 | 10.94 |
| 50 | 9.66 |
| 60 | 8.31 |
| 70 | 7.50 |
| 80 | 6.84 |
| 90 | 6.40 |
| 100 | 6.42 |
| 110 | 6.34 |
| 120 | 6.17 |
| 130 | 6.21 |

### EP 速度演进（同配置下的代码优化）

```
110.0s ──┐  ZeRO-3 + CPU Offload (HF Trainer, 基线)
         │
 30.0s ──┤  EP=8 初版 (batch=1, grad_accum=4, grad_ckpt=ON)
         │  向量化dispatch + 合并all-to-all + 扁平化梯度同步
 22.5s ──┤  (vs EP初版 1.33x)
         │  关闭gradient_checkpointing + batch_size 1→2 + grad_accum 4→2
  5.9s ──┤  (vs EP初版 5.1x)
         │  torch.bmm批量expert + argsort转float32
  5.5s ──┘  (vs EP初版 5.5x, vs ZeRO-3 19.4x)
```

---

## 六、当前瓶颈分析

5.5秒/step的时间分布（估算）：

| 环节 | 耗时占比 | 说明 |
|------|---------|------|
| EP all-to-all通信 | ~35% | 40层 × 2次round-trip = 80次all-to-all/forward |
| Expert计算 (bmm) | ~25% | 40层 × bmm(32, max_tokens, hidden) |
| Attention计算 | ~25% | 30层GatedDeltaNet(torch慢速) + 10层标准Attention |
| 梯度同步 | ~10% | 非expert参数all-reduce |
| 其他 | ~5% | 数据加载、optimizer.step等 |

### 进一步优化方向

1. **flash-linear-attention + triton-ascend**: 加速GatedDeltaNet (~25%的时间)
   - 风险: triton-ascend实验性质, causal_conv1d不支持NPU
   - 收益: 如果可用, 预计10-15%加速

2. **通信-计算overlap**: 异步all-to-all与shared_expert计算重叠
   - 难度: 高, 需要重构forward逻辑
   - 收益: 预计5-10%

3. **减少EP通信层数**: 只对部分层做EP, 其他层用小expert
   - 会改变模型语义

4. **更快的互联**: NVLink/NVSwitch vs HCCL
   - 硬件限制, 代码层面无法优化

---

## 七、文件清单

### 核心文件 (当前使用)
```
model_ep.py        # EP分片模型 + 权重加载
train_ep.py        # 自定义训练循环
train.py           # 数据集定义 (SpeechTranslationDataset, DataCollator)
run_ep.sh          # 8卡启动脚本
```

### 文档
```
EP_IMPLEMENTATION.md  # EP改造技术文档
EP_FULL_RECORD.md     # 本文 (完整踩坑+优化记录)
```

### 历史文件 (可清理)
```
model.py              # 原始模型 (ZeRO-3方案)
train_tp_ep.py        # 早期TP+EP尝试 (已废弃)
ep_only_model.py      # 早期EP尝试 (已废弃)
sharded_model.py      # 早期分片尝试 (已废弃)
train_ep_profile.py   # 性能profiling脚本
run_ep_fast.sh        # 快速训练脚本 (路径问题)
run_train_tp_ep.sh    # ZeRO-3方案启动脚本
ds_config_tp_ep.json  # ZeRO-3配置
```

---

## 八、使用方法

```bash
# 启动EP训练 (8卡, EP=8, 5.5s/step)
bash run_ep.sh

# 监控训练
tail -f logs/train_ep_*.log

# 查看显存
npu-smi info
```

### 关键训练参数
```
EP size:                    8 (每卡32 experts)
batch_size:                 2
gradient_accumulation:      2
有效batch_size:             2 × 2 × 8 = 32
learning_rate:              1e-5 (cosine decay)
warmup_steps:               50
gradient_checkpointing:     关闭 (换速度)
```
