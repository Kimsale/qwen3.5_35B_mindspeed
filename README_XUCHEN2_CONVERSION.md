# xuchen2 Qwen3-Omni-30B-A3B MoE 模型转换与训练指南

## 概述

本文档说明如何将 `/data/xuchen2/model/` 下的 **Qwen3-Omni-30B-A3B MoE** 模型转换为 MindSpeed-LLM 26.0.0 格式，并使用完全对齐 hulk 的配置进行 LoRA 微调训练。

## 模型信息

- **源模型**: `/data/xuchen2/model/Qwen3-Omni-30B-A3B-Captioner`
- **架构**: Qwen3-Omni MoE (audio encoder + text LLM)
  - 48 层 Transformer
  - 128 专家 (每次激活 top-8)
  - hidden_size: 2048
  - MoE FFN hidden_size: 768
  - vocab_size: 152064
- **格式**: HuggingFace safetensors (16 个分片文件，共 ~60GB)

## hulk 对齐配置

基于 `/data/sejin/baseline_26/reports/HULK_VS_BASELINE_COMPARISON.md` 的分析，完全对齐 hulk 的配置如下：

### 并行策略
- **TP** (张量并行): 1 (无张量切分)
- **PP** (流水并行): 1 (无流水切分)
- **EP** (专家并行): 8 (128 专家切到 8 卡)
- **CP** (上下文并行): 2 (Ulysses 算法)
- **DP** (数据并行，派生): 4 = 8 / (TP=1 × CP=2)

### LoRA 超参
- **lora-r**: 32 (基线是 16)
- **lora-alpha**: 64 (基线是 32)
- **lora-dropout**: 0.1 (基线是 0)
- **target-modules**: `linear_qkv linear_proj` (仅注意力层，不含 MLP)

### 序列与 Batch
- **seq-length**: 8192 (基线是 4096)
- **max-position-embeddings**: 8192
- **global-batch-size**: 16
- **micro-batch-size**: 1

### 训练超参
- **lr**: 5e-6 (基线是 1.25e-5)
- **min-lr**: 1e-6 (基线是 1.25e-7)
- **lr-warmup-fraction**: 0.0 (基线是 0.01)
- **clip-grad**: 5.0 (基线是 1.0)

### 优化器
- **不使用** `--swap-optimizer` (hulk 用纯 GPU ZeRO-2 分片)
- 保留 `--use-distributed-optimizer` (ZeRO-1 等价)

### MoE 参数 (保持不变)
- num-experts: 128
- moe-router-topk: 8
- moe-ffn-hidden-size: 768
- moe-grouped-gemm: ✓
- moe-permutation-async-comm: ✓
- moe-token-dispatcher-type: alltoall_seq

## 操作步骤

### 步骤 1: 权重转换 (HF → MindSpeed MCore)

```bash
# 执行转换脚本
bash /data/sejin/baseline_26/scripts/convert_xuchen2_qwen3omni_moe.sh
```

**预计耗时**: 20-40 分钟  
**预计输出**: ~60GB (bf16 权重)  
**输出路径**: `/data/sejin/models/Qwen3-Omni-30B-A3B-hulk_tp1_pp1_ep8`

转换脚本会：
1. 加载 CANN 8.5 环境
2. 检查源模型和磁盘空间
3. 调用 `convert_ckpt_v2.py` 进行权重转换
4. 验证输出目录结构

### 步骤 2: 准备训练数据

您需要准备符合 hulk 数据特征的训练数据：

**hulk 数据特征** (参考 HULK_VS_BASELINE_COMPARISON.md):
- 样本数: ~380K+
- mean 长度: ~614 tokens
- 序列长度分布: 21-2048 (中位数 466)
- 格式: MindSpeed IndexedDataset (`.bin` + `.idx`)

**选项 A**: 使用现有测试数据 (仅验证流程)
```bash
export HULK_DATA_PATH="/data/sejin/baseline_26/data_hulk_dist_30k/qwen3_sft_packed_input_ids_document"
```

**选项 B**: 准备真实数据
```bash
# 使用 MindSpeed 的数据预处理工具
cd /data/sejin/third_party/mindspeed-llm-26.0.0
python tools/preprocess_data.py \
  --input your_data.jsonl \
  --output-prefix /data/sejin/baseline_26/data_hulk/qwen3_sft_packed \
  --tokenizer-name-or-path /data/xuchen2/model/Qwen3-Omni-30B-A3B-Captioner \
  --seq-length 8192 \
  --workers 32

export HULK_DATA_PATH="/data/sejin/baseline_26/data_hulk/qwen3_sft_packed_input_ids_document"
```

### 步骤 3: 启动训练

```bash
# 设置数据路径 (如果未设置，使用默认值)
export HULK_DATA_PATH="/path/to/your/data_input_ids_document"

# 启动训练 (默认 60 iters)
bash /data/sejin/baseline_26/scripts/train_xuchen2_hulk_aligned.sh

# 或自定义训练轮数
ITERS=100 bash /data/sejin/baseline_26/scripts/train_xuchen2_hulk_aligned.sh
```

训练日志默认保存到: `/data/sejin/baseline_26/logs/xuchen2_hulk_<timestamp>.log`

### 步骤 4: 监控训练

**实时查看日志**:
```bash
tail -f /data/sejin/baseline_26/logs/xuchen2_hulk_<timestamp>.log | grep -E "(iteration|loss|samples_per_sec)"
```

**监控 NPU 利用率**:
```bash
npu-smi info -l
```

**关键指标**:
- **loss**: 应稳定下降
- **samples_per_sec**: 吞吐率
- **AI Core 利用率**: 目标 ≥70%
- **HBM 占用**: 应在 50-60GB 之间

## 关键差异说明

### vs 当前基线 (Qwen3-30B-A3B-Base)

| 维度 | 当前基线 | xuchen2 转换 | 说明 |
|------|---------|-------------|------|
| 模型来源 | `/data/sejin/models/Qwen3-30B-A3B-Base` | `/data/xuchen2/model/Qwen3-Omni-30B-A3B-Captioner` | xuchen2 是 Omni (含 audio encoder) |
| 原始格式 | HF safetensors (已转换) | HF safetensors | 需要转换 |
| 并行配置 | TP=2/EP=4/CP=1 | TP=1/EP=8/CP=2 | xuchen2 对齐 hulk |
| LoRA rank | 16 | 32 | xuchen2 对齐 hulk |
| 序列长度 | 4096 | 8192 | xuchen2 对齐 hulk |
| swap-optimizer | 使用 (CPU offload) | 不使用 (纯 GPU) | xuchen2 对齐 hulk |

### vs hulk 原始环境

| 维度 | hulk | xuchen2 转换 | 说明 |
|------|------|-------------|------|
| 框架 | Theta (自研) | MindSpeed-LLM 26.0.0 | 框架不同 |
| 模型 | Qwen3-Omni-30B (dense) | Qwen3-Omni-30B-A3B (MoE) | **架构一致** (文档错误已修正) |
| 并行 | TP=1/EP=8/CP=2 | TP=1/EP=8/CP=2 | ✓ 一致 |
| LoRA | r=32/alpha=64/dropout=0.1 | r=32/alpha=64/dropout=0.1 | ✓ 一致 |
| 超参 | lr=5e-6/clip=5.0 | lr=5e-6/clip=5.0 | ✓ 一致 |
| ZeRO | Stage-2 (os_v2) | distributed-optimizer | 等效 ZeRO-1 |

## 故障排查

### 1. 转换失败: "Cannot find config.json"
```bash
# 检查源模型路径
ls -la /data/xuchen2/model/Qwen3-Omni-30B-A3B-Captioner/config.json
# 如果路径不对，修改转换脚本中的 HF_MODEL_DIR
```

### 2. 训练 OOM (显存溢出)
```bash
# 选项 A: 减小 global-batch-size
GBS=8 bash /data/sejin/baseline_26/scripts/train_xuchen2_hulk_aligned.sh

# 选项 B: 启用 swap-optimizer (回退方案)
# 编辑训练脚本，在 OPTIMIZE_ARGS 中添加:
#   --swap-optimizer \
#   --swap-optimizer-times 16 \
```

### 3. CP=2 (Ulysses) 崩溃
如果遇到 context-parallel 相关错误，可能是 CANN 8.5/MindSpeed 26.0.0 对 Ulysses 的支持有问题：

```bash
# 临时回退: CP=1, EP=4 (与当前基线一致)
# 修改训练脚本中的并行配置:
# TP=1, PP=1, EP=4, CP=1
```

### 4. 数据格式错误
```bash
# 检查数据文件
ls -lh /path/to/your/data_input_ids_document.{bin,idx}

# 验证数据可读性
cd /data/sejin/third_party/mindspeed-llm-26.0.0
python -c "
from megatron.core.datasets.indexed_dataset import IndexedDataset
ds = IndexedDataset('/path/to/your/data_input_ids_document', multimodal=False)
print(f'Dataset size: {len(ds)}')
print(f'First sample shape: {ds[0].shape}')
"
```

## 文件清单

### 转换脚本
- `/data/sejin/baseline_26/scripts/convert_xuchen2_qwen3omni_moe.sh`

### 训练脚本
- `/data/sejin/baseline_26/scripts/train_xuchen2_hulk_aligned.sh`

### 输出目录
- 转换后权重: `/data/sejin/models/Qwen3-Omni-30B-A3B-hulk_tp1_pp1_ep8/`
- 训练日志: `/data/sejin/baseline_26/logs/xuchen2_hulk_*.log`
- Checkpoint: `/data/sejin/baseline_26/output/ckpt_xuchen2_hulk/`

### 参考文档
- `/data/sejin/baseline_26/reports/HULK_VS_BASELINE_COMPARISON.md` (hulk 配置对比)
- `/data/sejin/CLAUDE.md` (项目约束)

## 下一步

转换和训练完成后，可以：

1. **性能分析**: 对比 xuchen2 模型与当前基线的性能指标
2. **数据对齐**: 使用真实的 hulk 数据集 (mean ~614 tokens)
3. **超参调优**: 基于 MindSpeed Auto Tuning 自动优选参数
4. **生成报告**: 输出完整的性能评测报告

---

**创建日期**: 2026-06-03  
**CANN 版本**: 8.5.0  
**MindSpeed 版本**: 26.0.0  
**硬件平台**: 8×昇腾 910B
