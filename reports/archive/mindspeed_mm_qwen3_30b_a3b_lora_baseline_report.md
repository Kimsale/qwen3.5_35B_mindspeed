# Qwen3-30B-A3B LoRA 微调基线性能报告

## 一、实验环境

### 硬件配置
- **芯片型号**: 昇腾 910B3
- **NPU 数量**: 8 卡
- **显存**: 每卡 65536 MB (64GB HBM)

### 软件版本
- **CANN 版本**: 8.5.0
- **框架**: MindSpeed-LLM 26.0.0
- **Megatron-LM**: Core v0.12.1
- **PyTorch**: 2.7.1
- **torch_npu**: 2.7.1

### 模型配置
- **模型**: Qwen3-30B-A3B (MoE 架构)
- **总参数量**: 30.52B (305.2亿)
- **专家数量**: 128 个
- **Top-K 路由**: 8
- **层数**: 48
- **隐藏层维度**: 2048
- **FFN 维度**: 6144
- **注意力头数**: 32
- **KV 头数**: 4 (GQA)
- **Vocab Size**: 152064

## 二、训练配置

### LoRA 参数
- **LoRA Rank (r)**: 16
- **LoRA Alpha**: 32
- **LoRA Dropout**: 0.0
- **Target Modules**: linear_qkv, linear_proj, linear_fc1, linear_fc2
- **可训练参数**: 135,659,520 (1.356亿)
- **可训练参数占比**: 2.99%

### 并行策略
- **Tensor Parallel (TP)**: 2
- **Pipeline Parallel (PP)**: 1
- **Expert Parallel (EP)**: 4
- **Data Parallel (DP)**: 1
- **Context Parallel (CP)**: 1

### 训练超参数
- **Micro Batch Size**: 1
- **Global Batch Size**: 8
- **序列长度**: 4096
- **学习率**: 1.0e-5
- **最小学习率**: 1.0e-7
- **学习率衰减**: Cosine
- **Warmup 比例**: 0.01
- **Weight Decay**: 0.01
- **Adam Beta1**: 0.9
- **Adam Beta2**: 0.95
- **Gradient Clipping**: 1.0
- **精度**: BF16
- **优化器**: Distributed AdamW

### 算子优化配置
- **Flash Attention**: 启用
- **Fused RMSNorm**: 启用
- **Fused SwiGLU**: 启用
- **Fused Rotary Pos Emb**: 启用
- **Sequence Parallel**: 启用
- **MoE Grouped GEMM**: 启用
- **MoE Permutation Async Comm**: 启用
- **MoE Token Dispatcher**: alltoall_seq

### 数据配置
- **数据格式**: Packed MCore Format
- **数据路径**: /data/sejin/data_hulk_dist_30k_mcore/hulk_sft
- **数据集大小**: ~30K 样本
- **Tokenizer**: Qwen3-30B-A3B-Base

## 三、基线性能指标

### 训练吞吐
基于 iteration 2-24 的稳定阶段统计（排除第1次迭代的初始化开销）：

| 指标 | 数值 |
|------|------|
| **平均单步耗时** | 5.08 秒/iteration |
| **样本吞吐** | 1.575 samples/s |
| **Token 吞吐** | 约 6,450 tokens/s (基于 seq_len=4096) |
| **Global Batch Size** | 8 samples |
| **训练迭代数** | 500 (计划) |

### 显存占用

| NPU Rank | 已分配显存 | 峰值显存 | 保留显存 | 显存占用率 |
|----------|------------|----------|----------|------------|
| Rank 0 | 10,034 MB | 28,179 MB | 28,606 MB | 43.7% |
| Rank 1 | 10,034 MB | 28,179 MB | 28,606 MB | 43.7% |

**说明**:
- 峰值显存约 28GB/卡，占用率 43.7%，距离 CLAUDE.md 要求的 50-60GB 还有较大空间
- 理论显存占用（权重+优化器）: 130,979 MB (约 128GB)
- 实际显存利用较为保守，可进一步增大 batch size 以提升显存占用和吞吐

### Loss 收敛情况

| Iteration | Loss | Learning Rate | Grad Norm |
|-----------|------|---------------|-----------|
| 1 | 0.5613 | 2.00e-06 | 0.309 |
| 5 | 0.5609 | 1.00e-05 | 0.402 |
| 10 | 0.5559 | 9.998e-06 | 0.835 |
| 15 | 0.5068 | 9.990e-06 | 0.375 |
| 20 | 0.5151 | 9.978e-06 | 0.343 |
| 24 | 0.4384 | 9.964e-06 | 0.277 |

**Loss 趋势**: 从 0.5613 下降到 0.4384，呈现良好的收敛趋势，无异常波动，梯度范数稳定。

## 四、性能瓶颈分析

### 1. 显存利用率偏低
- **现象**: 峰值显存仅 28GB/卡 (43.7%)，远低于目标 50-60GB
- **影响**: 吞吐未充分利用硬件资源
- **根因**: Micro Batch Size=1 过小，Global Batch Size=8 偏保守

### 2. 单步耗时构成
- **第1次迭代**: 85.2秒 (包含模型初始化、编译、第一次前向/反向)
- **稳定阶段**: 约 5.08秒/iteration
- **分析**: 初始化开销较大，但稳定后单步耗时合理

### 3. 通信开销
- **并行策略**: TP=2, EP=4
- **通信模式**: AllToAll (MoE), AllReduce (TP)
- **潜在优化**: 可考虑调整 EP 大小，减少 MoE 通信开销

### 4. AI Core 利用率
- **注意**: 日志中未直接显示 AI Core 利用率
- **需要**: 通过 npu-smi 或 profiling 工具进一步采集

## 五、优化建议

### 优先级1: 提升显存占用和吞吐

1. **增大 Micro Batch Size**
   - 当前: 1
   - 建议: 2-4
   - 预期效果: 显存占用提升至 50-60GB，吞吐提升 50-100%

2. **增大 Global Batch Size**
   - 当前: 8
   - 建议: 16-32
   - 预期效果: 进一步提升吞吐，改善训练稳定性

### 优先级2: 算子和通信优化

3. **MoE 专家并行调优**
   - 当前: EP=4
   - 建议: 尝试 EP=2 或 EP=8，对比通信开销
   - 预期效果: 降低 AllToAll 通信延迟

4. **Gradient Accumulation**
   - 当前: 未启用
   - 建议: 配合大 Global Batch Size 启用
   - 预期效果: 在不增加显存的情况下提升有效 batch size

### 优先级3: 启用 MindSpeed Auto Tuning

5. **自动超参优化**
   - 工具: MindSpeed 内置 Auto Tuning
   - 目标: 自动遍历并行策略、算子开关组合
   - 预期效果: 发现最优配置，AI Core 利用率提升至 ≥70%

## 六、下一步工作

### 1. 增大 Batch Size 验证
- 修改 `--micro-batch-size 2` 和 `--global-batch-size 16`
- 重新运行训练，采集性能指标
- 对比显存占用和吞吐提升

### 2. 启用 Profiling
- 使用 MindSpeed 内置 profiling 工具
- 采集 AI Core 利用率、HBM 带宽、算子耗时
- 生成详细性能分析报告

### 3. Auto Tuning 调优
- 配置 Auto Tuning 参数空间
- 自动遍历并行策略和算子组合
- 输出最优配置和性能对比

### 4. 多轮迭代优化
- 基于 profiling 结果识别瓶颈
- 逐项验证优化效果
- 输出完整优化路径文档

## 七、结论

### 基线状态
- ✅ **训练成功启动**: Qwen3-30B-A3B LoRA 微调已成功运行
- ✅ **Loss 正常收敛**: 从 0.56 降至 0.44，无异常
- ✅ **梯度稳定**: 梯度范数在合理范围，无爆炸/消失
- ⚠️ **显存利用偏低**: 28GB/64GB (43.7%)，有较大优化空间
- ⚠️ **吞吐待提升**: 1.575 samples/s，可通过增大 batch size 优化

### 优化潜力
- **显存优化空间**: 可提升至 50-60GB (目标显存占用率 78-94%)
- **吞吐提升潜力**: 预期 2-3倍 (通过增大 batch size + 算子优化)
- **AI Core 利用率**: 需 profiling 采集，目标 ≥70%

### 符合项目要求
- ✅ 使用 CANN 8.5.0 环境
- ✅ 固定 Qwen3-30B-A3B 模型和 LoRA 配置
- ✅ 使用指定数据集和 checkpoint
- ✅ 训练稳定，无 OOM 或崩溃
- ⏳ 显存占用需进一步提升至 50-60GB
- ⏳ AI Core 利用率待 profiling 验证

---

**报告生成时间**: 2026-06-04 15:10
**训练状态**: 进行中 (Iteration 24/500)
**日志文件**: /data/sejin/logs/train_qwen3_30b_a3b_lora_20260604_150506.log
**Checkpoint 保存路径**: /data/sejin/output/qwen3_30b_a3b_lora_mindspeed_mm
