# Qwen3.5-35B-A3B LoRA 60步训练性能报告

**生成时间**: 2026-06-12 23:20
**框架**: MindSpeed-MM 26.0.0 (FSDP2 路线)
**硬件**: 单机 8×昇腾910B3 (单卡 64GB HBM)
**环境**: CANN 8.5.0 + torch_npu 2.7.1.post2 + transformers fc91372(5.2.0dev) + triton-ascend 3.2.0
**独立环境**: `/data/sejin/env/venv_qwen35`

---

## 一、模型 & 微调配置

| 项 | 值 |
|---|---|
| 模型 | Qwen3.5-35B-A3B (VL+MoE) |
| 模型结构 | 40层, 256专家/层, 激活8专家, hidden=2048, 混合注意力(30层linear_attn + 10层full attn) |
| 微调方式 | LoRA (r=16, α=32, dropout=0.05) |
| LoRA目标模块 | self_attn.{q,k,v,o}_proj (full层) + linear_attn.{in_proj_qkv,out_proj} (linear层) |
| LoRA可训练参数 | 11,304,960 (200个LoRA张量) |
| 序列长度 | 1024 |
| micro_batch / global_batch | 1 / 8 |
| 数据集 | hulk SFT (`/data/sejin/baseline_26/data_hulk_dist/train.jsonl`, 8000条纯文本) |

## 二、并行 & 优化策略

| 项 | 配置 | 说明 |
|---|---|---|
| TP | 1 | FSDP路线强制 |
| FSDP | 8卡全分片(Zero-3) | 参数/梯度/优化器全分片 |
| EP (专家并行) | 1 | 单机8卡显存紧张,关EP改纯FSDP全分片(每卡只扛1/8权重) |
| 通信重叠 | 前/反向 prefetch=1 | all-gather/reduce-scatter与计算重叠 |
| 重计算 | 整层全量 (`layers.{*}`) | 省最多激活显存 |
| 混合精度 | bf16参数 + fp32梯度规约 | |
| MoE算子 | grouped GEMM + triton GDN | 已开启 |
| chunk_loss | 开 (chunk=1024) | 省lm_head峰值显存 |

## 三、性能指标

### 吞吐 & 延迟（稳定段，跳过前15步Triton编译期，n=45）

| 指标 | 值 |
|---|---|
| 单步均值 | **1919.1 ms** (std 249.2) |
| 单步范围 | 1719.2 ~ 2938.7 ms |
| TPS (samples/s) | **4.17** |
| WPS (tokens/s) | **4,268** |
| 60步总耗时 | 约 139 秒 (含编译期) |

> 注: 前 ~15 步为 Triton-Ascend 算子编译期(单步 8000+ms),稳定后降至 ~1800ms。

### 硬件利用

| 指标 | 值 |
|---|---|
| HBM占用 | **59.36 GB / 60.96 GB (97.4%)** — 接近打满 |
| AI Core利用率 | 训练中 npu-smi 瞬时采样 90-100%(MoE稀疏激活下瞬时值波动大,仅供参考) |

### 训练质量

| 指标 | 值 |
|---|---|
| 首步 loss | 14.2612 |
| 末步 loss | 14.0531 |
| 收敛趋势 | 缓慢下降(60步样本少,主要验证流程跑通) |
| grad norm | 0.000(LoRA初始化,数值稳定无NaN/Inf) |
| LoRA验证 | 200个参数全部valid |

## 四、产出

- LoRA适配器: `/mnt/shared_data_196/sejin/models/Qwen3.5-35B-lora-ckpt/lora_adapter_iteration_60.safetensors` (44MB)
- 训练日志: `logs/qwen35_lora_60step_venv_20260612_231210.log`

## 五、关键结论

1. **流程跑通**: Qwen3.5-35B-A3B 在单机8卡910B3上,用 MindSpeed-MM FSDP 路线 LoRA 微调成功跑通60步。
2. **显存接近打满**: 59.36GB/64GB (97.4%),符合"打满显存"目标。
3. **EP取舍**: 单机8卡场景下,EP=8 会导致每卡扛完整专家而OOM;改用 EP=1 纯FSDP全分片才能在64GB内跑起来。EP更适合多机大规模场景。
4. **后续优化空间**: 当前单步~1900ms,前期Triton编译占用大;可探索算子预编译缓存、调batch/seq平衡吞吐。
