# Pad 格式调优 + MC2 分支 (mc2-perf-eval)

> **分支用途**: Pad 格式 38 轮配置扫描 + MC2 通信-计算重叠代码接通  
> **核心成果**: Pad 最优稳定配置 WPS 1133 (HBM 56.4GB)，MC2 代码已接通（待实测）  
> **完整报告**: [`reports/qwen35_audio_manual_ep8_perf_tuning_20260616.md`](reports/qwen35_audio_manual_ep8_perf_tuning_20260616.md)

---

## 一、Pad 格式调优成果（38 轮配置扫描）

### 最优配置矩阵

| 配置 | WPS | HBM/卡 | 单步耗时 | 状态 | 备注 |
|------|-----|--------|----------|------|------|
| **pad1536_nosync** | 1133 | 56.4 GB | 4.89s | ✅ 80步稳定 | **最优稳定配置**（HBM 55-60GB 目标） |
| pad1408_nosync | 1158 | 54.6 GB | 4.79s | ✅ 80步稳定 | WPS 略高，HBM 未达目标 |
| pad1280_current | 1296 | 51.9 GB | 4.28s | ✅ 80步稳定 | WPS 最高但 HBM 低于目标 |
| pad1024_pregather_nosync | 1415 | 48.8 GB | 3.92s | ✅ 80步稳定 | 历史最高 WPS（HBM 不达标） |

### 38 轮扫描覆盖范围

- **Padding 长度**: 128/256/512/1024/1152/1216/1248/1280/1344/1408/1536/2048
- **Bucket 策略**: bucket16/32/64, chunk512, pregather
- **同步策略**: sync / nosync
- **Recompute**: rc_on / rc_off
- **Batch size**: mbs=1 (稳定) / mbs=2 (23 次尝试全部 hang)

**扫描结论**:
- **生产推荐**: `pad1536_nosync` (WPS 1133, HBM 56.4GB, 严格满足 HBM 55-60GB 目标)
- **历史最高 WPS**: `pad1024_pregather_nosync` (WPS 1415, HBM 48.8GB, 不满足 HBM 目标)
- **mbs=2 结论**: 外部环境级问题，23 次调参组合均挂在 SIGTERM，暂不继续

详细数据见 [`reports/qwen35_audio_manual_ep8_perf_tuning_20260616.md`](reports/qwen35_audio_manual_ep8_perf_tuning_20260616.md)。

---

## 二、MC2 通信-计算重叠

### 状态

| 项 | 状态 | 说明 |
|---|---|---|
| **算子可用性** | ✅ 已探测 | `npu_alltoallv_gmm` / `npu_gmm_alltoallv` 在 CANN 8.5 可用 |
| **代码接通** | ✅ 已完成 | `expert_parallel.py` + `modeling_qwen3_5_moe.py:946` 支持 `dispatcher: mc2` |
| **音频 EP8 实测** | ⏳ 待完成 | 数学一致性验证 + 性能复测 |

### 预期收益（理论分析）

| 指标 | Pad 基线 (pad1536 fused) | MC2 预期 | 预期收益 |
|---|---|---|---|
| **WPS** | 1133 | 1230-1290 | +10~+15% |
| **单步耗时** | 4.89s | 4.3-4.5s | -10~-15% |
| **AI Core 利用率** | 23.46% | 25-28% | +2~5% |
| **HBM 占用** | 56.4 GB | 55-60GB | 维持 |

**收益来源**: forward (2.66s) 和 backward (1.62s) 阶段的 AllToAll 通信被掩盖到专家 GEMM 后面，节省通信暴露时间。

详见 [`reports/moe_optimization_strategy_from_blog_20260616.md`](reports/moe_optimization_strategy_from_blog_20260616.md) Phase 1。

---

## 三、快速开始

### 1. 环境准备

```bash
# 加载 CANN 8.5 环境
source scripts/env_cann85.sh
source /data/sejin/env/venv_cann85/bin/activate

# 验证环境
npu-smi info | head -5
python3 -c "import torch_npu; print(torch_npu.__version__)"  # 应输出 2.6.0.post1+cann85
```

### 2. Pad 最优配置（已验证）

```bash
# 配置文件
cd /data/sejin/third_party/mindspeed-mm-26.0.0
config=examples/qwen3_5_audio/perf_tuning/ep8_mbs1_ga4_rc_off_pad1536_nosync.yaml

# 启动训练
bash examples/qwen3_5_audio/scripts/train_qwen35_audio.sh \
  --config $config \
  --max_steps 80
```

### 3. MC2 配置（待实测）

```bash
# MC2 配置文件
config=examples/qwen3_5_audio/perf_tuning/ep8_mbs1_ga4_rc_off_pad1536_nosync_mc2.yaml

# 启动训练（需验证数学一致性）
bash examples/qwen3_5_audio/scripts/train_qwen35_audio.sh \
  --config $config \
  --max_steps 80
```

**验证项**:
- Loss 轨迹与 fused 对比（允许 ±0.05 波动）
- WPS 是否达到预期（1230-1290）
- 80 步稳定完成，无 hang/OOM/NaN

---

## 四、配置说明

### Pad 最优配置（pad1536_nosync）

```yaml
# data.dataloader_param.collate_param
collate_param:
  pad_to_multiple_of: 1536            # 填充到 1536
  ignore_pad_token_for_loss: true

# parallel
parallel:
  recompute: false
  expert_parallel_size: 8
  ep_plan:
    apply_modules:
    - model.language_model.layers.{*}.mlp.experts
    dispatcher: fused                 # 当前生产配置
    pre_gather_experts: false
    no_sync_experts: true             # ← 关键优化

# memory
memory:
  micro_batch_size: 1
  gradient_accumulation_steps: 4
  max_seq_length: 1536
```

### MC2 配置（待实测）

```yaml
# parallel.ep_plan 段
ep_plan:
  dispatcher: mc2  # ← 改为 mc2，启用通信-计算重叠
  # 其他配置与 fused 一致
```

---

## 五、38 轮扫描关键发现

### 1. padding 长度 vs 性能

| Padding | WPS | HBM | 趋势 |
|---------|-----|-----|------|
| 1024 | 1415 | 48.8 GB | WPS 最高，HBM 最低 |
| 1280 | 1296 | 51.9 GB | 平衡点 |
| 1408 | 1158 | 54.6 GB | - |
| 1536 | 1133 | 56.4 GB | **生产推荐**（HBM 达标） |
| 2048 | 743 | 65.0 GB | WPS 暴跌，HBM OOM 边缘 |

**结论**: padding 越小，WPS 越高，HBM 越低；但需满足 HBM 55-60GB 目标。

### 2. no_sync_experts 优化

| 配置 | WPS | 收益 |
|------|-----|------|
| pad1536 (sync) | 1107 | baseline |
| pad1536 (nosync) | 1133 | **+2.3%** |

**原理**: `no_sync_experts: true` 避免专家权重同步时的额外通信开销。

### 3. mbs=2 失败分析

23 次尝试组合（bucket/chunk/timeout/nosync/rc_on）全部挂在外部 SIGTERM，非简单调参能解。

**结论**: 放弃 mbs=2，通过 MC2/pack 优化吞吐。

---

## 六、与 Pack 格式对比

| 方案 | WPS | HBM/卡 | 状态 | 分支 |
|------|-----|--------|------|------|
| **Pad1536 nosync** | 1133 | 56.4 GB | ✅ 本分支 | `mc2-perf-eval` |
| Pad + MC2 | 1230-1290 | 55-60 GB | ⏳ 待实测 | `mc2-perf-eval` |
| **Pack rc_off** | 2111 | 40 GB | ✅ 已验证 | `feat/llm-pad-to-pack-recompute` |
| Pack rc_on | 1475 | 33 GB | ✅ 已验证 | `feat/llm-pad-to-pack-recompute` |
| **Pack + MC2** | 2320+ | ~40 GB | 🎯 最高优先级 | 待组合验证 |

**结论**: Pack 格式吞吐远高于 Pad（+86%），是后续优化主方向。

---

## 七、下一步方向

### 🎯 Priority 1: Pad + MC2 验证

验证 MC2 在 pad1536 配置上的收益（预期 WPS 1133 → 1230-1290）

**意义**:
- 为 Pack + MC2 提供对照基准
- 验证 MC2 在 manual EP 权重布局下的兼容性

### 📋 Priority 2: Pack + MC2 组合

在 pack rc_off (WPS 2111) 基础上启用 MC2，预期 WPS 2320+

---

## 八、参考资料

### 完整报告

- **38 轮扫描**: [`reports/qwen35_audio_manual_ep8_perf_tuning_20260616.md`](reports/qwen35_audio_manual_ep8_perf_tuning_20260616.md)
- **MoE 优化策略**: [`reports/moe_optimization_strategy_from_blog_20260616.md`](reports/moe_optimization_strategy_from_blog_20260616.md)
- **详细日志**: [`reports/perf_runs/`](reports/perf_runs/) (38 个配置逐次日志)

### 项目文档（main 分支）

- [项目约束 (CLAUDE.md)](https://github.com/Kimsale/qwen3.5_35B_mindspeed/blob/main/CLAUDE.md)
- [项目状态 (STATUS.md)](https://github.com/Kimsale/qwen3.5_35B_mindspeed/blob/main/STATUS_QWEN35_AUDIO_TRAINING.md)
- [快速开始](QWEN35_AUDIO_TRAINING_GUIDE.md)

### 其他分支

- **main**: https://github.com/Kimsale/qwen3.5_35B_mindspeed
- **feat/llm-pad-to-pack-recompute**: Pack 格式完整版（推荐，WPS 2111）
- **feat/llm-pad-to-pack**: Pack 格式初版

---

**最后更新**: 2026-06-17  
**分支状态**: 活跃开发（MC2 待实测）  
**下次同步**: Pad + MC2 实测完成后
