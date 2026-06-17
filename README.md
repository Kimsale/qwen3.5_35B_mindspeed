# Qwen3.5-35B-A3B + Whisper-large-v3 LoRA 微调复现仓库

**任务**: Qwen3.5-35B-A3B + Whisper-large-v3 LoRA 微调性能优化  
**环境**: 单机 8 卡 Ascend 910B3, CANN 8.5.0, MindSpeed-MM 26.0.0  
**约束**: 不改模型结构/MoE 路由/专家数量; LoRA-only; 数学一致; HBM 55-60G

**本分支成果（mc2-perf-eval）**: Pad 格式 38 轮配置扫描，EP8 最优稳定配置 **pad1536_nosync
（WPS 1133，HBM 56.4GB）**；MC2 通信-计算重叠代码已接通（`dispatcher: mc2`），预期 WPS +10-15%，
**待音频 EP8 实测验证**。详见 [`reports/qwen35_audio_manual_ep8_perf_tuning_20260616.md`](reports/qwen35_audio_manual_ep8_perf_tuning_20260616.md)。

---

## 一、仓库结构

```
baseline_26/
├── README.md                          # 本文件
├── CLAUDE.md                          # 项目约束（核心规则）
├── QWEN35_AUDIO_TRAINING_GUIDE.md    # 快速开始指南
├── STATUS_QWEN35_AUDIO_TRAINING.md   # 当前状态总结
├── WORK_SUMMARY_JUN1-5.md            # 历史工作总结
├── README_XUCHEN2_CONVERSION.md      # 权重转换说明
├── STATUS_XUCHEN2_CONVERSION.md      # 转换状态
│
├── reports/                          # 性能报告&分析文档
│   ├── moe_optimization_strategy_from_blog_20260616.md  # **核心优化策略**
│   ├── qwen35_audio_manual_ep8_perf_tuning_20260616.md  # 最新性能调优报告
│   └── ...                           # 历史报告
│
├── mindspeed_mm_patches/             # **MindSpeed-MM 26.0.0 源码改动 (patch)**
│   ├── README.md                     # 版本锚点 + 复现步骤 + 与报告对应关系
│   ├── 01_source_code.patch          # 框架源码改动 (MC2 核心 + 音频插件)
│   ├── 02_examples_configs.patch     # 训练脚本 + 221 个 perf_tuning yaml
│   └── 00_full_commit_46de4e18.patch # 全量兜底
│
├── configs/perf_tuning/              # 头牌配置便捷副本 (fused 基线 vs MC2)
│
└── scripts/                          # 训练&调试脚本
    ├── train_qwen35_audio.sh         # 训练启动脚本
    ├── train_qwen35_audio.yaml       # 训练配置
    ├── env_cann85.sh                 # CANN 8.5 环境脚本
    ├── run_audio_perf_experiment.sh  # 性能实验脚本
    ├── run_qwen35_audio_moe_blog_tuning_suite.sh  # MC2优化suite
    ├── make_audio_perf_configs.py    # 配置生成器
    ├── analyze_audio_perf_run.py     # 性能分析脚本
    ├── verify_mc2_equivalence.py     # MC2数学一致性验证
    ├── mc2_equivalence_hook.py       # MC2一致性hook
    ├── npu_monitor_full.py           # NPU监控脚本
    ├── prepare_audio_data.py         # 数据准备工具
    └── ...                           # 其他辅助脚本
```

---

## 二、快速开始

### 1. 环境准备

**硬件要求**：
- 8×Ascend 910B3 (64GB HBM/卡)
- CANN 8.5.0 + ATB/NNAL

**软件环境**：
```bash
# 固定 CANN 8.5 环境（全程使用）
source /usr/local/Ascend/cann-8.5.0/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# Python 虚拟环境
python3 -m venv venv_qwen35
source venv_qwen35/bin/activate

# 安装 MindSpeed-MM 26.0.0
# （参考 MindSpeed-MM 官方文档安装 torch_npu, transformers, peft 等依赖）
```

### 2. 模型准备

**需要下载的模型**：
1. **Qwen3.5-35B-A3B-audio-dcp**（69GB，MCore DCP格式）
   - 路径配置：修改 `scripts/train_qwen35_audio.yaml` 中的 `model_path`
   
2. **whisper-large-v3**（2.9-5.8GB，HF格式）
   - 路径配置：修改 `scripts/train_qwen35_audio.yaml` 中的 `whisper_path`

### 3. 数据准备

按照 `QWEN35_AUDIO_TRAINING_GUIDE.md` 准备音频训练数据：

```bash
# 创建数据目录
mkdir -p data_audio

# 准备 JSONL 格式训练数据（示例）
# data_audio/train.jsonl:
# {"id": "sample_001", "audios": ["/path/to/audio1.wav"], "messages": [{"role": "user", "content": "<|AUDIO|>\n请转写这段语音。"}, {"role": "assistant", "content": "今天天气很好。"}]}
```

### 4. 训练启动

**小规模测试（10步）**：
```bash
cd scripts
# 修改 train_qwen35_audio.yaml: max_steps: 10
./train_qwen35_audio.sh
```

**完整训练（500步）**：
```bash
# 修改 train_qwen35_audio.yaml: max_steps: 500
./train_qwen35_audio.sh
```

---

## 三、核心文档速查

| 文档 | 用途 |
|---|---|
| `CLAUDE.md` | **项目硬约束**（必读）：环境切换、故障处置、优化范围、指标规范 |
| `QWEN35_AUDIO_TRAINING_GUIDE.md` | 训练快速开始指南 |
| `STATUS_QWEN35_AUDIO_TRAINING.md` | 当前工作状态&下一步行动 |
| `reports/moe_optimization_strategy_from_blog_20260616.md` | **MoE 优化策略**（基于博客+实测） |
| `reports/qwen35_audio_manual_ep8_perf_tuning_20260616.md` | 最新性能调优报告 |

---

## 四、性能优化工作流

### Phase 1: 基线采集
```bash
# 使用默认配置跑通训练，采集基线指标
cd scripts
./train_qwen35_audio.sh

# 性能分析
python3 analyze_audio_perf_run.py --run-dir ../output/qwen35_audio_ckpt
```

### Phase 2: MC2 通信-计算重叠优化
```bash
# 生成 MC2 优化配置
python3 make_audio_perf_configs.py --enable-mc2

# 运行优化实验
./run_qwen35_audio_moe_blog_tuning_suite.sh

# 对比分析
python3 analyze_audio_perf_run.py --baseline baseline_run --optimized mc2_run
```

### Phase 3: 报告生成
```bash
# 生成性能对比报告
python3 write_qwen35_audio_moe_blog_report.py \
    --baseline-metrics ../metrics/baseline.json \
    --mc2-metrics ../metrics/mc2.json \
    --output ../reports/mc2_optimization_report_$(date +%Y%m%d).md
```

---

## 五、关键配置说明

### 5.1 并行策略（EP8 手动分片）

当前使用 **手动专家并行 EP=8**（非自动 FSDP2 EP）：

```yaml
# train_qwen35_audio.yaml
parallel:
  tp: 1
  pp: 1
  cp: 1
  ep: 8  # 专家并行度

ep_plan:
  mode: manual
  ep_size: 8
  dispatcher: fused  # 可选: fused (默认), mc2 (通信-计算重叠)
```

### 5.2 LoRA 配置

```yaml
lora:
  enable: true
  rank: 16
  alpha: 32
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]  # 仅专家 FFN
  lora_dtype: bfloat16
```

### 5.3 显存优化

```yaml
recompute:
  enable: false  # mbs=1 下显存足够，不需要重计算

memory:
  micro_batch_size: 1
  gradient_accumulation_steps: 4
  max_seq_length: 1536  # 填充到此长度
```

---

## 六、故障排查

### 6.1 OOM（显存不足）

```bash
# 降低 max_seq_length
# 修改 train_qwen35_audio.yaml: max_seq_length: 1024

# 或启用重计算
# recompute.enable: true
```

### 6.2 通信超时

```bash
# 拉长 HCCL 超时时间
export HCCL_EXEC_TIMEOUT=1800

# 开启全量日志
export ASCEND_GLOBAL_LOG_LEVEL=0
export HCCL_DETERMINISTIC=1
```

### 6.3 mbs=2 挂死问题

**现象**：micro_batch_size=2 时，训练在 step 24 左右挂死（外部 SIGTERM）。

**原因**：已尝试 23 种配置（bucket16/32/64, chunk512, emptycache, timeout, nosync, rc_on），均失败。推测为 CANN 8.5 环境级 bug。

**建议**：
- 维持 mbs=1（稳定）
- 通过 MC2 通信重叠优化吞吐，而非强推 mbs=2
- 如必须 mbs=2，需联系昇腾支持

---

## 七、性能指标参考

### 7.1 Pad 基线配置（EP8, mbs=1, ga=4, rc_off, pad1536, nosync, fused）

| 指标 | 值 | 来源 |
|---|---|---|
| **单步耗时** | 4.89s | 38 轮配置扫描最优稳定配置 |
| **吞吐（WPS）** | 1132 words/s | 严格满足 HBM 55-60GB 目标 |
| **AI Core 利用率** | 23.46% (均值), 38.31% (峰值) | npu-smi 采集 |
| **HBM 占用** | 56.4 GB/64GB (88%) | 稳定 80 步无 OOM |
| **功耗** | 340.32W | - |
| **Loss** | 正常收敛，无 NaN | - |

### 7.2 MC2 通信-计算重叠（代码已接通，待音频 EP8 实测）

**状态**：
- ✅ 算子可用性已探测（`npu_alltoallv_gmm` / `npu_gmm_alltoallv` 在 CANN8.5 可用）
- ✅ 代码已接通（`expert_parallel.py` + `modeling_qwen3_5_moe.py:946` 支持 `dispatcher: mc2`）
- ⏳ **音频 EP8 实测待完成**（数学一致性验证 + 性能复测）

**预期收益（理论分析）**：

| 指标 | Pad 基线 (pad1536 fused) | MC2 预期 | 预期收益 |
|---|---|---|---|
| **单步耗时** | 4.89s | 4.3-4.5s | -10~-15% |
| **吞吐（WPS）** | 1132 | 1230-1290 | +10~+15% |
| **AI Core 利用率** | 23.46% | 25-28% | +2~5% |
| **HBM 占用** | 56.4 GB | 55-60GB | 维持 |

> 预期收益来自通信掩盖：forward (当前 2.66s) 和 backward (当前 1.62s) 阶段的 AllToAll
> 通信被掩盖到专家 GEMM 后面，节省通信暴露时间。详见
> [`reports/moe_optimization_strategy_from_blog_20260616.md`](reports/moe_optimization_strategy_from_blog_20260616.md) Phase 1。

**启用方式**（待验证）：

```yaml
# parallel.ep_plan 段
ep_plan:
  apply_modules:
  - model.language_model.layers.{*}.mlp.experts
  dispatcher: mc2  # ← 启用 MC2 融合算子
```

参考配置：`configs/perf_tuning/ep8_mbs1_ga4_rc_off_pad1536_nosync_mc2.yaml`

### 7.3 Pack 格式优化（其他分支已实测）

Pack 格式（消除样本内 padding）在 `feat/llm-pad-to-pack` 和 `feat/llm-pad-to-pack-recompute` 分支已验证：
- `feat/llm-pad-to-pack`: mbs=1 rc_off，**WPS 2069 (+79%)**，HBM ~40GB (-27%)
- `feat/llm-pad-to-pack-recompute`: 新增 recompute 配置，rc_on WPS 1475，HBM 33GB (-7GB)

MC2 与 pack 的组合（MC2 + pack mbs=1）为后续最高优先级方向。

---

## 八、参考资料

### 8.1 博客与文档

- **MoE 优化方案知识分享**（核心参考）：
  - 《从 Token 路由到昇腾/MindSpeed 落地》（知乎，2026）
  
- **昇腾官方文档**：
  - https://www.hiascend.com/document/detail/zh/Pytorch/700/modthirdparty/Mindspeedguide/mindspeed_0044.html
  - https://www.hiascend.com/developer/techArticles/20250702-1

### 8.2 MindSpeed-MM 26.0.0

- 官方示例：`examples/qwen3_5_audio/`
- 模型插件：`mindspeed_mm/fsdp/models/qwen3_5_audio/`
- EP 实现：`mindspeed_mm/core/parallel/expert_parallel.py`

---

## 九、贡献者

- **作者**: Sejin
- **生成时间**: 2026-06-17
- **框架**: MindSpeed-MM 26.0.0 on Ascend 910B3

---

## 十、许可证

本仓库仅用于学术研究和性能评测，不包含模型权重和训练数据。

模型权重需自行下载：
- Qwen3.5-35B-A3B: [Qwen 官方](https://github.com/QwenLM/Qwen)
- Whisper-large-v3: [OpenAI Whisper](https://github.com/openai/whisper)

---

**快速链接**：
- [快速开始](QWEN35_AUDIO_TRAINING_GUIDE.md)
- [项目约束](CLAUDE.md)
- [MoE 优化策略（含 MC2）](reports/moe_optimization_strategy_from_blog_20260616.md)
- [Pad 调优报告（38 轮扫描）](reports/qwen35_audio_manual_ep8_perf_tuning_20260616.md)
