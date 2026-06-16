# sejin 使用 Claude Code 工作总结 (2026年6月1日-6月5日)

## 总览

**时间跨度**: 2026-06-01 至 2026-06-05  
**主要项目**: baseline_26 - Qwen3-30B-A3B MoE LoRA 微调优化  
**框架**: MindSpeed-LLM 26.0.0 + CANN 8.5 + NPU (Ascend 910B)  
**核心目标**: 对齐 hulk 配置，完成 xuchen2 模型转换，性能分析与优化

---

## 主要工作阶段

### 第一阶段：Baseline 训练与性能分析 (6月1-2日)

#### 工作内容
1. **Baseline LoRA 训练环境搭建**
   - 配置 Qwen3-30B-A3B MoE 模型 (128 experts, A3B=8 active)
   - TP=2, PP=1, EP=4, CP=1 并行配置
   - LoRA 微调 (r=16, alpha=32, target=qkv+proj)
   - Sequence length: 4096

2. **训练性能基准测试**
   - 多组超参数扫描 (micro batch size, gradient accumulation)
   - NPU 利用率监控和性能指标收集
   - 训练日志: `baseline_*.log` (10+ 次训练)

3. **性能分析报告**
   - 创建 `workflow_analysis_raw.md` (69KB, 1319行)
   - 训练吞吐量、HBM 使用、通信开销分析

#### 产出
- ✅ 训练脚本: `train_baseline_lora.sh`
- ✅ Baseline checkpoint: `/data/sejin/baseline_26/output/ckpt_baseline/`
- ✅ 性能基准数据

---

### 第二阶段：hulk 配置对齐与优化 (6月3日)

#### 工作内容
1. **hulk 配置深度分析**
   - 对比 hulk 的训练配置与 baseline 差异
   - 关键发现:
     - CP=2 (Ulysses context parallel)
     - Sequence length: 8192 (2倍于 baseline)
     - 不使用 swap-optimizer (纯 GPU)
     - LoRA r=32, alpha=64 (更大的秩)

2. **创建 hulk 对齐训练脚本**
   - `train_hulk_aligned_ready.sh` - 完全对齐 hulk 的配置
   - TP=1, PP=1, EP=8, CP=2 (Ulysses)
   - Sequence: 8192, LoRA r=32/alpha=64

3. **数据准备**
   - 生成 hulk 风格的 30k 训练数据 (`gen_hulk_dist_data.py`)
   - 数据预处理为 packed IndexedDataset (8192 token packing)
   - 预处理脚本: `preprocess_hulk_data.sh`

4. **对比分析报告**
   - `HULK_VS_BASELINE_COMPARISON.md` (21KB, 288行)
   - 详细对比 18 个配置维度的差异
   - `HULK_ALIGNMENT_EXECUTION_PLAN.md` - 执行计划

5. **hulk 对齐训练**
   - 多次训练尝试和调试 (20+ 训练日志)
   - NPU 监控脚本: `npu_monitor.py`
   - 参数扫描: `opt_sweep.sh`, `auto_sweep.sh`

6. **性能报告**
   - `hulk_aligned_20260603_210624_report.md` (16KB)
   - 最终性能: ~2.3 samples/sec (seq=8192, CP=2)

#### 产出
- ✅ hulk 对齐脚本: `train_hulk_aligned_ready.sh`
- ✅ 数据生成: `data_hulk_dist_30k/` (30k samples, packed)
- ✅ 对比分析报告: `HULK_VS_BASELINE_COMPARISON.md`
- ✅ 执行计划: `HULK_ALIGNMENT_EXECUTION_PLAN.md`
- ✅ 性能报告: `hulk_aligned_*_report.md`

---

### 第三阶段：xuchen2 Qwen3-Omni 模型转换 (6月3日 下午-晚上)

#### 工作内容
1. **模型抽取 (Omni 多模态 → 纯文本 MoE)**
   - 源模型: `/data/xuchen2/model/Qwen3-Omni-30B-A3B-Captioner`
   - 抽取 `thinker.model.*` 和 `thinker.lm_head` (18867 权重)
   - 去除 audio_tower/visual 编码器
   - 平铺嵌套 config 为标准 `Qwen3MoeConfig`
   - 脚本: `extract_thinker_text.py`

2. **权重转换 (HF → MindSpeed MCore)**
   - 转换为 TP=1, PP=1, EP=8 (对齐 hulk)
   - Grouped GEMM 格式 (weight1/weight2)
   - 输出: `/data/sejin/models/Qwen3-Omni-30B-A3B-hulk_tp1_pp1_ep8/` (77GB)
   - 耗时: 962 秒 (~16 分钟)
   - 脚本: `convert_xuchen2_qwen3omni_moe.sh`

3. **环境问题修复**
   - 解决 stale editable install 冲突 (mindspeed 0.8.0 → 26.0.0)
   - 修复 torchrun workers PYTHONPATH 继承问题
   - Tokenizer 对齐验证 (Base vs Captioner)

4. **训练脚本创建**
   - `train_xuchen2_hulk_aligned.sh`
   - 完全对齐 hulk: TP1/PP1/EP8/CP2, seq=8192, LoRA r=32

#### 产出
- ✅ 抽取脚本: `extract_thinker_text.py`
- ✅ 转换脚本: `convert_xuchen2_qwen3omni_moe.sh`
- ✅ 转换后权重: `Qwen3-Omni-30B-A3B-hulk_tp1_pp1_ep8/` (77GB, 8 ranks)
- ✅ 训练脚本: `train_xuchen2_hulk_aligned.sh`
- ✅ 转换文档: `README_XUCHEN2_CONVERSION.md`
- ✅ 状态文档: `STATUS_XUCHEN2_CONVERSION.md`

---

### 第四阶段：MindSpeed-MM 多模态训练测试 (6月4日)

#### 工作内容
1. **MindSpeed-MM 框架测试**
   - 测试 mindspeed-mm-26.0.0 对 Qwen3-Omni 的支持
   - 训练日志: `mindspeed_mm_qwen3_30b_a3b_lora_20260604_*.log`

2. **性能对比分析**
   - 对比 mindspeed-llm vs mindspeed-mm 性能
   - 报告: `mindspeed_llm_vs_mm_performance_comparison*.md` (2个版本)

#### 产出
- ✅ MM 训练日志分析
- ✅ 性能对比报告 (mindspeed-llm vs mindspeed-mm)

---

### 第五阶段：项目打包与部署 (6月3日晚)

#### 工作内容
1. **项目完整打包**
   - 打包 baseline_26 项目全部代码、脚本、文档
   - 包含 MindSpeed 26.0.0 依赖源码
   - 生成可部署包: `mindspeed-26.0.0-qwen3-lora-package-20260603_203446.tar.gz` (33MB)
   - 脚本: `package/pack.sh`

2. **部署文档**
   - `package/DEPLOYMENT.md` - 部署指南

#### 产出
- ✅ 部署包: `mindspeed-26.0.0-qwen3-lora-package-*.tar.gz` (33MB)
- ✅ 部署文档: `DEPLOYMENT.md`

---

### 第六阶段：磁盘清理与资源管理 (6月4-5日)

#### 工作内容
1. **重复 checkpoint 清理**
   - 删除 `qwen3_30b_a3b_lora_mindspeed_mm` (62GB)
   - 删除 `Qwen3-30B-A3B-Base/._____temp` (21GB)
   - 删除 ms_swift 测试 checkpoints (~750MB)
   - 清理 aicore_optimization 空目录

2. **磁盘空间优化**
   - 释放前: 0GB 可用 (100% 满)
   - 释放后: 74GB 可用 (98% 使用)

#### 产出
- ✅ 释放 ~84GB 磁盘空间
- ✅ 保留所有关键权重和 checkpoint

---

## 核心产出清单

### 📄 文档 (8份核心报告)

1. **workflow_analysis_raw.md** (69KB, 1319行)
   - 完整的训练工作流分析

2. **HULK_VS_BASELINE_COMPARISON.md** (21KB, 288行)
   - hulk 与 baseline 的 18 维度配置对比

3. **HULK_ALIGNMENT_EXECUTION_PLAN.md** (9.3KB, 243行)
   - hulk 对齐的详细执行计划

4. **FINAL_PERFORMANCE_REPORT.md** (16KB, 242行)
   - 最终性能报告

5. **hulk_aligned_20260603_210624_report.md** (16KB, 247行)
   - hulk 对齐训练的性能报告

6. **mindspeed_llm_vs_mm_performance_comparison*.md** (2版本)
   - MindSpeed-LLM vs MM 框架性能对比

7. **README_XUCHEN2_CONVERSION.md** (7.5KB)
   - xuchen2 模型转换完整指南

8. **STATUS_XUCHEN2_CONVERSION.md** (9.3KB)
   - xuchen2 转换项目当前状态

### 🔧 脚本 (26+ 个脚本)

#### 训练脚本 (7个)
- `train_baseline_lora.sh` - baseline LoRA 训练
- `train_hulk_aligned_ready.sh` - hulk 对齐训练
- `train_xuchen2_hulk_aligned.sh` - xuchen2 模型 hulk 训练
- `train_hulkdata.sh` - hulk 数据训练
- `train_hulk8k.sh` - 8k 序列长度训练
- `train_param.sh` - 参数化训练脚本
- `train_r5.sh` - r=5 LoRA 训练

#### 转换脚本 (4个)
- `extract_thinker_text.py` - Omni 模型抽取
- `convert_xuchen2_qwen3omni_moe.sh` - HF→MCore 转换
- `convert_weights_tp1_ep8.sh` - 权重转换 (TP1/EP8)
- `check_conversion_status.sh` - 转换状态检查

#### 数据脚本 (3个)
- `gen_hulk_dist_data.py` - hulk 分布数据生成
- `gen_hulk_full.sh` - 完整数据生成
- `preprocess_hulk_data.sh` - 数据预处理

#### 工具脚本 (12个)
- `env_cann85.sh` - CANN 8.5 环境
- `npu_monitor.py` - NPU 监控
- `parse_metrics.py` - 指标解析
- `opt_sweep.sh` - 超参扫描
- `auto_sweep.sh` - 自动扫描
- `run_hulk_aligned_eval.sh` - hulk 评估
- `package/pack.sh` - 项目打包
- ... 其他工具脚本

### 🗂️ 数据与模型

#### 训练数据
- `data_hulk_dist_30k/train.jsonl` (30,000 samples)
- `qwen3_sft_packed_*_document.{bin,idx}` (packed 8192-token)
  - input_ids, labels, attention_mask
  - 共 2318 packed samples

#### 模型权重
1. **Qwen3-30B-A3B-Base** (33GB)
   - HF 格式，16 个 safetensors 分片
   - Tokenizer + config

2. **Qwen3-Omni-30B-A3B-hulk_tp1_pp1_ep8** (77GB)
   - MindSpeed MCore 格式
   - TP=1, PP=1, EP=8 (8 个 rank)
   - 从 xuchen2 Captioner 转换而来

#### Checkpoints
- `baseline_26/output/ckpt_baseline/` - baseline 训练
- `baseline_26/output/ckpt_hulk_aligned/` - hulk 对齐训练
- `baseline_26/output/ckpt_xuchen2_hulk/` - xuchen2 训练 (预留)

### 📦 部署包
- `mindspeed-26.0.0-qwen3-lora-package-20260603_203446.tar.gz` (33MB)
  - 完整代码 + 脚本 + 文档
  - MindSpeed 26.0.0 源码依赖
  - 部署指南

### 📊 训练日志
- **56 个训练日志文件**
- 主要类型:
  - `baseline_*.log` (10+) - baseline 训练
  - `hulk_aligned_*.log` (20+) - hulk 对齐训练
  - `xuchen2_*.log` (5+) - xuchen2 转换测试
  - `mindspeed_mm_*.log` (3+) - MM 框架测试
  - `sweep_*.log`, `param_*.log` - 超参扫描

---

## 关键技术成果

### 1. 完整的 hulk 对齐方案
- ✅ 识别并对比 18 个配置维度差异
- ✅ 创建完全对齐的训练脚本
- ✅ 验证 CP=2 Ulysses + seq=8192 配置
- ✅ 性能基准: ~2.3 samples/sec

### 2. Qwen3-Omni 多模态模型转换方案
- ✅ 解决嵌套 config 抽取问题
- ✅ 实现 HF → MindSpeed MCore 转换
- ✅ 支持 TP=1/PP=1/EP=8 并行切分
- ✅ Grouped GEMM 格式转换

### 3. 性能分析与优化
- ✅ 系统性能基准测试方法
- ✅ NPU 利用率监控工具
- ✅ 超参数自动扫描框架
- ✅ 训练指标解析工具

### 4. 数据流水线
- ✅ hulk 风格数据生成器
- ✅ ShareGPT → packed IndexedDataset 预处理
- ✅ ChatML + reasoning 模板支持

### 5. 环境问题解决
- ✅ 修复 editable install 冲突 (mindspeed 0.8.0 vs 26.0.0)
- ✅ 解决 torchrun workers 模块导入问题
- ✅ Tokenizer 兼容性验证

---

## 性能数据对比

### Baseline (TP2/PP1/EP4, seq=4096)
- Sequence length: 4096
- Micro batch size: 2-8
- 吞吐量: ~3-5 samples/sec (取决于 MBS)
- HBM 峰值: ~25-30GB per NPU

### hulk Aligned (TP1/PP1/EP8/CP2, seq=8192)
- Sequence length: 8192 (2x baseline)
- CP=2 (Ulysses context parallel)
- 吞吐量: ~2.3 samples/sec
- HBM 峰值: ~28-32GB per NPU
- 训练稳定性: ✅ 验证通过

### MindSpeed-MM vs MindSpeed-LLM
- 详见 `mindspeed_llm_vs_mm_performance_comparison.md`

---

## 遇到的主要挑战与解决

### 1. 嵌套 Config 抽取
**问题**: xuchen2 Omni 模型使用嵌套 config (`thinker.config`)，HF 和 MindSpeed 不支持  
**解决**: 创建 `extract_thinker_text.py`，平铺 config 并抽取纯文本 MoE 权重

### 2. EP=8 权重转换
**问题**: MindSpeed 转换工具默认 EP=4，需要适配 EP=8  
**解决**: 验证 grouped GEMM 格式，确认 EP 自动按需切分

### 3. Tokenizer 不一致
**问题**: Base tokenizer (151669 tokens) vs Captioner (151676 tokens)  
**解决**: 验证编码一致性，选择兼容的 Base tokenizer

### 4. Editable Install 冲突
**问题**: venv 中旧版 mindspeed (0.8.0) MetaPathFinder 覆盖 PYTHONPATH  
**解决**: 卸载 stale editable installs，依赖 PYTHONPATH 注入

### 5. 磁盘空间不足
**问题**: 磁盘 100% 满 (0GB 可用)  
**解决**: 清理重复 checkpoint 和临时文件，释放 84GB

### 6. 后台训练进程管理
**问题**: AI agent 工具调用环境中后台进程无法持久化  
**解决**: 指导用户使用 tmux 或直接在终端启动

---

## 工作量统计

### 代码与脚本
- **26+ 个脚本** (shell + python)
- 约 **3000+ 行代码**

### 文档
- **8 份核心报告**
- 约 **3430 行文档** (不含代码注释)

### 训练实验
- **56 次训练运行**
- **20+ 次超参扫描**
- 累计训练时间: 约 8-12 小时

### 模型转换
- **1 次完整转换** (HF → MCore)
- 耗时: 962 秒 (~16 分钟)
- 输出: 77GB 权重

### 数据处理
- **30,000 样本** 预处理
- **2318 packed samples** (8192 tokens)
- 耗时: ~3 分钟

---

## 项目状态

### ✅ 已完成
1. Baseline LoRA 训练环境 + 性能基准
2. hulk 配置对齐 + 训练脚本
3. xuchen2 Qwen3-Omni 模型完整转换
4. 数据预处理流水线
5. 性能分析与对比报告
6. 项目打包与部署文档
7. 磁盘空间清理与优化

### ⏸️ 待完成
1. xuchen2 模型的完整训练运行
   - 脚本已就绪: `train_xuchen2_hulk_aligned.sh`
   - 权重已转换: 77GB MCore 格式
   - 数据已准备: 30k packed samples
   - **需要用户在终端手动启动**

2. 最终性能对比
   - xuchen2 vs baseline vs hulk aligned

---

## 关键文件路径

### 项目根目录
```
/data/sejin/baseline_26/
├── scripts/          # 26+ 个脚本
├── reports/          # 8 份核心报告
├── data_hulk_dist_30k/  # 训练数据
├── logs/             # 56 个训练日志
├── output/           # checkpoint 目录
├── package/          # 部署包
└── [核心文档]
    ├── README_XUCHEN2_CONVERSION.md
    ├── STATUS_XUCHEN2_CONVERSION.md
    └── WORK_SUMMARY_JUN1-5.md (本文档)
```

### 模型与数据
```
/data/sejin/models/
├── Qwen3-30B-A3B-Base/  (33GB)
└── Qwen3-Omni-30B-A3B-hulk_tp1_pp1_ep8/  (77GB)

/data/sejin/baseline_26/data_hulk_dist_30k/
├── train.jsonl  (30k samples)
└── qwen3_sft_packed_*_document.{bin,idx}
```

---

## Sessions 对话记录

### 主要对话主题
基于日志时间戳和产出文件推断的主要对话 session:

1. **Session 1: Baseline 训练与性能分析** (6月1-2日)
   - 环境配置与 baseline 训练
   - 性能监控工具开发
   - workflow 分析报告

2. **Session 2: hulk 配置深度对比** (6月3日上午)
   - HULK_VS_BASELINE_COMPARISON 文档创建
   - 超参数扫描实验
   - 配置对齐方案设计

3. **Session 3: hulk 对齐训练实施** (6月3日中午)
   - 训练脚本创建与调试
   - 多次训练尝试 (20+ 次)
   - 性能报告生成

4. **Session 4: xuchen2 模型转换** (6月3日下午-晚上)
   - Omni 模型抽取脚本开发
   - HF → MCore 转换实施
   - 环境问题调试 (editable install)
   - 数据预处理

5. **Session 5: 项目打包** (6月3日晚)
   - 项目完整打包
   - 部署文档编写

6. **Session 6: MindSpeed-MM 测试** (6月4日)
   - MM 框架训练测试
   - 性能对比分析

7. **Session 7: 磁盘清理** (6月4-5日)
   - 重复 checkpoint 识别与清理
   - 工作总结文档编写 (当前 session)

---

## 总结

### 核心价值
在 6月1-5日 的 5 天时间里，通过 Claude Code 辅助，完成了:

1. **完整的 MoE LoRA 微调工作流**
   - 从零搭建到生产级别
   - baseline → hulk 对齐 → xuchen2 转换

2. **深度性能分析与优化**
   - 系统性基准测试
   - 配置对齐与调优
   - 详细的对比报告

3. **复杂的模型转换方案**
   - 多模态 → 纯文本抽取
   - HF → MindSpeed MCore 转换
   - 灵活的并行配置 (TP/PP/EP/CP)

4. **可复用的工具链**
   - 26+ 个脚本
   - 数据预处理流水线
   - 监控与分析工具

5. **完整的文档体系**
   - 8 份核心报告
   - 操作指南与部署文档
   - 问题排查记录

### 技术亮点
- ✅ MoE (128 experts, A3B=8) + LoRA 微调
- ✅ Context Parallel (Ulysses) + 8192 序列长度
- ✅ 多模态模型抽取与转换
- ✅ Grouped GEMM 权重格式处理
- ✅ 完整的性能分析方法论

### 下一步
1. 在终端启动 xuchen2 训练
2. 收集最终性能数据
3. 完成三方对比报告 (xuchen2 vs baseline vs hulk)

---

**创建时间**: 2026-06-05  
**作者**: sejin (with Claude Code)  
**项目**: baseline_26 - Qwen3 MoE LoRA Finetuning
