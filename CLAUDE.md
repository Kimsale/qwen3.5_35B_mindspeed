# 项目约束文档 CLAUDE.md

## 一、项目总纲

### 1. 项目信息

**项目**: Qwen3.5-35B-A3B + Whisper-large-v3 LoRA 微调性能优化  
**硬件**: 单机 8×昇腾910B3 (64GB HBM/卡)  
**环境**: CANN 8.5.0 + MindSpeed-MM 26.0.0  
**模型路径**:
- LLM: `/data/sejin/models/Qwen3.5-35B-A3B/` (67GB, HF safetensors)
- Audio Encoder: `/data/sejin/models/whisper-large-v3/` (冻结)
- 数据: `/data/sejin/audio_data/` (JSONL + 音频文件)

**环境切换**:
- **本机已部署完成 CANN 8.5.0**，无需安装包、不用重装驱动
- **项目全程禁用 CANN 8.1**，仅通过环境变量切换
- 环境脚本: `scripts/env_cann85.sh`

**框架固定**:
- MindSpeed-MM 26.0.0 (NPU 优化版 transformers + FSDP2)
- 源码路径: `/data/sejin/third_party/mindspeed-mm-26.0.0/`
- Patches 归档: `mindspeed_mm_patches/` (音频插件 + MC2 + pack collator)

---

### 2. 项目目标

1. ✅ **已完成**: Pad 格式 38 轮配置扫描，最优稳定配置 WPS 1133, HBM 56.4GB
2. ✅ **已完成**: Pack 格式优化验证，WPS 2111 (+86%), HBM 40GB (-29%)
3. ✅ **已完成**: Recompute 策略验证（layer-wise），HBM -7GB, WPS -30%
4. ⏳ **进行中**: MC2 通信-计算重叠实测（代码已接通，待 audio EP8 验证）
5. 🎯 **下一步**: Pack + MC2 组合，预期 WPS 2320+

**不改动项**（硬约束）:
- 模型结构（40层，256专家/层，MoE 路由算法）
- 数学一致性（loss 收敛轨迹需与 baseline 对齐）
- LoRA-only（LLM 冻结 + LoRA r=16，audio encoder 冻结，projector 全量训）

**优化范围**（允许调整）:
- 并行策略: EP/FSDP2/TP/SP 组合，当前 EP8 + FSDP2
- 序列格式: pad → pack (已验证 +86% WPS)
- 通信优化: MC2 通信-计算重叠（代码已接通）
- 显存优化: recompute layer-wise（已验证 -7GB HBM）
- 超参: mbs/ga/lr/max_seq_length/padding 策略

---

### 3. 环境切换（关键）

**CANN 8.5 环境脚本**（每次训练前必须 source）:
```bash
source /data/sejin/_repo_sync/scripts/env_cann85.sh
```

**验证环境**:
```bash
npu-smi info | head -5
python3 -c "import torch_npu; print(torch_npu.__version__)"  # 应输出 2.6.0.post1+cann85
```

**禁止事项**:
- ❌ 不要加载 CANN 8.1 环境变量
- ❌ 不要混用不同 CANN 版本的 torch_npu
- ❌ 出现环境冲突时强制锁定 8.5、屏蔽 8.1

---

## 二、硬约束

### 1. 优化边界（不可改）

- **模型结构**: Qwen3.5-35B-A3B 架构固定（40层，每层 256 专家，topk=8，shared experts=4）
- **MoE 路由**: 不改路由算法、专家选择策略、load balance loss
- **LoRA 范围**: 仅 LLM 的 attention + MLP（不含 shared experts），audio encoder 冻结
- **数学一致**: loss 必须正常收敛，无 NaN/梯度爆炸，对比 baseline 轨迹一致
- **HBM 目标**: ≤ 60GB/卡（当前 pack rc_off 40GB 已达成，留 24GB 余量）

### 2. 可调范围

- **并行配置**: EP/TP/FSDP2/SP 组合，默认 EP8 + FSDP2
- **序列格式**: pad（样本对齐）或 pack（样本拼接，当前最优）
- **通信优化**: MC2/fused dispatcher 选择
- **显存优化**: recompute 粒度（layer-wise / selective / 关闭）
- **超参**: micro_batch_size/gradient_accumulation_steps/lr/max_seq_length

### 3. AutoTuning 架构限制（已实测）

MindSpeed-MM 26.0.0 **架构上不支持** AutoTuning（5 次注入修复尝试，全部失败）:
- 框架入口无 `autotune` 参数
- `TrainingArguments` 无 `enable_autotune` 字段
- AutoTuning 模块 (`mindspeed.core.tune`) 不存在

**结论**: 依托手动配置扫描 + 报告分析优选参数，不引入第三方调优工具。

---

## 三、机器信息

### 硬件配置

| 项 | 值 |
|---|---|
| **NPU** | 8×昇腾910B3 |
| **HBM/卡** | 64 GB |
| **总HBM** | 512 GB |
| **NPU间互连** | HCCS (高速) |

### 已部署环境

| 项 | 路径/版本 |
|---|---|
| **CANN** | 8.5.0 (`/usr/local/Ascend/ascend-toolkit/8.5.RC1/`) |
| **Python 虚拟环境** | `/data/sejin/env/venv_cann85/` |
| **torch_npu** | 2.6.0.post1+cann85 |
| **transformers** | 4.57 (MindSpeed patched, 支持 NPU FA2 varlen) |
| **MindSpeed-MM** | 26.0.0 (`/data/sejin/third_party/mindspeed-mm-26.0.0/`) |

### 测试机器

- **172.29.226.188** (task3-910B-188): 主力测试机，8×910B3，所有实验在此完成

---

## 四、关键路径

### 模型与数据

```bash
# LLM 权重 (HF safetensors, 14分片, 67GB)
/data/sejin/models/Qwen3.5-35B-A3B/

# Audio encoder (Whisper-large-v3, 冻结)
/data/sejin/models/whisper-large-v3/

# 训练数据 (JSONL + 音频文件)
/data/sejin/audio_data/train.jsonl
/data/sejin/audio_data/wavs/*.wav

# 输出目录
/data/sejin/output/qwen35_audio_ckpt/
```

### 配置与脚本

```bash
# 仓库根目录
/data/sejin/_repo_sync/

# 头牌配置（各分支最优）
configs/perf_tuning/ep8_pack_188.yaml                    # Pack rc_off (WPS 2111)
configs/perf_tuning/ep8_mbs1_ga4_rc_on_pack_188.yaml     # Pack rc_on (HBM 33GB)
configs/perf_tuning/ep8_mbs1_ga4_rc_off_pad1536_nosync.yaml      # Pad 最优稳定
configs/perf_tuning/ep8_mbs1_ga4_rc_off_pad1536_nosync_mc2.yaml  # Pad + MC2 (待实测)

# 训练脚本
scripts/train_qwen35_audio.sh          # 标准训练入口
scripts/run_audio_perf_experiment.sh   # 性能实验脚本

# 环境脚本
scripts/env_cann85.sh                  # CANN 8.5 环境变量
```

### 源码改动（已归档到 patches）

```bash
mindspeed_mm_patches/
├── 01_source_code.patch               # 框架源码改动 (MC2 + 音频插件 + pack collator)
├── 02_examples_configs.patch          # 训练脚本 + 221 个 perf_tuning yaml
└── 00_full_commit_46de4e18.patch      # 全量兜底
```

---

## 五、故障处理流程

### 1. 环境问题

**症状**: `ModuleNotFoundError: No module named 'torch_npu'`

**排查**:
```bash
# 1. 确认 CANN 环境
echo $LD_LIBRARY_PATH | grep "8.5"  # 应包含 8.5 路径，不含 8.1

# 2. 确认虚拟环境
which python3  # 应指向 /data/sejin/env/venv_cann85/bin/python3

# 3. 重新加载环境
source /data/sejin/_repo_sync/scripts/env_cann85.sh
source /data/sejin/env/venv_cann85/bin/activate
```

**根因**: CANN 8.1/8.5 环境变量冲突，或虚拟环境未激活

---

### 2. OOM (Out of Memory)

**症状**: `RuntimeError: HBM out of memory`

**优先级排序**:
1. **启用 recompute** (layer-wise): HBM -7GB, WPS -30%
   ```yaml
   parallel:
     recompute: true
     recompute_plan:
       apply_modules:
       - model.language_model.layers.{*}
   ```

2. **降低 max_seq_length** (pad 格式): 1536 → 1408 → 1280
   - 每降 128: HBM -2~3GB, WPS +5~8%

3. **切换到 pack 格式**: HBM -29% (56.4GB → 40GB), WPS +86%
   - 需设置 `collate_param.model_name: qwen3vl_packed`
   - 需设置 `attn_implementation: flash_attention_2`
   - 需环境变量 `export AUDIO_PLACEHOLDER="<|AUDIO|>"`

**不推荐**:
- ❌ 降低 LoRA rank: 对 WPS/HBM 无收益（已实测 rank64 lora_nonexpert 无效）
- ❌ 增大 padding: HBM 上升，WPS 下降

---

### 3. 训练 hang

**症状**: 训练卡在某一步不动，无报错

**常见根因**:

| 场景 | 根因 | 解决 |
|------|------|------|
| **Pack mbs>1** | FSDP2 lazy init 时跨 rank 序列长度不一致 | 强制 mbs=1，或改 collator 加全局长度对齐 |
| **Pad mbs=2** | 外部环境级问题（23 次调参均无效） | 放弃 mbs=2，通过 MC2/pack 优化吞吐 |
| **MC2 dispatcher** | 算子 bug 或权重布局不兼容 | 回退 `dispatcher: fused`，上报昇腾 |

**通用诊断**:
```bash
# 1. 查看各 rank 是否都卡住
ps aux | grep python3 | grep train

# 2. 查看 NPU 利用率
npu-smi info  # AI Core 应 >10%，若=0% 说明 hang

# 3. 查看日志最后一行
tail -50 /data/sejin/output/qwen35_audio_ckpt/train.log
```

---

### 4. Loss NaN / 梯度爆炸

**症状**: `loss=nan` 或 `loss>1e10`

**排查优先级**:
1. **数据问题**: 检查 JSONL 是否有异常样本（音频损坏、文本编码错误）
2. **学习率过高**: 降低 lr (默认 1e-4 → 5e-5)
3. **MC2 数值精度**: 回退 `dispatcher: fused` 验证
4. **混合精度**: 检查 `bf16` 配置（默认已用 bf16）

**验证数学一致性**:
- 对比同配置下 fused vs mc2 的 loss 轨迹（允许 ±0.05 波动）
- 若偏差 >0.1，回退并上报

---

## 六、性能指标规范

### 必采集指标（每轮实验）

| 指标 | 采集方式 | 目标 |
|------|---------|------|
| **WPS** | 训练日志 `wps=xxx` | ≥1133 (pad) / ≥2111 (pack) |
| **单步耗时** | 训练日志 `ms=xxx` | ≤4.89s (pad) / ≤3.6s (pack) |
| **HBM/卡** | `npu-smi info` | ≤60GB (目标) / 40GB (pack 实测) |
| **AI Core 利用率** | `npu-smi info` | 均值 ≥20%, 峰值 ≥35% |
| **Loss 收敛** | 训练日志每 10 步 | 单调下降，无 NaN |
| **训练稳定性** | 完成步数 / 目标步数 | 80/80 (标准验证) |

### 可选指标（性能分析）

- 功耗 (W)
- 各阶段耗时分解 (forward / backward / optimizer)
- 通信时间 (AllToAll / AllGather / ReduceScatter)

---

## 七、文档规范

### 实验报告结构

1. **概述**: 实验目标、配置变更、预期收益
2. **环境**: 硬件、软件、分支、commit
3. **配置**: YAML 关键段（parallel / data / model）
4. **结果**: 表格（WPS / HBM / 单步耗时 / 完成步数）
5. **分析**: 瓶颈定位、与预期对比、根因
6. **结论**: 是否达成目标、下一步方向

### 表格格式（Markdown）

```markdown
| 配置 | WPS | HBM/卡 | 单步耗时 | 状态 |
|------|-----|--------|----------|------|
| pack rc_off | 2111 | 40 GB | 3.6s | ✅ 80步稳定 |
| pad1536 | 1133 | 56.4 GB | 4.89s | ✅ 80步稳定 |
```

### 命名规范

**配置文件**: `ep{EP}_mbs{MBS}_ga{GA}_rc_{on|off}_{pad{SIZE}|pack}_{feature}.yaml`  
**报告文件**: `{topic}_{date}.md` (例: `qwen35_audio_llm_pack_perf_20260616.md`)  
**分支命名**: `feat/{feature}` (功能) / `perf/{optimization}` (性能优化)

---

## 八、输出规范

### 命令脚本

- 优先输出 **可直接运行** 的 bash 脚本
- 脚本必须带注释说明用途
- 环境依赖写在开头（source / export）

### 问题排查

输出格式:
```
## 根因
{一句话描述}

## 修复
{bash 命令，可直接复制执行}

## 验证
{验证命令，确认修复生效}
```

### 禁止事项

- ❌ 不要输出 CUDA/NVIDIA/GPU 相关方案
- ❌ 不要引用 CANN 8.1 环境或路径
- ❌ 不要假设未验证的算子可用性（先用 probe 脚本探测）

---

## 九、补充说明

### 后续可接入方向

- **SWIFT**: 昇腾官方训练框架，同环境对标测试
- **FP8**: 需精度校准，且与 MC2 互斥（`validate_args_patch.py:790`）
- **更大 batch**: mbs>1 需解 FSDP2 hang（collator 加全局对齐）

### AutoTuning 结论

MindSpeed-MM 26.0.0 架构上**不支持 AutoTuning**（已实测确认）。  
当前依托手动配置扫描 + 性能报告分析优选参数，不引入第三方调优工具。

### 环境冲突处理

出现多 CANN 环境冲突时:
1. 强制锁定 CANN 8.5: `source scripts/env_cann85.sh`
2. 屏蔽 CANN 8.1 环境变量: `unset` 相关 `LD_LIBRARY_PATH` / `PYTHONPATH`
3. 验证环境: `npu-smi info`, `python3 -c "import torch_npu; print(torch_npu.__version__)"`

---

**最后更新**: 2026-06-17  
**维护**: 随实验进展同步更新（新增优化方向、已知限制、故障案例）
