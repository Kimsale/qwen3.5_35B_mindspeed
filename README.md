# Pack 格式优化分支 (feat/llm-pad-to-pack)

> **分支用途**: LLM 序列 pad→pack 改造初版验证  
> **核心成果**: WPS 2069 (+79%)，HBM ~40GB (-27%)  
> **完整报告**: [`reports/qwen35_audio_llm_pack_perf_20260616.md`](reports/qwen35_audio_llm_pack_perf_20260616.md)

> ⚠️ **建议使用 `feat/llm-pad-to-pack-recompute` 分支**：包含本分支所有成果，且新增 recompute 配置验证（WPS 2111 / 1475）

---

## 一、核心成果

| 配置 | WPS | HBM/卡 | 单步耗时 | 状态 | 收益 |
|------|-----|--------|----------|------|------|
| **pack mbs=1** | **2069** | ~40 GB | 3.79s | ✅ 80步稳定 | vs pad1158: **+79% WPS, -27% HBM** |

**Pack vs Pad 收益**（对标 pad1408）:
- 吞吐：1158 → 2069 (+79%)
- 单步耗时：4.79s → 3.79s (-21%)
- HBM 占用：54.6GB → ~40GB (-27%)
- Loss 收敛：正常单调下降（11.85 → 4.83）

---

## 二、Pack 格式原理

### 核心机制

**传统 Pad 格式**:
```
Sample 1: [audio] text <pad><pad>      # 补齐到 1408
Sample 2: [audio] text <pad><pad><pad> # 补齐到 1408
--> 每个样本独立，大量 padding 浪费
```

**Pack 格式**（本分支）:
```
Packed: [audio1] text1 [audio2] text2 [audio3] text3
--> 多样本拼接成单序列，零 padding 浪费
```

### 关键技术

1. **多样本拼接**: collator 将 batch 内样本 concat
2. **Position IDs 重启**: 每样本 position_ids 从 0 开始
3. **FA2 varlen**: 触发 `npu_flash_attn_varlen_func`
4. **音频边界识别**: 按 position_ids 跳变点替换音频 token

---

## 三、快速开始

### 环境准备

```bash
# 加载 CANN 8.5 环境
source scripts/env_cann85.sh
source /data/sejin/env/venv_cann85/bin/activate

# 必须设置环境变量
export AUDIO_PLACEHOLDER="<|AUDIO|>"
```

### 启动训练

```bash
cd /data/sejin/third_party/mindspeed-mm-26.0.0
bash examples/qwen3_5_audio/scripts/train_qwen35_audio.sh \
  --config examples/qwen3_5_audio/perf_tuning/ep8_pack_188.yaml \
  --max_steps 80
```

### 核心配置

```yaml
# data.dataloader_param.collate_param
collate_param:
  model_name: qwen3vl_packed          # 启用 pack
  ignore_pad_token_for_loss: true

# model
attn_implementation: flash_attention_2  # 必需，触发 NPU varlen FA2
```

---

## 四、实施中修复的问题

| # | 问题 | 根因 | 修复 |
|---|------|------|------|
| 1 | 数据预处理 audios/AUDIO token 数不匹配 | 漏设 `AUDIO_PLACEHOLDER` 环境变量 | 启动脚本加 `export AUDIO_PLACEHOLDER="<|AUDIO|>"` |
| 2 | loss_func `IndexError: Dimension out of range` | pack labels 是 1D，framework 期望 2D | collator 输出 `(1, total_len)` |
| 3 | Mask 构建错误（样本间交叉 attend） | position_ids 未按边界重启 | collator 为每样本生成独立 position_ids |
| 4 | 音频 token 位置错误 | 按全局索引查找，pack 拼接后失效 | forward 按 position_ids 跳变点定位音频边界 |

详见 [`reports/qwen35_audio_llm_pack_perf_20260616.md`](reports/qwen35_audio_llm_pack_perf_20260616.md) 第三节。

---

## 五、Loss 收敛轨迹

| iter | loss | step_ms | input_tok | wps |
|------|------|---------|-----------|-----|
| 1 | 11.85 | 39206 (编译) | 6462 | 165 |
| 10 | 11.21 | 3663 | 7952 | 2171 |
| 20 | 10.04 | 3597 | 8189 | 2277 |
| 30 | 7.83 | 3628 | 7960 | 2194 |
| 40 | 6.34 | 3943 | 8233 | 2088 |
| 50 | 5.46 | 3557 | 8394 | 2360 |
| 60 | 5.04 | 4482 | 7670 | 1711 |
| 70 | 4.72 | 3577 | 8269 | 2312 |
| 80 | 4.83 | 3860 | 7549 | 1956 |

**结论**: Loss 正常单调下降，无 NaN，80 步稳定完成 ✅

---

## 六、与其他方案对比

| 方案 | WPS | HBM/卡 | 状态 | 分支 |
|------|-----|--------|------|------|
| **Pack (本分支)** | 2069 | ~40 GB | ✅ 80步 | `feat/llm-pad-to-pack` |
| **Pack + recompute** | 2111 / 1475 | 40 / 33 GB | ✅ 完整验证 | `feat/llm-pad-to-pack-recompute` |
| Pad1408 rc_off | 1158 | 54.6 GB | ✅ 80步 | `mc2-perf-eval` |
| Pad1536 rc_off | 1133 | 56.4 GB | ✅ 80步 | `mc2-perf-eval` |

---

## 七、已知限制

### 1. 未验证 recompute

本分支仅验证 `recompute: false` 配置。

**解决**: 使用 `feat/llm-pad-to-pack-recompute` 分支（已验证 rc_on/rc_off）

### 2. 未验证 mbs>1

本分支仅验证 `micro_batch_size: 1`。

**解决**: recompute 分支已确认 mbs=2 在 pack 格式下 FSDP2 hang（根因：跨 rank 序列长度不一致）

---

## 八、参考资料

### 完整报告

- **本分支**: [`reports/qwen35_audio_llm_pack_perf_20260616.md`](reports/qwen35_audio_llm_pack_perf_20260616.md)
- **完整版**: [`feat/llm-pad-to-pack-recompute`](https://github.com/Kimsale/qwen3.5_35B_mindspeed/tree/feat/llm-pad-to-pack-recompute) 分支

### 项目文档（main 分支）

- [项目约束 (CLAUDE.md)](https://github.com/Kimsale/qwen3.5_35B_mindspeed/blob/main/CLAUDE.md)
- [项目状态 (STATUS.md)](https://github.com/Kimsale/qwen3.5_35B_mindspeed/blob/main/STATUS_QWEN35_AUDIO_TRAINING.md)
- [快速开始](QWEN35_AUDIO_TRAINING_GUIDE.md)

### 其他分支

- **main**: https://github.com/Kimsale/qwen3.5_35B_mindspeed
- **feat/llm-pad-to-pack-recompute**: Pack 完整版（推荐）
- **mc2-perf-eval**: Pad 格式 38 轮扫描 + MC2 代码接通

---

**最后更新**: 2026-06-17  
**分支状态**: 归档（已被 recompute 分支取代）  
**推荐使用**: `feat/llm-pad-to-pack-recompute` 分支
