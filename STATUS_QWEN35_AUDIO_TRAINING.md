# Qwen3.5-35B 音频训练 - 当前状态总结

**更新时间**: 2026-06-17  
**任务**: Qwen3.5-35B-A3B + Whisper Large v3 语音多模态 LoRA 微调性能优化  
**环境**: 单机 8×910B3 (64GB HBM/卡), CANN 8.5.0, MindSpeed-MM 26.0.0

---

## 一、核心成果总览（2026-06-15 至 06-17）

### ✅ 已完成的优化验证

| 优化方向 | 实测 WPS | HBM/卡 | 状态 | 分支 | 收益 |
|---------|---------|--------|------|------|------|
| **Pack rc_off** | **2111** | 40 GB | ✅ 80步稳定 | `feat/llm-pad-to-pack-recompute` | vs pad1133: **+86% WPS, -29% HBM** |
| Pack rc_on | 1475 | 33 GB | ✅ 80步稳定 | `feat/llm-pad-to-pack-recompute` | HBM -7GB, WPS -30% |
| Pad 最优稳定 | 1133 | 56.4 GB | ✅ 80步稳定 | `mc2-perf-eval` | 38 轮扫描基准 |
| Pad MC2 | 1230-1290 (预期) | 55-60 GB | ⏳ 待实测 | `mc2-perf-eval` | 代码已接通 |

**历史最高吞吐**: Pack mbs=1 rc_off, **WPS 2111**, HBM 40GB, 单步 3.6s  
**最优稳定配置**: Pack mbs=1 rc_off（吞吐优先）或 Pack mbs=1 rc_on（显存受限场景）

---

## 二、分支工作明细

### 1. `feat/llm-pad-to-pack-recompute` — Pack 格式完整验证 ✅

**核心改造**：LLM 序列 pad→pack，消除样本内 padding，原生 FA2 varlen

**实测结果**：
- ✅ mbs=1 rc_off: WPS 2111, HBM 40GB, 80步稳定，loss 正常收敛（4.83→4.62）
- ✅ mbs=1 rc_on: WPS 1475, HBM 33GB, 80步稳定（recompute layer-wise）
- ❌ mbs=2: FSDP2 lazy init hang（跨 rank 序列长度不一致，需 collator 加全局对齐）

**技术细节**：
- `modeling_qwen3_5_audio.py`: forward 支持 pack 检测（`cu_seqlens` → batch=1 + varlen FA2）
- `packed_collator_wrapper.py`: 多样本拼接 + position_ids 每样本从 0 重启
- NPU 算子：`npu_flash_attn_varlen_func`（transformers 4.57 原生路径）
- Recompute：`model.language_model.layers.{*}` layer-wise checkpoint，避免 checkpoint 整个 `language_model` 导致 layer loop 中间态显存回升

**报告**：[`pack_format_validation_report.md`](https://github.com/Kimsale/qwen3.5_35B_mindspeed/blob/feat/llm-pad-to-pack-recompute/pack_format_validation_report.md)

---

### 2. `mc2-perf-eval` — Pad 格式调优 + MC2 接通 ⏳

**Pad 格式 38 轮配置扫描**（6月15日）：
- ✅ 最优稳定配置：`ep8_mbs1_ga4_rc_off_pad1536_nosync`
  - WPS 1133, HBM 56.4GB, 单步 4.89s, 80步稳定
  - 严格满足 HBM 55-60GB 目标
- 历史最高 WPS（HBM 不达标）：`pad1024_pregather_nosync`, WPS 1415, HBM 48.75GB
- mbs=2 尝试 23 次全部挂在外部 SIGTERM（bucket/chunk/timeout/nosync/rc_on 组合均无效）

**MC2 通信-计算重叠**（6月16日）：
- ✅ 算子可用性已探测（`npu_alltoallv_gmm` / `npu_gmm_alltoallv` in CANN8.5）
- ✅ 代码已接通（`expert_parallel.py` + `modeling_qwen3_5_moe.py:946` 支持 `dispatcher: mc2`）
- ⏳ **音频 EP8 实测待完成**（数学一致性验证 + 性能复测）
- 预期收益：WPS 1133 → 1230-1290 (+10-15%)，通过掩盖 forward/backward 的 AllToAll 通信

**报告**：
- [`reports/qwen35_audio_manual_ep8_perf_tuning_20260616.md`](reports/qwen35_audio_manual_ep8_perf_tuning_20260616.md)
- [`reports/moe_optimization_strategy_from_blog_20260616.md`](reports/moe_optimization_strategy_from_blog_20260616.md)

---

### 3. `feat/llm-pad-to-pack` — Pack 格式初版验证 ✅

**实测结果**（6月16日）：
- ✅ mbs=1 rc_off: WPS 2069, HBM ~40GB, 80步稳定
- vs pad1408: WPS +79%, HBM -27%, 单步 -21%

**状态**：已被 `feat/llm-pad-to-pack-recompute` 分支取代（后者增加了 recompute 配置和更全面验证）

**报告**：[`reports/qwen35_audio_llm_pack_perf_20260616.md`](https://github.com/Kimsale/qwen3.5_35B_mindspeed/blob/feat/llm-pad-to-pack/reports/qwen35_audio_llm_pack_perf_20260616.md)

---

## 三、下一步行动（按优先级）

### 🎯 Priority 1: Pack + MC2 组合验证（最高优先级）

**目标**：在 pack 格式基础上启用 MC2，验证是否能叠加收益

**预期**：
- Pack 已实测 WPS 2111
- MC2 预期 +10-15% 通信掩盖
- **组合预期 WPS 2320+**（理论上限）

**实施**：
```yaml
# 在 pack 配置基础上加入
parallel:
  expert_parallel_size: 8
  ep_plan:
    apply_modules:
    - model.language_model.layers.{*}.mlp.experts
    dispatcher: mc2  # ← 启用 MC2
```

**验证项**：
- 数学一致性（loss 轨迹与 pack fused 对比）
- 性能收益（WPS 是否达到预期）
- 稳定性（80 步无 hang/OOM/NaN）

---

### ⏳ Priority 2: Pack mbs>1 解锁

**当前障碍**：FSDP2 lazy init 在 all-gather 处 hang（跨 rank 序列长度不一致）

**解决方案**：在 `PackedCollatorWrapper` 加跨 rank 全局长度对齐
- 各 rank 在 collate 前同步 `max_seq_length`（通过 `dist.all_reduce` 获取全局最大值）
- 所有 rank 统一 pad 到该长度（仅跨 rank 对齐，样本内仍保持 pack 无 padding）

**预期收益**：mbs=2 可将 global_batch_size 从 32 提升到 64，理论上吞吐进一步提升（但需实测验证 HBM 是否够用）

---

### 📋 Priority 3: Pad + MC2 基准验证

**目标**：在 pad 格式（`pad1536_nosync`）上启用 MC2，验证预期收益（WPS 1133 → 1230-1290）

**意义**：
- 为 Pack + MC2 提供对照基准
- 验证 MC2 在 manual EP 权重布局下的兼容性
- 确认 MC2 收益是否与通信时间分析一致

---

## 四、技术栈验证状态

| 优化点 | 状态 | 算子/机制 | 分支 |
|-------|------|----------|------|
| **FA2 varlen** | ✅ 已用 | `npu_flash_attn_varlen_func` | pack 系列 |
| **MoE GMM** | ✅ 已用 | `torch_npu.npu_grouped_matmul` | 所有分支 |
| **Fused permute/unpermute** | ✅ 已用 | `npu_moe_token_permute/unpermute` | 所有分支 |
| **Fused SwiGLU** | ✅ 已用 | `torch_npu.npu_swiglu` | 所有分支 |
| **MC2 通信-计算重叠** | ⏳ 代码✅ 实测⏳ | `npu_alltoallv_gmm` / `npu_gmm_alltoallv` | mc2-perf-eval |
| **Layer-wise recompute** | ✅ 已用 | PyTorch `checkpoint` | pack-recompute |
| **EP=8 手动分片** | ✅ 已用 | `manual_ep.py` | 所有分支 |
| **FSDP2** | ✅ 已用 | `fully_shard` | 所有分支 |
| **Pipeline** | ❌ 未用 | - | 无需（单机） |

---

## 五、已知限制与待解决问题

| 问题 | 根因 | 影响 | 解决方向 |
|------|------|------|---------|
| Pack mbs>1 hang | FSDP2 lazy init 时跨 rank 序列长度不一致 | 无法提升 mbs | Collator 加全局长度对齐 |
| MC2 audio EP8 未实测 | 时间优先给了 pack 验证 | MC2 收益未量化 | Priority 1/3 实测 |
| mbs=2 (pad) 外部 SIGTERM | 环境级问题，23 次调参均无效 | Pad 格式无法提升 mbs | 成本高，暂不继续 |

---

## 六、参考资料

### 性能报告
- [Pack 格式完整验证](https://github.com/Kimsale/qwen3.5_35B_mindspeed/blob/feat/llm-pad-to-pack-recompute/pack_format_validation_report.md)
- [Pad 调优 38 轮扫描](reports/qwen35_audio_manual_ep8_perf_tuning_20260616.md)
- [MoE 优化策略（含 MC2）](reports/moe_optimization_strategy_from_blog_20260616.md)

### 分支链接
- `feat/llm-pad-to-pack-recompute`: https://github.com/Kimsale/qwen3.5_35B_mindspeed/tree/feat/llm-pad-to-pack-recompute
- `mc2-perf-eval`: https://github.com/Kimsale/qwen3.5_35B_mindspeed/tree/mc2-perf-eval
- `main`: https://github.com/Kimsale/qwen3.5_35B_mindspeed

---

**最后更新**: 2026-06-17  
**下次同步**: Pack + MC2 组合验证完成后

**选择一个数据来源**：

**选项 A：使用演示数据（快速测试）**
```bash
cd /data/sejin/baseline_26/scripts
python3 prepare_audio_data.py --mode demo --output /data/sejin/baseline_26/data_audio --num-samples 10
```
然后手动准备 10 个 .wav 文件到 `data_audio/audio/` 目录，或修改 JSONL 中的路径指向你的真实音频文件。

**选项 B：转换现有数据集**
```bash
# 如果你有 CSV 格式数据（audio_path, transcription）
python3 prepare_audio_data.py --mode csv \
    --input your_data.csv \
    --output /data/sejin/baseline_26/data_audio/train.jsonl \
    --audio-dir /path/to/audio/files

# 如果你有目录格式（音频在一个目录，转写在文本文件）
python3 prepare_audio_data.py --mode dir \
    --audio-dir /path/to/audio/files \
    --transcription-file transcriptions.txt \
    --output /data/sejin/baseline_26/data_audio/train.jsonl
```

**选项 C：直接编写 JSONL**

按照以下格式手动创建 `/data/sejin/baseline_26/data_audio/train.jsonl`：
```jsonl
{"id": "sample_001", "audios": ["/path/to/audio1.wav"], "messages": [{"role": "user", "content": "<|AUDIO|>\n请转写这段语音。"}, {"role": "assistant", "content": "今天天气很好。"}]}
{"id": "sample_002", "audios": ["/path/to/audio2.wav"], "messages": [{"role": "user", "content": "<|AUDIO|>\n这段音频说了什么？"}, {"role": "assistant", "content": "会议定于下周三召开。"}]}
```

**验证数据**：
```bash
python3 prepare_audio_data.py --mode validate --input /data/sejin/baseline_26/data_audio/train.jsonl
```

---

### 🚀 Step 2：小规模测试（10步）

数据准备完成后，先跑 10 步验证配置：

```bash
# 1. 修改配置为小规模测试
cd /data/sejin/baseline_26/scripts
# 编辑 train_qwen35_audio.yaml，改 max_steps: 10, save_interval: 5

# 2. 启动训练
chmod +x train_qwen35_audio.sh
./train_qwen35_audio.sh

# 3. 监控日志
tail -f /data/sejin/baseline_26/logs/qwen35_audio_*.log
```

**关键检查点**：
- [ ] 训练成功启动（无模型加载错误）
- [ ] 音频数据正常加载（无路径错误）
- [ ] 首步完成（无 OOM、无崩溃）
- [ ] Loss 正常（首步约 2-5，后续下降）
- [ ] HBM 占用合理（55-62GB，与之前接近）
- [ ] 单步耗时合理（2-5s）

---

### 📊 Step 3：完整训练（500步）

测试通过后，恢复完整配置并重新训练：

```bash
# 1. 恢复配置
# 编辑 train_qwen35_audio.yaml，改回 max_steps: 500, save_interval: 100

# 2. 重新启动训练
./train_qwen35_audio.sh

# 3. 等待训练完成（预计 20-40 分钟）
```

**预期产出**：
- Checkpoint: `/data/sejin/baseline_26/output/qwen35_audio_ckpt/checkpoint-{100,200,300,400,500}/`
- LoRA adapter: `checkpoint-500/lora_adapter_model.safetensors` (~44MB)
- 日志: `/data/sejin/baseline_26/logs/qwen35_audio_*.log`

---

### ✅ Step 4：验证效果

训练完成后，用 LoRA adapter 进行推理验证效果。

---

## 五、关键文件速查

```bash
# 配置文件
/data/sejin/baseline_26/scripts/train_qwen35_audio.yaml      # 训练配置
/data/sejin/baseline_26/scripts/train_qwen35_audio.sh        # 启动脚本
/data/sejin/baseline_26/scripts/prepare_audio_data.py        # 数据准备工具

# 文档
/data/sejin/baseline_26/QWEN35_AUDIO_TRAINING_GUIDE.md       # 详细指南

# 模型
/mnt/shared_data_196/sejin/models/Qwen3.5-35B-A3B-audio-dcp  # LLM 权重
/mnt/shared_data_196/sejin/models/whisper-large-v3           # Whisper 权重

# 数据（需你准备）
/data/sejin/baseline_26/data_audio/train.jsonl               # 训练数据
/data/sejin/baseline_26/data_audio/val.jsonl                 # 验证数据（可选）

# 输出
/data/sejin/baseline_26/output/qwen35_audio_ckpt/            # Checkpoint
/data/sejin/baseline_26/logs/qwen35_audio_*.log              # 训练日志

# 环境
/data/sejin/env/venv_qwen35                                   # Python 环境
```

---

## 六、常见问题速查

| 问题 | 解决方案 |
|---|---|
| 数据路径错误 | 检查 JSONL 中 `audios` 字段的路径是否正确 |
| OOM（显存不足） | 降低 `max_seq_length` (2048→1024) 或启用 `activation_offload` |
| 音频格式不支持 | 转换为 16kHz .wav: `ffmpeg -i input.mp3 -ar 16000 output.wav` |
| 模型加载失败 | 确认 `model_path` 指向 **audio-dcp** 版本 |
| 单步过慢 | 正常现象（Triton 编译），后续会加速 |

---

## 七、性能预期（基于历史经验）

| 指标 | 预期值 | 备注 |
|---|---|---|
| 单步耗时 | 2-5s | 音频预处理比纯文本略慢 |
| HBM 占用 | 55-62GB/64GB | 与纯文本训练接近（audio_tower 冻结） |
| Loss 初始值 | 2-5 | 取决于任务类型 |
| Loss 收敛 | 0.5-1.5 | 500步后 |
| 训练总时长 | 20-40 分钟 | 500步，单步约 2-5s |

---

## 八、联系与参考

- **详细指南**：`/data/sejin/baseline_26/QWEN35_AUDIO_TRAINING_GUIDE.md`
- **官方示例**：`/data/sejin/third_party/mindspeed-mm-26.0.0/examples/qwen3_5_audio/`
- **历史报告**：`/data/sejin/baseline_26/reports/qwen35_35B_lora_60step_perf_report.md`

---

**总结**：所有配置和脚本已就绪，只需准备音频数据即可立即开始训练。数据准备是唯一的阻塞项，完成后即可启动。
