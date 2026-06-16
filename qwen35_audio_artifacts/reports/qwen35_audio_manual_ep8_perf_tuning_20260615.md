# Qwen3.5-35B Audio LoRA Manual EP8 Performance Report

生成时间: 2026-06-15  
代码目录: `/data/sejin/third_party/mindspeed-mm-26.0.0`  
数据目录: `/data/sejin/baseline_26/data_audio_distfix_3200`  
目标: 不修改 Qwen3.5/Whisper 模型结构，在语音数据 LoRA 训练下调整分布式策略和训练参数，尽量提高 AICORE、HBM、功耗和 WPS。  

## 结论

当前约束下，最接近目标的稳定配置是:

`/data/sejin/third_party/mindspeed-mm-26.0.0/examples/qwen3_5_audio/perf_tuning/ep8_mbs1_ga4_rc_off_pad1536.yaml`

关键结果如下，全部为 warmup 后统计，不包含 init、加载、首步编译:

| 指标 | 结果 |
|---|---:|
| 有效统计步数 | 70 |
| 平均 step time | 5.026 s |
| input WPS | 1103.1 |
| AICORE 平均 / 峰值 | 22.88% / 43% |
| HBM 平均 / 峰值 | 55.08 / 56.68 GB |
| 功耗平均 / 峰值 | 162.1 / 203.0 W |
| loss 首个有效步 -> 最后一步 | 11.2927 -> 4.7051 |

这个配置满足 HBM 55-60GB 目标，训练稳定，WPS 仍可接受。但 AICORE 平均 40% 和功耗平均 240W 没达到。继续单纯增加 padding 到 2048 会 OOM，且 OOM 前 AICORE 平均只有 17.57%、WPS 降到 743.4，因此不是可行方向。

## 当前参数快照

通用模型和训练配置:

| 项 | 值 |
|---|---|
| Qwen 权重 | `/mnt/shared_data_196/sejin/models/Qwen3.5-35B-A3B` |
| Whisper encoder | `/mnt/shared_data_196/sejin/models/whisper-large-v3` |
| model_id | `qwen3_5_audio_manual_ep` |
| 语音 encoder | Whisper-large-v3 encoder, 冻结 |
| LoRA | rank 16, alpha 32, dropout 0.05 |
| LoRA target | Qwen self-attn `q_proj/k_proj/v_proj/o_proj` |
| 冻结模块 | `model.visual`, `audio_tower` |
| dtype | param `bf16`, reduce `fp32` |
| optimizer | AdamW, `adam_fused: true` |
| LR | 1e-4, cosine decay, warmup ratio 0.03 |
| grad clip | 1.0 |
| cutoff_len | 4096 |
| global batch | 32 |
| train_iters | 80 |
| 数据 | 3200 条语音 SFT 样本 |
| sampler | `BaseRandomBatchSampler` 或 mbs2 探索时的 `LengthBucketBatchSampler` |
| dataloader workers | 8 |
| loss | `train_on_prompt: false`, `ignore_pad_token_for_loss: true` |

并行配置:

| 项 | 值 |
|---|---|
| TP | 1 |
| EP | 8 |
| FSDP | `fully_shard_parallel_size: auto` |
| Ulysses | 1 |
| MoE impl | `qwen3_5` |
| EP apply module | `model.language_model.layers.{*}.mlp.experts` |
| recompute | 推荐配置为 `false`; 仅 OOM 规避实验中打开 |

运行环境:

```bash
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH
source /usr/local/Ascend/cann-8.5.0/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export NON_MEGATRON=true
export MULTI_STREAM_MEMORY_REUSE=2
export TASK_QUEUE_ENABLE=2
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_CONNECT_TIMEOUT=1800
export TOKENIZERS_PARALLELISM=false
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export AUDIO_PLACEHOLDER="<|AUDIO|>"
```

## 实现边界

没有改变 Qwen3.5 MoE 专家数量、路由规则、Whisper encoder、LoRA target 或训练数学目标。已做的是加载策略、分布式策略、batch/padding/recompute/pregather/监控统计等训练侧改动。

手动 EP8 关键实现:

- `mindspeed_mm/fsdp/models/qwen3_5_audio/manual_ep.py`
- `model_id: qwen3_5_audio_manual_ep`
- 使用 `safe_open` 逐 safetensors 张量读取，避免一次性把全量专家权重搬到设备。
- Qwen3.5 专家权重是融合 3D 张量，不是 `nn.ModuleList`:
  - `experts.gate_up_proj`: `[256, 1024, 2048]`
  - `experts.down_proj`: `[256, 2048, 512]`
- EP=8 时按 dim=0 切专家，rank 内只保留 32 个专家:
  - `expert_start = ep_rank * 32`
  - `expert_end = expert_start + 32`
- 非专家参数保持全量复制，专家参数本地切片，数学上等价于专家并行分片。

配套训练和分析改动:

- `mindspeed_mm/fsdp/train/trainer.py`: 支持 manual EP HF loading，并在 FSDP 后固定 LoRA-only trainable。
- `mindspeed_mm/fsdp/train/train_engine.py`: 增加 `perf_timing`，记录 `pregather/get_batch/move/forward/backward/clip/optimizer` 等阶段耗时和 token/WPS。
- `/data/sejin/baseline_26/scripts/run_audio_perf_experiment.sh`: 固定 CANN8.5 环境，启动 npu monitor，训练结束后自动分析。
- `/data/sejin/baseline_26/scripts/analyze_audio_perf_run.py`: 只按第 10 步后的时间窗统计硬件指标，排除 init、加载、首步编译。
- `/data/sejin/baseline_26/scripts/make_audio_perf_configs.py`: 生成 `mbs/recompute/pad/pregather/bucket` 实验配置。

## 统计方法

每轮分析使用:

- `skip_steps=10`
- 窗口起点: 第 11 个 iteration 的起始时间
- 窗口终点: 最后一个已记录 iteration 的结束时间
- `npu-smi` 采样间隔: 1s
- `AICORE/HBM/Power` 均只在上述窗口内取均值和峰值
- init、manual weight loading、首步编译、LoRA 保存均不计入性能均值

单轮详细报告在:

`/data/sejin/baseline_26/reports/perf_runs/`

## 稳定完成结果

| Tag | Strategy | Step s | Input WPS | AICORE avg/peak | HBM avg/peak GB | Power avg/peak W | Fwd/Bwd/Move ms |
|---|---|---:|---:|---:|---:|---:|---:|
| `ep8_mbs1_ga4_rc_on` | mbs1/ga4/rc_on/padnone | 5.568 | 995.7 | 6.40/32 | 32.91/33.29 | 131.4/170.9 | 2855.9/2640.1/10.5 |
| `ep8_mbs1_ga4_rc_off` | mbs1/ga4/rc_off/padnone | 3.851 | 1439.7 | 7.84/32 | 39.50/40.06 | 141.7/179.8 | 2244.8/1435.2/78.3 |
| `ep8_mbs1_ga4_rc_off_pad1024` | mbs1/ga4/rc_off/pad1024 | 4.042 | 1371.7 | 17.67/36 | 47.28/48.27 | 161.3/190.8 | 2331.2/1548.5/84.8 |
| `ep8_mbs1_ga4_rc_off_pad1280` | mbs1/ga4/rc_off/pad1280 | 5.068 | 1094.0 | 25.50/53 | 61.13/62.75 | 171.4/231.8 | 2826.1/1693.0/385.5 |
| `ep8_mbs1_ga4_rc_off_pad1024_pregather` | mbs1/ga4/rc_off/pad1024/pregather | 4.457 | 1243.8 | 20.90/50 | 55.69/59.36 | 164.9/233.2 | 2327.3/1599.8/214.9 |
| `ep8_mbs1_ga4_rc_on_pad1536` | mbs1/ga4/rc_on/pad1536 | 7.025 | 789.2 | 22.23/52 | 35.97/45.44 | 157.0/230.2 | 3122.5/3848.2/9.7 |
| `ep8_mbs1_ga4_rc_off_pad1536` | mbs1/ga4/rc_off/pad1536 | 5.026 | 1103.1 | 22.88/43 | 55.08/56.68 | 162.1/203.0 | 2785.7/1629.1/437.8 |

说明:

- `pad1536/recompute_off` 曾在 21:27 run 中 OOM，当时机器存在外部显存占用。清理后 22:34 重跑完整成功，最终表采用最新成功结果。
- `pad1280` 的 HBM 为芯片级总占用，早期实验窗口可能包含外部进程影响，因此用作趋势参考；最终推荐优先看清理后重跑的 `pad1536`。

## 失败或不推荐结果

| Tag | Strategy | Valid steps | OOM/失败前指标 | 结论 |
|---|---|---:|---|---|
| `ep8_mbs2_ga2_rc_off_pad1024_bucket` | mbs2/ga2/rc_off/pad1024 | 0 | 无 warmup 后窗口 | startup/首步阶段失败 |
| `ep8_mbs2_ga2_rc_off_pad1024_bucket_fixmask` | mbs2/ga2/rc_off/pad1024 | 0 | 无 warmup 后窗口 | startup/首步阶段失败 |
| `ep8_mbs2_ga2_rc_off_pad1024_bucket_trainmask` | mbs2/ga2/rc_off/pad1024 | 0 | 无 warmup 后窗口 | startup/首步阶段失败 |
| `ep8_mbs2_ga2_rc_off_pad512_bucket` | mbs2/ga2/rc_off/pad512 | 0 | 无 warmup 后窗口 | OOM/异常 |
| `ep8_mbs2_ga2_rc_off_pad256_bucket` | mbs2/ga2/rc_off/pad256 | 0 | 无 warmup 后窗口 | OOM |
| `ep8_mbs2_ga2_rc_off_pad256_bucket_tq1_alloc256` | mbs2/ga2/rc_off/pad256 | 0 | 无 warmup 后窗口 | allocator/运行异常 |
| `ep8_mbs2_ga2_rc_off_pad256_bucket_tq1_split256` | mbs2/ga2/rc_off/pad256 | 0 | 无 warmup 后窗口 | 运行异常 |
| `ep8_mbs2_ga2_rc_off_pad128_bucket` | mbs2/ga2/rc_off/pad128 | 13 | 2.132s, WPS 2718.3, AICORE 23.62%, HBM 59.80GB | OOM，不稳定 |
| `ep8_mbs2_ga2_rc_off_pad128_bucket_tq1` | mbs2/ga2/rc_off/pad128 | 14 | 2.513s, WPS 2536.1, AICORE 18.07%, HBM 59.32GB | hang/异常，不稳定 |
| `ep8_mbs1_ga4_rc_off_pad2048` | mbs1/ga4/rc_off/pad2048 | 20 | 7.432s, WPS 743.4, AICORE 17.57%, HBM 61.74GB | 第 30 步后 OOM |

`pad2048` OOM 原因:

```text
rank2: NPUWorkspaceAllocator tried to allocate 1.89 GiB
NPU 2 total capacity 60.96 GiB, only 1.79 GiB free
current operator: aclnnMul
```

这说明 `pad2048` 已越过当前 64GB 910B3 的可用显存边界。它在 OOM 前 AICORE 均值仍只有 17.57%，所以不是通向 40% AICORE 的有效方向。

## 阶段耗时分析

推荐配置 `pad1536/recompute_off` 的 warmup 后平均阶段耗时:

| Phase | Mean ms | Share |
|---|---:|---:|
| forward | 2785.7 | 55.4% |
| backward | 1629.1 | 32.4% |
| move | 437.8 | 8.7% |
| clip | 137.1 | 2.7% |
| optimizer | 3.9 | 0.1% |
| get_batch | 1.9 | 0.0% |
| loss_setup | 1.0 | 0.0% |

关键对比:

| 配置 | forward ms | backward ms | move ms | clip ms | 结论 |
|---|---:|---:|---:|---:|---|
| pad1024 | 2331.2 | 1548.5 | 84.8 | 39.3 | WPS 好，HBM/AICORE偏低 |
| pad1280 | 2826.1 | 1693.0 | 385.5 | 120.7 | AICORE最高但 HBM 偏高且 WPS下降 |
| pad1536 | 2785.7 | 1629.1 | 437.8 | 137.1 | HBM达标，综合最好 |
| pad2048 | 4613.4 | 2026.5 | 572.2 | 183.0 | HBM越界，长尾严重，最终 OOM |

启动和加载时间，均不计入性能均值:

| 配置 | Qwen safe_open load s | Whisper load s | post-load 到 step1 完成 s | step1 s |
|---|---:|---:|---:|---:|
| pad1024 | 40.28 | 1.27 | 56.76 | 23.13 |
| pad1280 | 38.02 | 1.62 | 68.37 | 31.72 |
| pad1536 | 36.07 | 1.30 | 40.64 | 24.85 |
| pad2048 | 33.86 | 1.58 | 71.86 | 38.85 |

## 为什么 AICORE 到不了 40%

1. LoRA 训练可训练参数很少: 当前 LoRA-only trainable 为 80 个张量，3,440,640 elements。优化器和梯度更新负载很轻，无法像全参或大范围解冻那样持续拉高计算密度。
2. 语音样本 token 长度偏短且波动大。实际 input token 吞吐高时，很多 step 的真实 token 数并不大，padding 只能制造额外计算，不能提高真实 WPS。
3. `mbs=1` 是当前稳定边界。`mbs=2` 的短窗口 WPS 很高，但在当前 CANN/torch_npu/算子组合下会 OOM、hang 或异常，不能作为生产配置。
4. 单纯增加 padding 会先推高 HBM 和搬移开销。`pad1536` 的 move 已到 437.8ms，`pad2048` 到 572.2ms，并且 `pad2048` OOM 前 AICORE 平均反而只有 17.57%。
5. recompute 不适合本任务的性能目标。`recompute_on/pad1536` 稳定但 backward 升到 3848.2ms，HBM 降到 35.97GB，WPS 只有 789.2。
6. 功耗平均跟 AICORE 一致，没有足够持续的 AI Core 计算压力，平均功耗维持在 150-170W，只有个别峰值接近 230W。

## 最终推荐

首选综合配置:

```bash
bash /data/sejin/baseline_26/scripts/run_audio_perf_experiment.sh \
  ep8_mbs1_ga4_rc_off_pad1536 \
  /data/sejin/third_party/mindspeed-mm-26.0.0/examples/qwen3_5_audio/perf_tuning/ep8_mbs1_ga4_rc_off_pad1536.yaml \
  1500 10 1.0
```

推荐理由:

- HBM 55.08GB 平均，56.68GB 峰值，满足 55-60GB。
- AICORE 22.88%，是清理外部负载后 HBM 达标配置中的最高稳定结果。
- WPS 1103.1，比 `pad1280` 略高，远高于 `pad2048`。
- 完整 80 step 训练成功，loss 正常下降，无 NaN/OOM。

备选:

| 场景 | 配置 | 原因 |
|---|---|---|
| WPS 优先 | `ep8_mbs1_ga4_rc_off_pad1024.yaml` | 1371.7 WPS，step 4.042s，但 HBM 47.28GB、AICORE 17.67% |
| HBM 55-60 且 WPS 更高 | `ep8_mbs1_ga4_rc_off_pad1024_pregather.yaml` | 1243.8 WPS，HBM 55.69GB，但 AICORE 20.90%，pregather 有额外开销 |
| AICORE趋势观察 | `ep8_mbs1_ga4_rc_off_pad1280.yaml` | 25.50% AICORE，但 HBM 61.13GB 且早期窗口可能受外部负载影响 |
| 不推荐 | `pad2048`, `mbs2`, `recompute_on` | OOM/不稳定/显著降 WPS |

## 目标达成情况

| 目标 | 结果 |
|---|---|
| EP=8 训练启动和稳定训练 | 已完成，manual EP safe_open + dim0 expert slicing 生效 |
| 不修改模型架构 | 已遵守，改动集中在加载、并行、训练配置和监控 |
| warmup 后再统计 | 已完成，所有核心表格跳过前 10 step |
| 记录每部分耗时 | 已完成，记录 forward/backward/move/clip/optimizer/加载阶段等 |
| HBM 55-60GB | 已达成，推荐配置 55.08/56.68GB |
| AICORE 平均 40%+ | 未达成，最佳稳定清理后配置 22.88%，历史最高稳定 25.50% |
| 功耗平均约 240W | 未达成，最佳稳定均值约 171.4W，推荐配置 162.1W |
| WPS 尽量高 | 稳定最高 1439.7 WPS，无 padding；HBM达标下推荐 1103.1 WPS 或 pregather 1243.8 WPS |

在不改模型结构、不扩大真实训练负载、不切换到不稳定的 `mbs=2` 路径的前提下，当前调参空间已经验证到 HBM 上限。AICORE 40% 和 240W 平均功耗不是本轮配置侧可达目标，瓶颈来自 LoRA 小训练面、真实 token 负载偏小、mbs=1 稳定边界和 Ascend 当前算子/allocator 对高 padding 或 mbs2 的不稳定。

