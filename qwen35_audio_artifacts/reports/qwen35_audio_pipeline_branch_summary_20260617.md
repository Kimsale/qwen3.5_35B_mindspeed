# Qwen3.5-35B Audio Pipeline Experts Overlap Branch Summary

**日期**: 2026-06-17
**分支**: `feat/qwen35-audio-pipeline-experts-overlap`
**worktree**: `/data/sejin/third_party/mindspeed-mm-pipeline-experts-overlap`
**远端发布目标**: `github/feat/qwen35-audio-pipeline-experts-overlap`
**任务**: 在不修改模型结构、专家数、MoE 路由语义和 Whisper encoder 的前提下，为 Qwen3.5-35B-A3B + Whisper-large-v3 + LoRA 语音训练实现并验证 Pipeline Experts overlap。

本总结只把 warmup 后训练窗口作为性能口径。init、safe_open 加载、Whisper 加载、首步编译和 LoRA 保存不进入主性能均值。

---

## 1. 分支代码工作

| Commit | 内容 |
|---|---|
| `c90023bc` | 实现 Qwen audio Pipeline Experts overlap：新增 pipeline dispatcher、chunk/multi-stream A2A dispatch/combine、Qwen MoE 接入和相关参数。 |
| `f40cb3c6` | 修复自定义模型配置缺少 `dtype` 时的兼容性问题。 |
| `8ada09f36` | 增加稳定 `mbs1/pad1536` 80-step 配置；修复 `_AsyncA2AWork.wait()` 等待共享通信 stream 导致后续 chunk 被串行阻塞的问题。 |

主要代码文件：

| 文件 | 工作 |
|---|---|
| `mindspeed_mm/fsdp/distributed/expert_parallel/ep_dispatcher.py` | 新增 pipeline EP forward；按本地专家范围切 chunk；异步 AllToAll dispatch/combine；可选多 stream；记录 `dispatch_preprocess`、`pipeline_wait_dispatch`、`pipeline_compute`、`pipeline_wait_combine` 等 MoE phase timing。 |
| `mindspeed_mm/fsdp/distributed/expert_parallel/expert_parallel.py` | 将 `ep_plan.dispatcher`、`pipeline_chunks`、`pipeline_multi_stream`、`pipeline_min_tokens_per_chunk` 透传给 Qwen MoE forward。 |
| `mindspeed_mm/fsdp/models/qwen3_5_moe/modeling_qwen3_5_moe.py` | Qwen3.5 fused 3D expert 权重路径接入 `dispatcher="pipeline"`。 |
| `mindspeed_mm/fsdp/params/parallel_args.py` | `EPPlanConfig` 增加 `pipeline` dispatcher 和 chunk/multi-stream 参数。 |
| `mindspeed_mm/fsdp/ops/moe_ops/unpermute.py` | 修正 eager unpermute 对 inverse permutation 的恢复方式。 |
| `mindspeed_mm/fsdp/models/modelhub.py` | 兼容自定义模型配置缺少 `dtype` 的初始化路径。 |

新增/关键配置：

| 配置 | 用途 |
|---|---|
| `examples/qwen3_5_audio/perf_tuning/mbs1_pipeline_pad1536_nosync_80.yaml` | 最终稳定 Pipeline Experts 80-step 配置。 |
| `examples/qwen3_5_audio/perf_tuning/mbs2_fa2_pipeline_bucket64_chunk512_80.yaml` | mbs2 + FA2 + pipeline 探索配置，进入训练后挂起，不作为有效性能口径。 |

---

## 2. 已发布报告

| 报告 | 内容 |
|---|---|
| `qwen35_audio_artifacts/reports/pack_format_validation_report.md` | EP8 LLM pack 主基线副本：`pack rc_off, mbs=1` 为 `2111.4 WPS`。 |
| `qwen35_audio_artifacts/reports/qwen35_audio_manual_ep8_perf_tuning_20260616.md` | manual EP8 非 pack 全量调参结果，已改为 WPS-first 对比口径。 |
| `qwen35_audio_artifacts/reports/moe_optimization_strategy_from_blog_20260616.md` | 根据 MoE 优化博客筛选出的策略和执行后补充结论。 |
| `qwen35_audio_artifacts/reports/qwen35_audio_moe_blog_tuning_20260616.md` | 博客策略对应的候选配置和参考结果。 |
| `qwen35_audio_artifacts/reports/qwen35_audio_pipeline_experts_overlap_20260617.md` | Pipeline Experts overlap 代码状态、训练配置、性能、问题解释和建议。 |
| `qwen35_audio_artifacts/reports/perf_runs/qwen35_audio_pipeline_mbs1_pad1536_ep8_8ada09f36_20260617_110652.md` | 自动生成的单次 run 性能分析。 |
| `qwen35_audio_artifacts/reports/perf_runs/qwen35_audio_pipeline_mbs1_pad1536_ep8_8ada09f36_20260617_110652.json` | 单次 run 的结构化分析 JSON。 |

---

## 3. 最终有效运行

| 项 | 值 |
|---|---:|
| 机器 | `172.29.226.188` / `task3-910B-188` |
| Run tag | `qwen35_audio_pipeline_mbs1_pad1536_ep8_8ada09f36_20260617_110652` |
| 状态 | 成功完成 `80/80`，无 OOM，无 runtime error |
| 配置 | `examples/qwen3_5_audio/perf_tuning/mbs1_pipeline_pad1536_nosync_80.yaml` |
| EP | `8` |
| micro batch / GA | `1 / 4` |
| dispatcher | `pipeline` |
| pipeline chunks | `4` |
| pipeline multi stream | `true` |
| pipeline min tokens per chunk | `1024` |
| pad_to_multiple_of | `1536` |
| recompute | `false` |
| gradient_accumulation_no_sync | `true` |
| grouped expert matmul | `true` |

Warmup 后 step 11-80，共 70 step：

| 指标 | Pipeline Experts run |
|---|---:|
| step time mean | `8.589s` |
| input WPS | `645.5` |
| samples/s | `3.726` |
| AICORE mean / peak | `12.04% / 42.0%` |
| HBM mean / peak | `56,581.7 / 58,286 MB` |
| Power mean / peak | `135.28 / 182.9 W` |
| forward mean | `5727.8ms` |
| backward mean | `2624.9ms` |

MoE phase 关键数据：

| MoE phase | Mean |
|---|---:|
| `pipeline_wait_dispatch` | `10.418ms` |
| `pipeline_compute` | `7.004ms` |
| `permute_post_a2a` | `6.094ms` |
| `dispatch_preprocess` | `3.674ms` |
| `pipeline_start_combine` | `1.216ms` |
| `pipeline_wait_combine` | `0.068ms` |

---

## 4. 主基线对比

主目标已改为提高 WPS，不再用 HBM 55-60GB 作为主优化目标。主基线来自 `feat/llm-pad-to-pack-recompute` 的 pack 报告：

| 配置 | 状态 | WPS | step time | HBM/card | 结论 |
|---|---:|---:|---:|---:|---|
| EP8 LLM pack `rc_off, mbs=1` | 成功 80/80 | `2111.4` | `~3.6s` | `~40GB` | 当前 WPS 最优基线。 |
| EP8 LLM pack `rc_on, mbs=1` | 成功 80/80 | `1475.3` | `~5.0s` | `~33GB` | 省显存备选。 |
| Pipeline Experts `mbs1/pad1536` | 成功 80/80 | `645.5` | `8.589s` | `56.6GB` | 稳定但性能回退。 |
| manual EP8 non-pack `pad1024_pregather_nosync` | 成功 80/80 | `1414.6` | `3.919s` | `48.75GB` | 非 pack 历史最高 WPS。 |
| manual EP8 non-pack `pad1280_current` | 成功 80/80 | `1295.8` | `4.279s` | `51.93GB` | current-code 非 pack 最高 WPS。 |
| manual EP8 non-pack `pad1536_nosync_rerun05` | 成功 80/80 | `1132.5` | `4.895s` | `56.40GB` | 仅作 strict-HBM 参考。 |

Pipeline Experts run 相对 pack `rc_off, mbs=1` 只有 `30.6%` 吞吐，WPS 下降 `69.4%`，且 HBM 多占约 `16.6GB/card`。因此本分支实现不应替换 pack 基线。

---

## 5. 遇到的问题和处理

| 问题 | 现象 | 处理/结果 |
|---|---|---|
| 原始 EP8 OOM | 早期 MindSpeed-MM EP8 在首个 optimizer step 或加载阶段 OOM。 | manual EP8 路径使用 `safe_open` 逐张量加载，对 Qwen3.5 fused 3D expert 权重沿 dim=0 切片，解决 EP8 可跑性；该部分作为本分支基础能力保留。 |
| Pipeline chunk 串行等待 | 初版 `_AsyncA2AWork.wait()` 等待共享 comm stream，可能把后续 chunk 也一起等完，削弱 overlap。 | 改为优先等待当前 A2A handle，避免共享 stream 上后续 chunk 造成额外串行化。 |
| `dtype` 兼容 | 自定义 model args 缺少 `dtype` 字段时初始化失败。 | `ModelHub` 初始化前补默认 `dtype=None`。 |
| mbs2 pipeline placeholder | `mbs2_fa2_pipeline_bucket64_chunk512_80.yaml` 初始运行出现音频 placeholder 不匹配。 | 通过运行环境设置 `AUDIO_PLACEHOLDER="<|AUDIO|>"` 修复，能进入训练。 |
| mbs2 pipeline 挂起 | 修复 placeholder 后 mbs2 能跑到约 step 24，但之后无推进。 | 判定不稳定，不作为有效性能口径；最终采用 mbs1 稳定 run。 |
| pack mbs2 挂起 | pack 报告中 mbs2 在 FSDP2 lazy initialization 阶段挂住。 | 根因记录为 pack 后各 rank 变长序列未对齐；需要 collator 做 rank alignment，当前分支未解决。 |
| Pipeline WPS 回退 | mbs1/pad1536 pipeline 稳定但只有 `645.5 WPS`。 | MoE timing 显示 chunk 后 `pipeline_compute` 太小，无法覆盖 dispatch 等待；记录为负向结果。 |
| AICORE 未达目标 | Pipeline 平均 AICORE `12.04%`，manual non-pack 稳定 mbs1 也约 `18-24%`。 | 当前 mbs1 每专家 token 太少，小 GEMM 吃不满；mbs2 短窗口可到更高峰值但不稳定。 |

---

## 6. 已做尝试和结果

| 尝试 | 结果 | 判断 |
|---|---|---|
| manual EP8 expert slicing | EP8 从 OOM 变为可稳定训练。 | 必要基础修复。 |
| padding sweep `1024/1248/1264/1280/1408/1536/2048` | WPS 随 padding 增大总体下降；`2048` OOM。 | 继续靠 padding 抬 HBM 是反向优化。 |
| `gradient_accumulation_no_sync` | 降低 GA 非最后 micro-step 同步开销。 | 对 mbs1 稳定配置有价值。 |
| FA2 on stable mbs1 | `pad1536_nosync_fa2` 为 `1115.2 WPS`，低于 non-FA2 `1132.5 WPS`。 | mbs1 稳定路径无收益。 |
| FA2 on mbs2 | 短窗口可到 `3287.8 WPS`，但 step 24 后挂起。 | 不能作为稳定配置。 |
| LoRA rank64 + nonexpert Linear | `1044.5 WPS`，AICORE/power 未提升。 | 增加小 GEMM 和 optimizer/backward 开销，无收益。 |
| Pipeline Experts chunks=4 | 完成 80 step，但 `645.5 WPS`。 | 稳定但性能回退。 |
| strict-HBM `pad1536_nosync` | `1132.5 WPS`，HBM `56.40GB`。 | 只保留资源占用参考。 |
| pack `rc_off, mbs=1` | `2111.4 WPS`，约 `40GB/card HBM`。 | 当前主基线。 |

---

## 7. 结论

本分支完成了 Pipeline Experts overlap 的工程实现、稳定配置和 80-step 实测，但结果不满足 WPS 目标。核心原因是当前单机 EP8、mbs1 下每个专家的 token 数太少，chunk 后专家 GMM 更小，通信等待没有被专家计算覆盖，新增调度开销反而拉长 forward。

后续优化建议：

1. 主线回到 EP8 LLM pack `rc_off, mbs=1`，以 `2111.4 WPS` 作为目标基线。
2. 优先解决 pack `mbs=2` 的 rank-aligned collator，让变长 pack 在各 rank 上对齐，避免 FSDP2 lazy initialization hang。
3. 当前 Pipeline Experts 只保留为实验分支。若继续验证，只做低成本 ablation：`pipeline_chunks=1/2`、提高 `pipeline_min_tokens_per_chunk`、小 token 数直接回退 fused dispatcher。
4. 不再把 HBM 55-60GB 作为主目标；`pad1536_nosync` 只用于资源占用对照。
