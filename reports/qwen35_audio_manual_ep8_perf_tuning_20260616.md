# Qwen3.5-35B Audio Manual EP8 Performance Tuning Report

**生成时间**: 2026-06-16  
**环境**: Ascend 910B3 x 8, CANN 8.5.0, MindSpeed-MM 26.0.0  
**模型**: Qwen3.5-35B-A3B + Whisper-large-v3 audio encoder + LoRA  
**约束**: 不改模型结构、专家数量、MoE 路由、Whisper encoder 结构；只调整训练/分布式策略、padding、recompute、no_sync、runtime env 和 LoRA adapter 训练参数。  
**统计口径**: 所有有效指标均跳过前 10 step warmup；不计 init、safe_open 加载、数据构建、首轮编译。  

---

## 1. 当前固定参数快照

| 项 | 当前值 |
|---|---:|
| model_id | `qwen3_5_audio_manual_ep` |
| Qwen 权重 | `/mnt/shared_data_196/sejin/models/Qwen3.5-35B-A3B` |
| Whisper 权重 | `/mnt/shared_data_196/sejin/models/whisper-large-v3` |
| EP | `8` |
| TP / Ulysses | `1 / 1` |
| FSDP | `fully_shard_parallel_size: auto` |
| dtype | `param_dtype=bf16`, `reduce_dtype=fp32` |
| LoRA | rank `16`, alpha `32`, dropout `0.05` |
| trainable | `80/1599` tensors, LoRA only |
| micro batch / GA | `1 / 4` for stable best configs |
| global batch | `32` |
| cutoff_len | `4096` |
| lr / schedule | `1e-4`, cosine, warmup ratio `0.03` |
| clip_grad | `1.0` |
| dataloader workers | `8` |
| runtime env | `MULTI_STREAM_MEMORY_REUSE=2`, `TASK_QUEUE_ENABLE=2`, `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`, `ACLNN_CACHE_LIMIT=100000`, `CPU_AFFINITY_CONF=1` |

Manual EP8 继续使用 safe_open 逐张量加载和 fused expert dim=0 切片；专家权重本身未改结构。

---

## 2. 本轮代码和配置改动

| 文件 | 改动 |
|---|---|
| `/data/sejin/third_party/mindspeed-mm-26.0.0/mindspeed_mm/fsdp/params/training_args.py` | 增加 `gradient_accumulation_no_sync` 训练参数。 |
| `/data/sejin/third_party/mindspeed-mm-26.0.0/mindspeed_mm/fsdp/train/train_engine.py` | GA 非最后 micro-step 使用 `model.no_sync()`，减少 FSDP 同步频率；保留数学等价的累计梯度。 |
| `/data/sejin/baseline_26/scripts/analyze_audio_perf_run.py` | 分析 JSON/MD 记录 runtime env、warmup 后硬件窗口、phase timing、run phase timing。 |
| `/data/sejin/third_party/mindspeed-mm-26.0.0/examples/qwen3_5_audio/perf_tuning/ep8_mbs1_ga4_rc_off_pad1408_nosync.yaml` | 新增稳定候选配置，只改 padding/cache/save 路径。 |
| `/data/sejin/third_party/mindspeed-mm-26.0.0/examples/qwen3_5_audio/perf_tuning/ep8_mbs1_ga4_rc_off_pad1248.yaml` | 当前代码复跑新增 padding 1248 边界点，验证 1216-1280 区间 HBM/WPS。 |
| `/data/sejin/third_party/mindspeed-mm-26.0.0/examples/qwen3_5_audio/perf_tuning/ep8_mbs1_ga4_rc_off_pad1264.yaml` | 当前代码复跑新增 padding 1264 边界点，保持 16 对齐。 |
| `/data/sejin/third_party/mindspeed-mm-26.0.0/examples/qwen3_5_audio/perf_tuning/ep8_mbs1_ga4_rc_off_pad1280.yaml` | 当前代码复跑 `pad1280_current`，替换早期 pad1280 过期指标。 |
| `/data/sejin/third_party/mindspeed-mm-26.0.0/examples/qwen3_5_audio/perf_tuning/ep8_mbs1_ga4_rc_off_pad1536_nosync_fa2.yaml` | 验证 MindSpeed flash attention patch 后端，不改模型结构。 |
| `/data/sejin/third_party/mindspeed-mm-26.0.0/examples/qwen3_5_audio/perf_tuning/ep8_mbs2_ga2_rc_off_pad128_bucket_fa2.yaml` | 验证 mbs2 + FA2 短窗口吞吐和显存。 |
| `/data/sejin/third_party/mindspeed-mm-26.0.0/examples/qwen3_5_audio/perf_tuning/ep8_mbs1_ga4_rc_off_pad1408_lora64_nonexpert_nosync.yaml` | 验证 LoRA rank 64，并覆盖 full attention、linear attention、shared expert 的普通 Linear；专家融合 3D 张量不注入 LoRA。 |

---

## 3. 关键结果

HBM 按 `npu-smi` 采集 MB 展示，括号内为 MB/1000 近似 GB。

| 配置 | 状态 | step mean | input WPS | AICORE mean/peak | HBM mean/peak | Power mean/peak | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| `ep8_mbs1_ga4_rc_off_pad1024_pregather_nosync` | 成功 80/80 | 3.919s | 1414.6 | 18.47% / 38% | 48,753 / 49,773 MB | 163.06 / 197.7 W | 历史最高 WPS，但 HBM 只有 48.75GB。 |
| `ep8_mbs1_ga4_rc_off_pad1248` | 成功 80/80 | 4.413s | 1256.2 | 22.17% / 39% | 51,503 / 52,782 MB | 163.42 / 199.0 W | 当前代码边界点，HBM 仍低。 |
| `ep8_mbs1_ga4_rc_off_pad1264` | 成功 80/80 | 4.306s | 1287.6 | 23.07% / 39% | 51,711 / 53,002 MB | 166.39 / 202.2 W | 当前代码高 WPS，HBM 仍低。 |
| `ep8_mbs1_ga4_rc_off_pad1280_current` | 成功 80/80 | 4.279s | 1295.8 | 23.58% / 38% | 51,925 / 53,244 MB | 166.53 / 195.9 W | current-code 复跑中最高 WPS/AICORE，但 HBM 不达 55G。 |
| `ep8_mbs1_ga4_rc_off_pad1280_nosync` | 成功 80/80 | 4.365s | 1270.0 | 19.57% / 38% | 51,933 / 53,245 MB | 164.08 / 204.4 W | no_sync 对照，低于 current pad1280。 |
| `ep8_mbs1_ga4_rc_off_pad1408_nosync` | 成功 80/80 | 4.786s | 1158.3 | 22.48% / 40% | 54,638 / 56,111 MB | 162.57 / 203.7 W | HBM 接近目标，WPS 高于 pad1536。 |
| `ep8_mbs1_ga4_rc_off_pad1536_nosync_rerun05` | 成功 80/80 | 4.895s | 1132.5 | 23.43% / 43% | 56,401 / 58,059 MB | 164.54 / 205.6 W | 严格满足 HBM 55-60G 的最佳稳定配置。 |
| `ep8_mbs1_ga4_rc_off_pad1536_nosync_fa2` | 成功 80/80 | 4.971s | 1115.2 | 22.22% / 43% | 55,822 / 56,626 MB | 161.49 / 202.5 W | FA2 稳定但未提升 mbs1。 |
| `ep8_mbs1_ga4_rc_off_pad1408_lora64_nonexpert_nosync` | 成功 80/80 | 5.308s | 1044.5 | 21.92% / 40% | 57,523 / 59,671 MB | 159.94 / 200.4 W | rank64 非专家 LoRA 提高训练负载，但 WPS/AICORE/功耗均无收益。 |
| `ep8_mbs1_ga4_rc_off_pad2048` | OOM | 7.432s | 743.4 | 17.57% / 51% | 63,219 / 65,447 MB | 151.97 / 216.6 W | HBM 过高且 OOM，无收益。 |
| `ep8_mbs2_ga2_rc_off_pad128_bucket` | 失败 | 2.132s* | 2718.3* | 23.62% / 64% | 61,235 / 64,387 MB | 167.72 / 243.2 W | 短窗口快，但随后 OOM/挂起，不可用。 |
| `ep8_mbs2_ga2_rc_off_pad128_bucket_fa2` | 失败 | 1.938s* | 3287.8* | 14.94% / 57% | 53,585 / 55,878 MB | 156.07 / 216.6 W | FA2 显著提高短窗口 WPS、降低 HBM，但 step24 后挂起。 |
| `ep8_mbs2_ga2_rc_on_pad128_bucket_nosync_probe` | 失败 | 2.957s* | 2561.3* | 11.88% / 46% | 35,697 / 37,197 MB | 141.52 / 185.8 W | 重计算后 HBM 低，但 step14 后挂起。 |
| `ep8_mbs4_ga1_rc_on` | 失败 | N/A | N/A | N/A | N/A | N/A | step1 后无 AICORE、无日志推进，终止。 |

`*` 表示失败 run 中 warmup 后短窗口数据，只用于定位，不作为可上线配置。

### 3.1 相比原始 Baseline 的提升

原始可跑 baseline 采用 EP=1/FSDP2，全分片，60 step 成功；主口径为 step10-60：step time 6.910s、input WPS 804.2、吞吐 4.631 samples/s、端到端 npu-smi AICORE 9.5%、HBM 32.5GB、power 158.3W。早期 EP=8 原始配置不是有效性能 baseline：它完成配置解析、EP mesh、数据/tokenizer、LoRA 注入和 DCP load，但在首个 `optimizer.step()` 初始化 Adam 状态时 OOM，无法报告 post-warmup WPS 或训练窗口 AICORE。

按本报告的生产建议，最优稳定配置是 `ep8_mbs1_ga4_rc_off_pad1536_nosync`。它严格满足 HBM 55-60GB，相比原始 EP=1 baseline 的提升如下：

| 指标 | 原始 EP=1 baseline | `pad1536_nosync` 最优稳定配置 | 变化 |
|---|---:|---:|---:|
| step time | 6.910s | 4.895s | 降低 29.2%，约 1.41x 加速 |
| 吞吐 | 4.631 samples/s | 6.537 samples/s | +41.2% |
| input WPS | 804.2 | 1132.5 | +40.8% |
| AICORE mean | 9.5% | 23.43% | +13.93 pp |
| HBM mean | 32.5GB | 56.4GB | +73.5%，达到 55-60GB 目标 |
| Power mean | 158.3W | 164.5W | +3.9%，仍未达到 240W |

如果按 WPS 优先而不是严格 HBM 55-60GB，`pad1280_current` 是 current-code 复跑中最高推荐 WPS：step 4.279s、input WPS 1295.8、AICORE 23.58%、HBM 51.93GB、power 166.53W。相比原始 EP=1 baseline，step time 降低 38.1%，input WPS 提升 61.1%，但 HBM 不满足 55GB 下限。

最优配置解决的核心问题是：

1. **把早期不可用的 EP8 路径变成稳定训练路径。**  
   原始 EP8 在首个 optimizer step OOM；manual EP8 的 safe_open 逐张量加载和 fused expert dim=0 切片避免一次性加载全量专家权重到单卡，使 EP8 能稳定完成 80/80 step。专家权重结构、专家数量、MoE 路由和 Whisper encoder 均未改变。

2. **在吞吐提升的同时把 HBM 拉到目标区间。**  
   `pad1024_pregather_nosync` 历史 WPS 最高，为 1414.6，但 HBM 只有 48.75GB；`pad1280_current` current-code WPS 最高，为 1295.8，但 HBM 只有 51.93GB；`pad1536_nosync` 将 HBM 提到 56.40GB，满足 55-60GB，同时保留 1132.5 WPS。

3. **明确排除了几条无效优化路径。**  
   `mbs2/mbs4` 能提高短窗口 WPS 或峰值 AICORE，但会 OOM/挂起；FA2 对稳定 mbs1 无收益；rank64 非专家 LoRA 增加 backward/optimizer 时间，但没有提升平均 AICORE、功耗或 WPS；继续 padding 到 2048 会 HBM 越界并 OOM。

本轮实际优化过的参数和策略包括：manual EP8 expert slicing、`mbs/GA`、recompute 开关、`pad_to_multiple_of`、`gradient_accumulation_no_sync`、FA2、`mbs2/ga2` 与 `mbs4/ga1` 探索、LoRA rank/target 扩展、`chunk_loss_size`、dataloader workers，以及 `MULTI_STREAM_MEMORY_REUSE`、`TASK_QUEUE_ENABLE`、`PYTORCH_NPU_ALLOC_CONF`、`ACLNN_CACHE_LIMIT`、`CPU_AFFINITY_CONF` 等 runtime env。

---

## 4. Phase Timing

| 配置 | pregather | move | forward | backward | clip | optimizer | 主要瓶颈 |
|---|---:|---:|---:|---:|---:|---:|---|
| `pad1024_pregather_nosync` | 202.2 ms | 81.9 ms | 2049.7 ms | 1524.8 ms | 24.6 ms | 3.9 ms | forward/backward；WPS 最优但 HBM 低。 |
| `pad1280_current` | 0.0 ms | 179.0 ms | 2405.8 ms | 1608.0 ms | 49.3 ms | 3.9 ms | 当前最高 WPS；HBM 约 52GB。 |
| `pad1264` | 0.0 ms | 186.4 ms | 2433.1 ms | 1597.1 ms | 52.6 ms | 3.9 ms | 与 pad1280 接近，HBM 仍低。 |
| `pad1248` | 0.0 ms | 166.3 ms | 2551.0 ms | 1614.0 ms | 45.6 ms | 4.0 ms | 边界探测，吞吐不如 1264/1280。 |
| `pad1408_nosync` | 0.0 ms | 379.7 ms | 2636.4 ms | 1616.1 ms | 118.5 ms | 3.9 ms | padding 增加 move/forward/clip，换 HBM。 |
| `pad1536_nosync_rerun05` | 0.0 ms | 441.9 ms | 2662.7 ms | 1616.2 ms | 139.2 ms | 3.9 ms | 严格 HBM 达标，但 padding 继续拉低 WPS。 |
| `pad1536_nosync_fa2` | 0.0 ms | 431.0 ms | 2740.6 ms | 1630.2 ms | 133.2 ms | 3.9 ms | FA2 没有改善稳定 mbs1。 |
| `pad1408_lora64_nonexpert` | 0.0 ms | 210.8 ms | 2791.3 ms | 2166.1 ms | 59.9 ms | 23.0 ms | LoRA 负载主要增加 backward/optimizer，平均 AICORE 未升。 |

---

## 5. Run Phase Timing

代表性稳定 run 的初始化阶段如下；最终吞吐和硬件统计不包含这些阶段。

| 配置 | startup->manual_load | Qwen safe_open load | Whisper load | post_load->iter1_end | all logged steps | last_step->LoRA save |
|---|---:|---:|---:|---:|---:|---:|
| `pad1024_pregather_nosync` | 3.61s | 32.44s | 1.27s | 54.75s | 332.83s | 0.10s |
| `pad1280_current` | 3.63s | 38.13s | 1.30s | 36.63s | 361.29s | 0.10s |
| `pad1264` | 3.65s | 36.65s | 1.25s | 54.60s | 363.14s | 0.10s |
| `pad1248` | 3.52s | 30.13s | 1.26s | 59.94s | 372.27s | 0.10s |
| `pad1408_nosync` | 3.41s | 29.79s | 1.24s | 62.39s | 408.77s | 0.10s |
| `pad1536_nosync_rerun05` | 3.39s | 32.23s | 1.27s | 57.05s | 413.49s | 0.10s |
| `pad1536_nosync_fa2` | 3.38s | 38.05s | 1.34s | 57.15s | 416.83s | 0.10s |
| `pad1408_lora64_nonexpert` | 5.97s | 34.12s | 1.75s | 57.34s | 444.85s | 1.88s |

---

## 6. 结论

在“不改模型结构、LoRA-only、单机 8 卡 EP8、Qwen3.5-35B-A3B + Whisper-large-v3 audio”约束下，本轮没有找到同时满足 `AICORE 平均 40%+`、`HBM 55-60G`、`功耗平均 240W`、`高 WPS` 的稳定训练策略。

原因来自实测而不是初始化噪声：

1. **mbs2/mbs4 能提高短窗口 WPS 或峰值 AICORE，但不稳定。**  
   mbs2 rc_off 可短暂到 WPS 2718、AICORE peak 64%、power peak 243W，但 HBM 接近满卡并出现 OOM/挂起；mbs2+FA2 短窗口 WPS 提升到 3288、HBM 降到 53.6-55.9GB，但 step24 后仍挂起；mbs4 rc_on step1 后停在 0% AICORE。

2. **稳定配置只能落在 mbs1/ga4，current-code 复跑最高稳定 AICORE 仍只有 23.58%。**  
   复跑 `pad1248/1264/1280_current` 后，current-code 复跑组里最高 WPS 是 `pad1280_current`: WPS 1295.8、AICORE 23.58%、power 166.53W，但 HBM 只有 51.93GB。历史 `pad1024_pregather_nosync` WPS 更高，但 HBM 只有 48.75GB，不满足显存目标。继续用 padding 增 HBM 会增加 move/forward/clip 时间，`pad1536_nosync` 虽满足 HBM 55-60GB，但 WPS 降到 1132.5。FA2 对 mbs1 没有收益。

3. **扩大非专家 LoRA 训练负载不是有效解。**  
   rank16 attention-only 是 80 个 LoRA 张量、3.44M 可训练参数；rank64 非专家 LoRA 扩到 500 个 LoRA 张量、76.68M 可训练参数，并覆盖 full attention、linear attention、shared expert 的普通 Linear。该配置 HBM 达 57.52GB，但 AICORE 均值只有 21.92%、power 均值 159.94W、WPS 降到 1044.5。低秩小 GEMM 增多主要拉长 backward/optimizer，没有把 910B 平均功耗推到 240W。

4. **数据加载不是 warmup 后瓶颈。**  
   代表性稳定 run 的 `get_batch` 约 2ms；瓶颈集中在 forward/backward 和 padding 后的 move/clip。指标窗口均从第 11 步开始，未把 init、safe_open、数据构建、首步编译计入均值。

5. **早期 `pad1280` 高 HBM 结果已被当前复跑替代。**  
   旧 `pad1280` 曾记录 HBM 62.6GB、step 5.07s；当前代码复跑 `pad1280_current` 为 HBM 51.93GB、step 4.279s。最终推荐和结论采用 current 复跑结果。

---

## 7. 推荐配置

### 严格满足 HBM 55-60G

使用：

```bash
MASTER_PORT=6052 bash /data/sejin/baseline_26/scripts/run_audio_perf_experiment.sh \
  ep8_mbs1_ga4_rc_off_pad1536_nosync \
  /data/sejin/third_party/mindspeed-mm-26.0.0/examples/qwen3_5_audio/perf_tuning/ep8_mbs1_ga4_rc_off_pad1536_nosync.yaml \
  1500 10 1.0
```

指标：step 4.929s，input WPS 1124.8，AICORE 23.40%，HBM 56.38GB，power 163.39W。
复测 `rerun05` 指标：step 4.895s，input WPS 1132.5，AICORE 23.43%，HBM 56.40GB，power 164.54W。

### Current-Code 复跑最高推荐 WPS，不满足 HBM 目标

使用：

```bash
MASTER_PORT=6063 bash /data/sejin/baseline_26/scripts/run_audio_perf_experiment.sh \
  ep8_mbs1_ga4_rc_off_pad1280_current \
  /data/sejin/third_party/mindspeed-mm-26.0.0/examples/qwen3_5_audio/perf_tuning/ep8_mbs1_ga4_rc_off_pad1280.yaml \
  1500 10 1.0
```

指标：step 4.279s，input WPS 1295.8，AICORE 23.58%，HBM 51.93GB，power 166.53W。

### HBM 接近 55G，WPS 高于 pad1536

`pad1408_nosync`: step 4.786s，input WPS 1158.3，AICORE 22.48%，HBM 54.64GB，power 162.57W。

---

## 8. 后续如果必须冲 AICORE 40%+

这些项会超出本轮“不改训练语义/稳定 mbs1 EP8”的范围，需另开口径：

| 方向 | 预期 | 风险 |
|---|---|---|
| 多机或更多卡，保留 EP8 同时增加 DP/eFSDP 余量 | 允许更大 micro batch，可能提高 AICORE | 需要额外硬件和通信验证。 |
| 改训练范围，例如训练 audio_projector、Whisper LoRA 或专家/路由相关参数 | 增加反向计算和功耗 | 改变训练口径，且可能触及模型结构/数学口径。 |
| mbs2 稳定化专项，包括 allocator、bucket、长样本裁剪、HCCL timeout/通信定位 | 可能提高 WPS | 当前多次复现 OOM/挂起，风险较高。 |

本轮生产建议采用 `pad1536_nosync`；若更看重吞吐且接受 HBM 约 52GB，采用 `pad1280_current`；若希望 HBM 更接近 55GB 且接受较低 WPS，采用 `pad1408_nosync`。
