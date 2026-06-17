# MindSpeed-MM 26.0.0 源码改动补丁

本目录存放对 **MindSpeed-MM 26.0.0** 框架源码的全部改动，以 patch 形式归档，
用于在干净的上游源码上一键复现 MC2 通信-计算重叠优化 + Qwen3.5-35B 音频 LoRA 训练能力。

> 设计原则：不 vendor 整个 221M 框架，只保留「相对上游的差异」。patch 体积小、可 review、
> 版本锁定，clone 上游 → apply patch 即可还原我们的训练环境。

---

## 一、版本锚点

| 项 | 值 |
|---|---|
| **上游仓库** | `https://gitcode.com/Ascend/MindSpeed-MM.git` |
| **base commit（纯净 26.0.0）** | `f79852d83f1aaf34266ee7c4fa356dc018425c19` |
| base subject | `DOCS：修改安装PyTorch章节名称` (2026-05-15) |
| **改动 commit** | `46de4e182dfa0895a40add04e6f43d523dc65f49` |
| 改动 subject | `Add Qwen3.5 35B LoRA audio training setup` |

所有 patch 均基于 `f79852d8` 生成，已验证 `git apply --check` 干净通过。

---

## 二、patch 文件说明

| 文件 | 体积 | 内容 |
|---|---|---|
| `01_source_code.patch` | 90K | **框架源码改动**：`mindspeed_mm/**/*.py` + `verl_plugin/**/*.py`（13 改 + 6 新） |
| `02_examples_configs.patch` | 343K | **配置与示例**：`examples/qwen3_5*/`（训练脚本 + 221 个 perf_tuning yaml） |
| `00_full_commit_46de4e18.patch` | 1.4M | **全量兜底**：整个 commit 的 format-patch（含上面两者 + `.gitignore` 等），权威全量参照 |
| `03_pack_source.patch` | 16K | **LLM Pack 格式源码改动**：`modeling_qwen3_5_audio.py`（forward 支持 pack）+ `packed_collator_wrapper.py`（新增）+ `data_collator.py`（注册 collator） |
| `04_pack_examples.patch` | 20K | **Pack 配置与脚本**：`ep8_mbs1_ga4_rc_off_pack.yaml` + `smoke_test_pack.sh` + `test_pack_cpu.py` + `PACK_FORMAT_QUICKSTART.md` |
| `05_audio_wps4000_runtime.patch` | 36K | **语音多模态 WPS4000 增量**：balanced pack sampler、DP pack shape 对齐、MoE runtime dispatcher override、最终 mbs4 pack 配置和 8 卡复现脚本 |

通常 `01` + `02` 已覆盖复现所需的全部内容；`00` 用于完整还原或核对。
`03` + `04` 为 **LLM pad→pack 优化**增量（基于 `01`+`02` 之上，即 base commit `46de4e18`），
显存 −27% / 每步 −21%，详见 `reports/qwen35_audio_llm_pack_perf_20260616.md`。
`05` 为本分支的 **8 卡语音多模态 WPS4000** 增量，基于 `01`+`02`+`03`+`04` 之后的源码树应用；
在 `172.29.226.188` 上完整 80 step 复跑验证，平均 input WPS `4195.787`，详见
`reports/qwen35_audio_wps4000_result_20260617.md`。

### 2.1 源码改动清单（01_source_code.patch）

**MC2 通信-计算重叠核心**：
- `mindspeed_mm/fsdp/distributed/expert_parallel/ep_dispatcher.py` (+172) — 新增 MC2 dispatcher 逻辑
- `mindspeed_mm/fsdp/distributed/expert_parallel/expert_parallel.py` (+8) — 透传 `dispatcher` 参数
- `mindspeed_mm/fsdp/models/qwen3_5_moe/modeling_qwen3_5_moe.py` (+41) — `ep_forward` 增加 mc2 分支

**训练流程 / 数据 / 优化器**：
- `mindspeed_mm/fsdp/train/train_engine.py` (+228)
- `mindspeed_mm/fsdp/train/trainer.py` (+73)
- `mindspeed_mm/fsdp/data/dataloader/{dataloader,sampler}.py` — 分布修复（distfix 数据）
- `mindspeed_mm/fsdp/params/training_args.py` (+62) — 新增并行/dispatcher 配置项
- `mindspeed_mm/fsdp/optimizer/clip_grad_norm.py`、`utils/lora_weight_manager.py`
- `mindspeed_mm/fsdp/data/data_utils/func_utils/{convert,model_args}.py`
- `mindspeed_mm/fsdp/models/qwen3_5/triton/utils.py`

**Qwen3.5 音频插件（6 个新增）**：
- `mindspeed_mm/fsdp/models/qwen3_5_audio/` — `whisper_encoder.py` / `projector.py` /
  `manual_ep.py` / `modeling_qwen3_5_audio.py` / `convert_weights.py` / `__init__.py`

### 2.2 配置与示例（02_examples_configs.patch）

- `examples/qwen3_5/` — 35B 纯文本 LoRA（`finetune_*.sh`、`*_optimal.yaml`、`run_35B_lora_pipeline.sh`）
- `examples/qwen3_5_audio/` — 音频训练 README、demo 数据、dist/distfix 配置
- `examples/qwen3_5_audio/perf_tuning/*.yaml` — **EP8 调优全部 yaml 配置**
  （`ep8_mbs1_ga4_rc_off_pad1536_nosync_mc2.yaml` 等，对应 `reports/perf_runs/*.md` 报告）

---

## 三、复现步骤

```bash
# 1. clone 上游 MindSpeed-MM 并锁定到我们的 base commit
git clone https://gitcode.com/Ascend/MindSpeed-MM.git
cd MindSpeed-MM
git checkout f79852d83f1aaf34266ee7c4fa356dc018425c19

# 2. 应用我们的改动（二选一）

#  方式 A：分步应用（推荐，便于核对）
git apply --check  /path/to/baseline_26/mindspeed_mm_patches/01_source_code.patch
git apply          /path/to/baseline_26/mindspeed_mm_patches/01_source_code.patch
git apply          /path/to/baseline_26/mindspeed_mm_patches/02_examples_configs.patch

#  可选：叠加 LLM pad→pack 优化（基于 01+02 之上）
git apply          /path/to/baseline_26/mindspeed_mm_patches/03_pack_source.patch
git apply          /path/to/baseline_26/mindspeed_mm_patches/04_pack_examples.patch

#  可选：叠加语音多模态 WPS4000 优化（基于 01+02+03+04 之上）
git apply          /path/to/baseline_26/mindspeed_mm_patches/05_audio_wps4000_runtime.patch

#  方式 B：全量还原（含 .gitignore 等，保留原作者信息）
git am             /path/to/baseline_26/mindspeed_mm_patches/00_full_commit_46de4e18.patch

# 3. 验证关键文件就位
ls mindspeed_mm/fsdp/distributed/expert_parallel/ep_dispatcher.py
ls examples/qwen3_5_audio/perf_tuning/ep8_mbs1_ga4_rc_off_pad1536_nosync_mc2.yaml
```

应用后，配合 `baseline_26/scripts/` 的训练脚本与 `reports/` 的配置说明即可复现全部实验。

---

## 四、与 baseline_26 报告的对应关系

| baseline_26 内容 | 依赖的源码改动 |
|---|---|
| `reports/moe_optimization_strategy_from_blog_20260616.md` | `ep_dispatcher.py` / `expert_parallel.py` / `modeling_qwen3_5_moe.py` 的 MC2 分支 |
| `reports/perf_runs/ep8_*.md` (38 份) | `examples/qwen3_5_audio/perf_tuning/*.yaml` |
| `scripts/verify_mc2_equivalence.py` | MC2 dispatcher 源码 |
| `scripts/run_qwen35_audio_moe_blog_tuning_suite.sh` | perf_tuning yaml + 训练入口 |
| v3 音频分布式数据报告 | `dataloader.py` / `sampler.py` 分布修复 |

---

## 五、重新生成 patch（源码再有改动时）

```bash
cd /data/sejin/third_party/mindspeed-mm-26.0.0
BASE=f79852d83f1aaf34266ee7c4fa356dc018425c19
OUT=/data/sejin/baseline_26/mindspeed_mm_patches

git diff $BASE HEAD -- 'mindspeed_mm/**/*.py' 'verl_plugin/**/*.py' > $OUT/01_source_code.patch
git diff $BASE HEAD -- 'examples/**'                                > $OUT/02_examples_configs.patch
git format-patch -1 HEAD --stdout                                  > $OUT/00_full_commit_$(git rev-parse --short HEAD).patch
```
