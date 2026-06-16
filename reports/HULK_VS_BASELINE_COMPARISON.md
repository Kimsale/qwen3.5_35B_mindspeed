# Hulk Qwen3-Omni-30B vs 当前基线（MindSpeed-LLM 26.0.0 Qwen3-30B-A3B）配置与数据对比

**日期**: 2026-06-03
**对比对象**:

- **Hulk**: 自研 Theta 框架 + Qwen3-Omni-30B（dense base + 音频 encoder，本次纯文本训练冻 encoder）
- **当前基线**: MindSpeed-LLM 26.0.0 + Qwen3-30B-**A3B**（MoE，纯文本路径，提取自 Qwen3-Omni-30B-A3B-Captioner 的 thinker text 子模块）

> **架构勘正**：之前我把 Hulk 模型描述为"Qwen3-30B dense"，**已修正**。Hulk 用的是 **Qwen3-Omni-30B**（即 Qwen3-30B base + 额外 audio encoder），本次纯文本训练时通过 `model.fix_encoder=True` 冻结音频塔，只训 LoRA。当前基线用的是 **Qwen3-30B-A3B**（MoE 变体，128 个 expert / topk 8），从 Qwen3-Omni-30B-A3B-Captioner 抽取了 thinker text-only 子模块。**两者底座虽都属 Qwen3 30B 家族，但 Hulk 是 dense / 基线是 MoE-A3B**——这是后续所有差异的根源。

---

## 第一部分：训练参数配置对比

### 1. LoRA 超参


| 维度           | Hulk Qwen3-Omni-30B                           | 当前基线                                                       | 是否一致                            |
| -------------- | --------------------------------------------- | -------------------------------------------------------------- | ----------------------------------- |
| LoRA rank      | **32**                                        | 16                                                             | ❌ 基线一半                         |
| LoRA alpha     | **64**                                        | 32                                                             | ❌ 基线一半 (alpha/rank 比都是 2.0) |
| LoRA dropout   | **0.1**                                       | 0（默认）                                                      | ❌                                  |
| Target modules | **Q, K, V, O**（仅注意力）                    | `linear_qkv linear_proj linear_fc1 linear_fc2`（注意力 + MLP） | ❌ 基线多训 MLP                     |
| 可训练参数量   | （未直接给出，按 4×rank32 attn 估约 60-90M） | **135.66M（peft 实测打印）**                                   | ❌ 基线多约 ~50M                    |
| 冻结策略       | encoder + LLM 主体 + 词嵌入冻结               | LoRA 自动冻结主体                                              | ✅ 概念一致                         |

**说明**：基线的 `linear_qkv` 是 Megatron 把 Q/K/V 三个矩阵融合后的单一权重；Hulk 拆开训 Q/K/V/O。注意基线**额外训了 MLP 的 fc1/fc2**（包括 MoE 每个 expert 的 MLP），这才是基线 trainable params 偏多的主因。

### 2. 分布式并行配置


| 维度                 | Hulk             | 基线           | 是否一致                        |
| -------------------- | ---------------- | -------------- | ------------------------------- |
| 卡数                 | 单机 8×910B     | 单机 8×910B3  | ✅                              |
| TP（张量并行）       | **1**            | **2**          | ❌                              |
| PP（流水并行）       | 1                | 1              | ✅                              |
| EP（专家并行）       | 8                | **4**          | ❌（基线模型是 MoE，必须开 EP） |
| CP（上下文并行）     | **2（Ulysses）** | 1              | ❌                              |
| DP（数据并行，派生） | 4 = 8/(TP×CP)   | 4 = 8/(TP×PP) | ✅ 数值一致但来源不同           |

**说明**：Hulk 用 CP=2 切分 8192 长序列；基线用 EP=4 切 MoE 专家。**通信模式完全不同**——Hulk 是 Ulysses all-to-all（按 head 切），基线是 MoE alltoall（按 token 切到专家组）。

### 3. ZeRO / 优化器分片


| 维度            | Hulk                                 | 基线                                                                       | 是否一致           |
| --------------- | ------------------------------------ | -------------------------------------------------------------------------- | ------------------ |
| ZeRO stage      | **Stage-2 (`os_v2`)** 优化器状态分片 | `--use-distributed-optimizer`（Megatron 等价 ZeRO-1）                      | ❌ Hulk 更激进分片 |
| zero_shard_size | 8（分到 8 卡）                       | 由 distributed-optimizer 自动管理                                          | -                  |
| CPU offload     | **否**（纯 GPU 分片）                | **是**（`--swap-optimizer --swap-optimizer-times 32`，每步 H2D/D2H 32 段） | ❌ 关键差异        |

**说明**：基线被迫开 swap-optimizer 是因为 30B-A3B 优化器状态理论 ≈ 130GB（日志 L1730: `weight and optimizer=130978.71 MB`），单卡 65GB 装不下，只能往 CPU 卸。这是基线单步 12.4s 偏高的主因之一。Hulk 在纯 GPU 内分片即可，无 H2D/D2H 开销。

### 4. 序列与 Batch


| 维度                         | Hulk                                                     | 基线                              | 是否一致                     |
| ---------------------------- | -------------------------------------------------------- | --------------------------------- | ---------------------------- |
| seq_length                   | **8192**                                                 | 4096                              | ❌ Hulk 2 倍                 |
| max_position_embeddings      | 8192                                                     | 4096                              | ❌                           |
| Pack 策略                    | **动态 pack**（`max_tokens=16000`，`sort_by_size=True`） | 无动态 pack，**固定 pad 到 4096** | ❌                           |
| max_tokens_per_sentence.text | 2000                                                     | 不适用（无 pack）                 | -                            |
| 等效每 step token 量         | ≤16000（动态控制）                                      | 16 samples × 4096 =**65536**     | ❌ 基线 4 倍但大量是 padding |
| 有效 token 占比              | ~95%+（pack 后基本无浪费）                               | **~2.2%**（88/4096）              | ❌ 基线浪费 ~98%             |

### 5. 重计算


| 维度                  | Hulk                                       | 基线                | 是否一致    |
| --------------------- | ------------------------------------------ | ------------------- | ----------- |
| recompute_granularity | full                                       | full                | ✅          |
| recompute method      | （Hulk 未明确，应是默认 uniform 或 block） | block, num_layers=1 | ⚠️ 差异小 |

### 6. 精度 / Loss Scale


| 维度       | Hulk                  | 基线                                                                                            | 是否一致      |
| ---------- | --------------------- | ----------------------------------------------------------------------------------------------- | ------------- |
| 主精度     | BF16                  | BF16                                                                                            | ✅            |
| FP16       | False                 | False                                                                                           | ✅            |
| Loss scale | 自适应（BF16 不需要） | `--initial-loss-scale 4096`（Megatron 框架强制设置，BF16 下实际不生效，日志 `loss scale: 1.0`） | ⚠️ 等效一致 |

### 7. 优化器


| 维度         | Hulk                            | 基线                                                                                           | 是否一致                |
| ------------ | ------------------------------- | ---------------------------------------------------------------------------------------------- | ----------------------- |
| 优化器       | AdamW                           | Adam（Megatron`optimizer=adam`，但实际 fused_adamw，见日志 `optimizer_selection=fused_adamw`） | ✅ 等效 AdamW           |
| β1 / β2    | 0.9 / 0.95                      | 0.9 / 0.95                                                                                     | ✅                      |
| eps          | 1e-8                            | 1e-8（Megatron 默认）                                                                          | ✅                      |
| weight_decay | 0.1                             | 0.1                                                                                            | ✅                      |
| **lr**       | **5e-6**                        | **1.25e-5**（基线 2.5 倍）                                                                     | ❌                      |
| 学习率调度   | Cosine, warmup=0.0, min_lr=1e-6 | Cosine, warmup_fraction=0.01, min_lr=1.25e-7                                                   | ⚠️ warmup/min_lr 不同 |

### 8. 梯度裁剪


| 维度                      | Hulk    | 基线    | 是否一致         |
| ------------------------- | ------- | ------- | ---------------- |
| `clip_grad` / `clip_norm` | **5.0** | **1.0** | ❌ Hulk 5 倍宽松 |

### 9. Checkpoint


| 维度     | Hulk            | 基线（性能测试模式）                          |
| -------- | --------------- | --------------------------------------------- |
| 保存间隔 | 每 5000 updates | 禁用（`--save-interval 999999`，无 `--save`） |

### 10. 算子融合 / 加速（基线特有，Hulk 未列）


| flag                                       | 基线状态               |
| ------------------------------------------ | ---------------------- |
| `--use-flash-attn`                         | ✅ 昇腾 FA             |
| `--use-fused-rotary-pos-emb`               | ✅                     |
| `--no-rope-fusion`                         | ✅（**MC2 规避必需**） |
| `--use-fused-swiglu`                       | ✅                     |
| `--use-fused-rmsnorm`                      | ✅                     |
| `--no-masked-softmax-fusion`               | 关                     |
| `--moe-grouped-gemm`                       | ✅                     |
| `--moe-permutation-async-comm`             | ✅                     |
| `--moe-token-dispatcher-type alltoall_seq` | ✅                     |

Hulk 文档未列具体算子融合配置（应该都开了昇腾 FA / fused-norm 等，但 MoE 相关无关）。

### 11. 环境（基线特有锁定）


| 变量                            | 基线值                          | 作用                                      |
| ------------------------------- | ------------------------------- | ----------------------------------------- |
| CANN                            | **8.5.0**（锁定，禁用系统 8.1） | set_env.sh from`cann-8.5.0`               |
| `TORCH_DEVICE_BACKEND_AUTOLOAD` | **0**                           | 必须显式 import torch_npu（本机硬性要求） |
| `PYTORCH_NPU_ALLOC_CONF`        | `expandable_segments:True`      | 显存分段扩展                              |
| `HCCL_CONNECT_TIMEOUT`          | 1800                            | HCCL 通信超时                             |
| `CUDA_DEVICE_MAX_CONNECTIONS`   | 1                               | 通信流串行（确定性）                      |

---

## 第二部分：训练数据样本长度分布对比

### Hulk 数据集（jilian_fast_common_maxtok_2k.mdb）

- **路径**: `/b3yc-home/asrprg/glzhong/data/jsonl_mdb/jilian_fast_common_maxtok_2k.mdb`
- **格式**: LMDB（从 jsonl 转换）
- **模态**: 纯文本（无音频）
- **样本数**: **382,746**


| 统计            | 值                 |
| --------------- | ------------------ |
| min / max       | 21 / 2048          |
| mean            | **614.20**         |
| median (P50)    | **466**            |
| std             | 460.32             |
| P10 / P25 / P75 | 158 / 277 / 837    |
| P90 / P95 / P99 | 1347 / 1622 / 1945 |

**长度区间**:


| 区间         | 样本数      | 占比               |
| ------------ | ----------- | ------------------ |
| 0–100       | 15,343      | 4.01%              |
| 100–200     | 42,820      | 11.19%             |
| **200–500** | **146,756** | **38.34%** ← 最多 |
| 500–1000    | 105,670     | 27.61%             |
| 1000–1500   | 45,281      | 11.83%             |
| 1500–2000   | 25,141      | 6.57%              |
| 2000–2048   | 1,735       | 0.45%              |

特征：分布跨度大（21–2048），中位 466，约 66% 集中在 200–1000，长尾 6.57% 超过 1500。属于真实业务对话/指令数据。

### 当前基线数据集（qwen3_sft_packed）

- **路径**: `/data/sejin/data/qwen3_sft_mcore/qwen3_sft_packed_input_ids_document.{bin,idx}`
- **格式**: Megatron IndexedDataset 二进制（packed）
- **来源**: 直接复制自 `/data/xuchen2/train/data/qwen3_8b_zh_sft_mcore/`（md5 一致，原本是给 Qwen3-8B 准备的 SFT smoke 数据，被复用到 30B-A3B）
- **模态**: 纯文本（中文 ChatML 格式 QA）
- **内容**: 1024 条合成的"大模型训练知识"中文问答（张量并行/梯度累积/LoRA 等话题），明显是 benchmark 烟雾数据
- **样本数**: **1024**


| 统计            | 值              |
| --------------- | --------------- |
| min / max       | **81 / 103**    |
| mean            | **88.38**       |
| median (P50)    | **88**          |
| std             | 6.58            |
| P10 / P25 / P75 | 81 / 83 / 91    |
| P90 / P95 / P99 | 103 / 103 / 103 |

**长度区间**:


| 区间       | 样本数  | 占比                   |
| ---------- | ------- | ---------------------- |
| **0–100** | **896** | **87.50%** ← 几乎全部 |
| 100–200   | 128     | 12.50%                 |
| 200–500   | 0       | 0%                     |
| 500+       | 0       | 0%                     |

特征：长度极短且高度集中（mean 88，std 仅 6.58，全部 ≤ 103），单条只占 seq_length 4096 的 ~2%。

### 数据分布差异汇总


| 维度                    | Hulk                          | 基线                            | 倍数            |
| ----------------------- | ----------------------------- | ------------------------------- | --------------- |
| 样本数                  | 382,746                       | 1,024                           | Hulk**374×**   |
| mean 长度               | 614                           | 88                              | Hulk**7×**     |
| max 长度                | 2048                          | 103                             | Hulk**20×**    |
| 长度跨度 (max−min)     | 2027                          | 22                              | Hulk**92×**    |
| std                     | 460.32                        | 6.58                            | Hulk**70×**    |
| 等效训练 token 总量     | 382K × 614 ≈**235M** tokens | 1024 × 88 ≈**90K** tokens     | Hulk**2614×**  |
| Pack 后每 step 有效样本 | 8–12（pack 后 ~95% 有效）    | 16（fixed pad，~2.2% 有效）     | -               |
| 每 step 有效 tokens     | ~7800（pack）                 | ~1408（基线 16 样本×88 token） | Hulk**5.5×**   |
| 每 step 名义 tokens     | ~16000（max_tokens 上限）     | 65536（gbs×seq）               | 基线 4× 但虚高 |

---

## 第三部分：差异的工程影响

### 1. 不可直接对比单步耗时 / 吞吐

基线 12.4s/step（gbs=16, 65536 名义 tokens）vs Hulk 单步耗时（未给）—— **数据特征、模型架构、并行策略、ZeRO 实现全部不同，单步耗时无可比性**。基线 WPS 5258 是按 padding 后 4096 算的名义值，**有效 token 吞吐仅约 113/s**；而 Hulk 在 pack 后真实吞吐应当远高于此（具体值需 Hulk 侧实测数据）。

### 2. 基线数据规模不足以反映真实训练负载

1024 条 × 88 token 的 smoke 数据本是为"跑通链路 + 测吞吐稳态"用的，**不能代表任何真实训练分布**。loss 在 15 步内从 2.20 降到 1.82 也是这个原因（小数据反复过拟合）。

### 3. MoE vs Dense 是最大的架构差

- **基线（MoE-A3B）**：每 step 激活 8/128 = 6.25% 专家，理论 FLOPs 远低于 dense 30B；但 MoE 通信（alltoall）成本高，需 EP 切分。算力实际利用率受通信和路由开销影响大。
- **Hulk（dense 30B）**：每 step 全参数参与计算，FLOPs 是 A3B 的 ~16 倍，但通信简单（无 alltoall），加上 ZeRO-2 纯 GPU 分片，单 token 计算更"纯"。

这意味着即使在同等 token 吞吐下，**两套训练消耗的计算量差距巨大**——A3B 是稀疏激活专家网络，dense 30B 是稠密计算，**两者对 NPU 算力的压榨方式根本不同**。

### 4. swap-optimizer 是基线被迫的选择

基线 30B-A3B 的优化器状态理论 130GB，必须 swap 到 CPU；而 Hulk 在 ZeRO-2 + 纯 GPU 分片下能装下（dense 30B 优化器 ≈ 60-70GB，分到 8 卡每卡 8-9GB），**没有 H2D/D2H 开销**。

### 5. LoRA 范围对算力压力不同

- 基线 target = QKV+Proj + MLP fc1/fc2（含 MoE expert MLP），LoRA forward/backward 路径更长 → 更贴近全参的算力分布
- Hulk target = Q/K/V/O，仅注意力层 → 算力主要在 attention，MLP 走主体（已冻结）的 fast path

---

## 第四部分：要做对标基线，需要怎么改

如果要把当前基线**严格对齐 Hulk** 做框架对标评测，需要的改动：


| # | 改动项                                                           | 工作量                                      |
| - | ---------------------------------------------------------------- | ------------------------------------------- |
| 1 | **换模型**：Qwen3-Omni-30B                                       | **大**（需重新转 dense 权重 + 改模型 spec） |
| 2 | **改并行**：TP=1, CP=2 (ulysses_cp_algo), EP=8                   | 中（需重转 ckpt + 改训练脚本）              |
| 3 | **去掉 swap-optimizer**：用纯 distributed-optimizer              | 小（可能 OOM 需重新评估）                   |
| 4 | **改 LoRA**：rank=32, alpha=64, dropout=0.1, target 仅 attention | 小                                          |
| 5 | **改序列**：seq=8192, 启用动态 pack（max_tokens=16000）          | 中（需 preprocess pack）                    |
| 6 | **换数据**：用真实长度分布的数据集（mean ~600，max 2048）        | 中（需准备/转换数据）                       |
| 7 | **改超参**：lr=5e-6, clip=5.0, warmup=0.0, min_lr=1e-6           | 小                                          |

特别注意：

- ** Qwen3-30B 在 8 卡 + TP1 + CP2 下能否装下需重新评估**.
- **昇腾 Ulysses CP** 在 CANN 8.5.0 / MindSpeed 26.0.0 是否完整可用、是否触发 MC2 类崩溃需要实测。
- 数据 pack 流程需要走 MindSpeed-LLM 的 `preprocess_data.py --pack` 路径，与 Hulk 的 LMDB 动态 pack 实现细节可能不一致。

---

## 第五部分：来源与可追溯性


| 内容                  | 来源                                                                                                                 |
| --------------------- | -------------------------------------------------------------------------------------------------------------------- |
| Hulk 配置             | 用户提供（基于`theta_code_cpV3/models/audioLLM.py:681-687`、`config/train_stage1.yaml`、`run_lora.cp2.onlytext.sh`） |
| Hulk 数据分布         | 用户提供（基于`jilian_fast_common_maxtok_2k.mdb` 统计）                                                              |
| 基线训练参数          | `/data/sejin/baseline_26/scripts/train_param.sh` + `verify_mc2fix.log` 实际生效值（已交叉核对）                      |
| 基线 trainable params | `opt_R5_actrecomp.log` L1050（peft `print_trainable_parameters`）                                                    |
| 基线优化器内存        | `opt_R3c.log` L1730（Megatron 启动打印 `weight and optimizer=130978.71 MB`）                                         |
| 基线数据真实分布      | 直接通过`megatron.core.datasets.indexed_dataset.IndexedDataset` 读取 `qwen3_sft_packed_input_ids_document` 计算      |
| 基线数据来源          | md5 比对确认是`/data/xuchen2/train/data/qwen3_8b_zh_sft_mcore/` 的副本（原本给 Qwen3-8B 用的 smoke 数据）            |
| 基线数据内容          | 用`AutoTokenizer.decode` 解码前 3 条样本，确认为 ChatML 格式中文 QA                                                  |

---

**结论一句话**：当前基线（MoE A3B + 极短 smoke 数据 + swap-optimizer + 静态 pad）**不能直接作为 Hulk（dense Omni + 真实长度分布数据 + 纯 GPU ZeRO-2 + 动态 pack）的对标参考**——两者在模型架构、并行拓扑、显存策略、数据分布五个维度都有根本差异，单步耗时和吞吐数据无可比性。要对标必须做上述 7 项改动。
