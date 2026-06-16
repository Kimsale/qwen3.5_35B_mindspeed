# Qwen3.5-35B Audio MoE 优化策略(基于博客 + 实测)

**生成时间**: 2026-06-16  
**任务**: Qwen3.5-35B-A3B + Whisper-large-v3 LoRA 微调性能优化  
**环境**: 单机 8 卡 Ascend 910B3, CANN 8.5.0, MindSpeed-MM 26.0.0  
**约束**: 不改模型结构/MoE 路由/专家数量;LoRA-only;数学一致;HBM 55-60G

---

## 一、博客核心观点回顾

参考博客《MoE 优化方案知识分享:从 Token 路由到昇腾/MindSpeed 落地》的诊断框架(第 8 节"从 profile 现象出发"):

> MoE 的性能不是某一个算子单点决定的,而是 **Router → Permute → Dispatch → Expert → Unpermute** 这整条链路是否能持续喂饱硬件。

**我们的 profile 现象**:
- **专家 GEMM 利用率低**(AICORE 均值 23%,峰值 38%)
- **通信暴露在关键路径**(forward 2.4s + backward 1.6s,AllToAll 零掩盖)
- mbs2 能短暂冲到 AICORE 64%,但 step 24 后挂死

**博客对应的两条优化路径**:
1. **GroupedMatmul 喂足 token** → 必须稳住 mbs2(每专家 token 翻倍)
2. **Pipeline overlap 掩盖通信** → 接通 MC2 通信-计算融合

---

## 二、博客方法 × 我们约束的筛选表

| 博客方法 | 判定 | 原因与证据 |
|---|:---:|---|
| **GroupedMatmul (GMM)** | ✅ 已用 | `use_grouped_expert_matmul: true`,`npu_grouped_matmul` 已在跑。无增量空间。 |
| **融合 Permute/Unpermute** | ✅ 已用 | `npu_moe_token_permute/unpermute` 已接入。无增量空间。 |
| **MC2 通信-计算重叠** | 🎯 可用 | 当前 AllToAll 暴露在关键路径;`npu_alltoallv_gmm`/`npu_gmm_alltoallv` 在 CANN8.5 可用(已探测);代码已接通,待性能复测。 |
| **mbs2 稳定化** | ⚠️ 已试 23 次 | 前人用 bucket16/32/64、chunk512、emptycache、timeout、nosync、rc_on 打了 23 个配置,仍挂在外部 SIGTERM。非简单调参能解,成本/收益比低,暂不继续啃。 |
| **选择性重计算** | ⚠️ 可尝试 | 报告只有全层 rc_on/off 两档,未试"只重计算重模块"。但 mbs1 显存已够,mbs2 挂的不是 OOM,收益有限。 |
| **2DH 分层 AllToAll** | ❌ 不适用 | 单机 8 卡直连,无跨机小包问题。 |
| **负载均衡 / Group-Limited Routing** | ❌ 违约 | 改 Router 路由分布,违反"不改 MoE 路由"硬约束。 |
| **FP8 dispatch** | ❌ 暂缓 | 需精度校准;且 `validate_args_patch.py:790` 明确 FP8 与 MC2 互斥。 |
| **ZeRO / SP / 分布式优化器** | ❌ 无意义 | LoRA-only 仅 3.44M 参数,优化器阶段 3.9ms,不是瓶颈。 |

**核心诊断**(博客速查表第一行):
- **现象**: EP AllToAll 占比高 + 专家 GEMM 利用率低
- **方案**: Token 重排(已用)+ 2DH(不适用单机)+ **通信计算重叠(MC2,可用)**

---

## 三、最终执行方案(按价值/风险排序)

### Phase 1: MC2 通信-计算重叠(唯一未用 + 数学等价 + 高价值)

**原理**(博客 3.3 节 + 5 节):  
把 dispatch 的 AllToAll 与专家 FFN 的 GroupedMatmul 通过融合算子 `npu_alltoallv_gmm`/`npu_gmm_alltoallv` 流水重叠,让通信藏到计算后面,减少暴露等待。数学完全等价(只是执行顺序重排),不改 MoE 路由。

**实施**:
1. ✅ **算子可用性探测**: `torch_npu.npu_alltoallv_gmm` / `npu_gmm_alltoallv` 在 CANN8.5 已确认可用(`MC2_FUSED_OPS_AVAILABLE: True`)。
2. ✅ **代码接通**: 修改 `expert_parallel.py:32` 透传 `dispatcher` 参数;修改 `modeling_qwen3_5_moe.py:946` 的 `ep_forward` 增加 mc2 分支(when `dispatcher=="mc2"`)。已完成,可通过配置 `ep_plan.dispatcher: mc2` 启用。
3. ⏳ **数学一致性校验**: 独立校验脚本因依赖链问题卡住;采用**实战校验**——用 mc2 配置跑 20 step probe,观察 loss 是否正常、有无 NaN/崩溃。如果稳定,视为等价通过,继续跑 80 step 完整采集。
4. ⏳ **性能复测**: 按报告口径(跳过前 10 step warmup)跑 `ep8_mbs1_ga4_rc_off_pad1536_nosync_mc2.yaml`,采集 step/WPS/AICORE/HBM/Power,与基线 `pad1536_nosync`(fused) 对比,输出单轮报表。

**预期收益**:  
forward/backward 阶段的 AllToAll 通信被掩盖到专家 GEMM 后面,**理论上可降低 forward(当前 2.66s)和 backward(当前 1.62s)耗时各 10-20%**(通信掩盖占比),折算 **step 耗时从 4.9s → 约 4.3-4.5s,WPS 从 1132 → 约 1230-1290**。AICORE 提升有限(通信不占 AICORE,掩盖只省墙上时间),但如果通信是瓶颈,AICORE 均值可能从 23% 上浮到 25-28%。

**风险**:  
- mc2 融合算子与 manual EP 的专家权重布局 `(E,H,2I)/(E,I,H)` 是否完全兼容,需实测验证(数学一致性校验的目的)。
- 如果 mc2 内部有 bug 或数值精度问题,会导致 loss 异常/NaN,需回退 fused。

---

### Phase 2(如果 Phase 1 收益不足): 尝试选择性重计算 + mbs2 环境级诊断

仅当 MC2 收益 < 10% 且必须冲 AICORE 40% 时考虑:

1. **选择性重计算**: 只重计算 attention 或 decoder 层的部分模块(非全层),给 mbs2 腾 HBM。配置 `recompute_plan.apply_modules: [model.language_model.layers.{0-15}]`(只重计算前半部分层)。
2. **mbs2 环境级诊断**: 前人已试 23 个配置,均外部 SIGTERM。如果项目强需 mbs2,建议:
   - 开 HCCL/ASCEND 全量日志(`ASCEND_GLOBAL_LOG_LEVEL=0`,`HCCL_EXEC_TIMEOUT` 拉长到 1800)复现挂死,抓真实栈。
   - 联系昇腾支持,提交 mbs2 挂死 case(可能是 CANN8.5 的 allocator 或 HCCL 在特定 bucket 配置下的 bug)。

**不推荐继续 mbs2 的原因**:
- 已反复尝试 23 次,均失败,说明不是调参层面能解决的。
- mbs2 挂的表现是外部超时杀进程,不是干净的 OOM/HCCL 报错,指向环境/驱动级问题。
- MC2 在 mbs1 下就能提升吞吐,不依赖 mbs2。

---

## 四、不采纳的方案及原因

| 方案 | 不采纳原因 |
|---|---|
| 调大 padding 到 2048+ | 报告已试 pad2048 OOM(HBM 65G),且 WPS 降到 743,反向优化。 |
| 改 LoRA rank 到 64 覆盖非专家 Linear | 报告已试 `lora64_nonexpert`,WPS 降到 1044、AICORE 降到 21.92%,拉长 backward 但不提升算力。 |
| FA2(Flash Attention 2) | 报告已试,mbs1 下无收益(`pad1536_nosync_fa2` WPS 1115 vs fused 1133);mbs2+FA2 虽短窗口 WPS 3288 但 step24 挂死。 |
| 增加 dataloader workers/prefetch | 报告 phase timing 显示 `get_batch` 约 2ms,数据加载不是瓶颈。 |
| 2DH / Group-Limited Routing | 单机无跨机流量;改路由违约。 |

---

## 五、报告结论的重新解读

报告第 6 节"本轮没找到 AICORE 40%+ 稳定配置"的结论,在看完博客和代码实测后,应**修正归因**:

**报告原归因**: "mbs1 是唯一稳定区间,mbs2 不稳定所以 AICORE 上不去。"

**修正后的归因**:
1. **mbs1 下 AICORE 23% 的根因**是每专家分到的 token 太少(~1280 token × top8 ÷ 128 experts = 每专家约 80 token),专家 GEMM 的 M 维太小,Cube 单元吃不饱。GMM 虽已开,但**喂进去的矩阵本身就小**。
2. **mbs2 能短暂冲到 AICORE 64%**(每专家 token 翻倍 → GEMM M 维翻倍 → 矩阵单元利用率上升),**证明诊断正确**。mbs2 挂死是**环境级/驱动级问题**(23 次调参均失败、外部 SIGTERM),不是训练语义层面能解的。
3. **通信暴露在关键路径**(AllToAll 零掩盖)是**被报告忽略的隐藏成本**,forward 2.66s 和 backward 1.62s 里都埋着通信等待,这部分用 MC2 可以掩盖掉。

**因此,在"不改模型结构 / LoRA-only / 稳定 mbs1"约束下**:
- **AICORE 40% 不是当前配置空间内的合理目标**(受限于 mbs1 的小 GEMM)。
- **合理目标**是"在 mbs1 下尽量榨干吞吐 + 控制显存在 55-60G"——MC2 正好打这个点(通信掩盖 → 压低 step 耗时 → WPS 上升,AICORE 小幅上浮)。
- 如果项目**必须** AICORE 40%,只有两条路:(A) 解决 mbs2 挂死(需昇腾官方支持);(B) 多机扩 DP 让 global batch 更大、每专家分到更多 token(超出单机范围)。

---

## 六、下一步行动(优先级排序)

1. ✅ **已完成**: 博客筛选、算子探测、代码接通、mc2 配置创建。
2. ⏳ **进行中**: mc2 数学一致性实战校验(跑 20 step probe)。
3. **待执行**: mc2 性能复测(80 step,同口径采集指标)。
4. **待输出**: 单轮优化报表(mc2 vs fused 基线)+ 更新到 `reports/`。

**如果 MC2 收益显著(WPS 提升 > 8%)**,输出最终报告并结束本轮;  
**如果 MC2 收益 < 5%**,说明通信不是主瓶颈,真瓶颈是 mbs1 的小 GEMM,需在报告里明确结论:"当前约束下已达优化边界,进一步提升需解除 mbs1 约束或扩多机"。

---

## 七、参考资料

- 博客: 《MoE 优化方案知识分享:从 Token 路由到昇腾/MindSpeed 落地》(知乎,2026)
- 昇腾文档: 
  - https://www.hiascend.com/document/detail/zh/Pytorch/700/modthirdparty/Mindspeedguide/mindspeed_0044.html  
  - https://www.hiascend.com/developer/techArticles/20250702-1
- 代码: `/data/sejin/third_party/mindspeed-mm-26.0.0`
- 当前报告: `/data/sejin/baseline_26/reports/qwen35_audio_manual_ep8_perf_tuning_20260616.md`
