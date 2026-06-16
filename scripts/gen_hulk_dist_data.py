#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按 Hulk 数据集 (jilian_fast_common_maxtok_2k.mdb) 的样本长度分布,
生成同分布的中文 SFT 训练数据 (messages/sharegpt 格式 jsonl).

Hulk 分布 (382746 条, tokens):
  min=21 max=2048 mean=614.20 median=466 std=460.32
  P10=158 P25=277 P50=466 P75=837 P90=1347 P95=1622 P99=1945
  区间占比:
    0-100:    4.01%
    100-200:  11.19%
    200-500:  38.34%
    500-1000: 27.61%
    1000-1500:11.83%
    1500-2000: 6.57%
    2000-2048: 0.45%

做法:
  1. 按上述"区间占比"对每条样本采样一个目标 token 长度 (区间内按经验分布细化,贴近给定分位数);
  2. 中文 ~0.68 token/字, ChatML 3轮模板固定开销 ~22 token;
  3. 用真实语义的中文训练知识语料拼接到目标字符数, 构造 system/user/assistant 三轮对话;
  4. 用真实 tokenizer 校验每条长度, 迭代微调字符数命中目标 token (±2%);
  5. 输出 messages 格式 jsonl, 后续走 preprocess_data.py (SharegptStyleInstructionHandler) 转 mcore packed.
"""
import argparse
import json
import random
from pathlib import Path

# 真实语义的中文语料池 (大模型训练/性能优化主题, 保证内容有意义而非乱码)
CORPUS = [
    "张量并行把同一层的权重矩阵沿行或列切分到多张设备上,每张卡只计算一部分矩阵乘法,再通过all-reduce或all-gather把结果拼接回来。",
    "流水线并行将模型按层切成多个阶段,不同阶段放在不同设备上,通过micro-batch在阶段间流水执行来提高设备利用率,但需要处理气泡开销。",
    "数据并行在每张卡上保留完整模型副本,各自处理不同的数据分片,反向传播后通过all-reduce同步梯度,是最简单也最常用的并行方式。",
    "专家并行是混合专家模型特有的并行策略,把不同的专家网络分布到不同设备,通过all-to-all通信把token路由到对应专家所在的卡上计算。",
    "序列并行沿序列维度切分激活值和部分计算,通常与张量并行配合使用,能进一步降低长序列训练时的显存峰值,但会引入额外的通信开销。",
    "上下文并行(如Ulysses)把超长序列切分到多张卡,通过对注意力头维度做all-to-all通信来完成全局注意力计算,是处理超长上下文的关键技术。",
    "梯度累积允许在显存受限时维持较大的等效全局批量,做法是连续执行多次前向反向并累加梯度,直到累积步数达到目标后再统一更新一次参数。",
    "重计算(激活检查点)在前向时不保存中间激活,反向时重新计算它们,用额外的计算换取显存空间,full粒度重计算能最大程度节省显存。",
    "混合精度训练用bf16或fp16执行大部分计算,显著降低显存和带宽压力。bf16因为动态范围更大,通常比fp16更稳定,不需要损失缩放。",
    "ZeRO优化器把优化器状态、梯度和参数分片到多张卡上,Stage-1分片优化器状态,Stage-2再分片梯度,Stage-3连参数也分片,逐级降低单卡显存。",
    "LoRA通过在原权重旁注入低秩矩阵A和B来微调模型,只训练这两个小矩阵而冻结主体权重,大幅减少可训练参数量和显存占用,适合快速适配。",
    "Flash Attention通过分块计算和在线softmax避免显式构造完整的注意力矩阵,把显存复杂度从平方降到线性,同时利用片上内存提升计算效率。",
    "算子融合把多个连续的小算子合并成一个大算子,减少kernel启动开销和中间结果的显存读写,常见的有融合的RMSNorm、SwiGLU和旋转位置编码。",
    "训练吞吐通常用每秒处理的token数或样本数衡量,结合单步耗时和全局批量可以推算。吞吐越高代表硬件利用越充分,但要和loss收敛一起看。",
    "AI Core利用率反映计算单元的繁忙程度,对MoE这类稀疏激活的模型,瞬时采样往往失真,更可靠的指标是从profiler得到的实际计算时间占比。",
    "通信掩盖技术通过让通信和计算重叠执行来隐藏通信延迟,比如梯度reduce-scatter与反向计算重叠,参数all-gather与前向计算重叠。",
    "HBM带宽是大模型训练的常见瓶颈之一,当计算访存比偏低时,设备大量时间花在等待数据从显存搬运,此时提高计算密度或融合算子能带来收益。",
    "学习率调度对训练稳定性影响很大,常用cosine退火配合warmup,先线性升温避免初期梯度震荡,再缓慢衰减帮助模型收敛到更好的极小值。",
    "梯度裁剪通过限制梯度范数的上界来防止梯度爆炸,在长序列或大学习率训练中尤为重要,裁剪阈值过小会拖慢收敛,过大则失去保护作用。",
    "MoE模型的负载均衡通过辅助损失鼓励路由器把token均匀分配给各个专家,避免少数专家过载而其他专家闲置,从而提升整体的计算效率。",
    "checkpoint保存策略需要在容错和开销之间权衡,保存太频繁会拖慢训练并占用大量磁盘,太稀疏则故障恢复时损失的进度过多,通常按固定步数保存。",
    "数据打包(pack)把多个短样本拼接到一个固定长度的序列里,配合注意力掩码隔离不同样本,能显著减少padding浪费,提升有效token的训练比例。",
    "swap优化器把优化器状态卸载到CPU内存,在需要时再搬回设备,用主机内存和PCIe带宽换取设备显存,代价是每步引入额外的数据搬运延迟。",
    "分布式训练的通信后端在昇腾上由HCCL承担,负责all-reduce、all-gather、all-to-all等集合通信操作,其性能直接影响多卡扩展的效率。",
]

SYSTEM_PROMPTS = [
    "你是一个帮助训练工程师理解大模型训练与性能优化的中文助手。",
    "你是一位精通昇腾NPU分布式训练的技术专家,请用清晰准确的中文回答。",
    "你是大模型训练框架的资深顾问,擅长解释并行策略、显存优化和性能调优。",
]

QUESTION_TEMPLATES = [
    "请详细解释{topic}的原理,并说明它在实际训练中的作用。",
    "在大模型训练中,{topic}是如何工作的?请举例说明。",
    "请深入分析{topic}的技术细节、适用场景以及可能的开销。",
    "能否系统地介绍一下{topic},包括它解决了什么问题?",
    "请从工程实践角度阐述{topic}的实现方式和注意事项。",
]
TOPICS = [
    "张量并行", "流水线并行", "数据并行", "专家并行", "序列并行", "上下文并行",
    "梯度累积", "激活重计算", "混合精度训练", "ZeRO优化器分片", "LoRA微调",
    "Flash Attention", "算子融合", "训练吞吐评估", "通信掩盖", "MoE负载均衡",
    "数据打包", "swap优化器", "HBM带宽瓶颈", "学习率调度", "梯度裁剪",
]

# Hulk 长度区间 + 占比 (token). 区间内用 (low, high) 均匀采样目标 token 长度.
# 末区间 2000-2048 极窄, 合并入 1500-2048 时单列保留占比.
HULK_BUCKETS = [
    (21,   100,  0.0401),
    (100,  200,  0.1119),
    (200,  500,  0.3834),
    (500,  1000, 0.2761),
    (1000, 1500, 0.1183),
    (1500, 2000, 0.0657),
    (2000, 2048, 0.0045),
]

TOK_PER_CHAR = 0.68     # 实测中文 token/字 比例
CHATML_OVERHEAD = 22    # ChatML 3 轮模板固定 token 开销 (system/user/assistant 标记)


def sample_target_tokens(rng):
    """按 Hulk 区间占比采样一个目标 token 长度。硬上限 2048 (Hulk maxtok_2k)。"""
    r = rng.random()
    cum = 0.0
    for low, high, p in HULK_BUCKETS:
        cum += p
        if r <= cum:
            return min(2048, rng.randint(low, high))
    return rng.randint(2000, 2048)


def build_text_to_chars(rng, n_chars):
    """从语料池拼接出约 n_chars 字的中文文本 (有真实语义)。"""
    if n_chars <= 0:
        return CORPUS[rng.randrange(len(CORPUS))][:max(1, n_chars)]
    parts = []
    total = 0
    pool = CORPUS[:]
    rng.shuffle(pool)
    i = 0
    while total < n_chars:
        seg = pool[i % len(pool)]
        parts.append(seg)
        total += len(seg)
        i += 1
    text = "".join(parts)
    return text[:n_chars]


def build_sample(rng, target_tokens, tokenizer):
    """构造一条 messages 样本, 迭代微调 assistant 长度使总 token 命中 target (±2%)。"""
    system = SYSTEM_PROMPTS[rng.randrange(len(SYSTEM_PROMPTS))]
    topic = TOPICS[rng.randrange(len(TOPICS))]
    question = QUESTION_TEMPLATES[rng.randrange(len(QUESTION_TEMPLATES))].format(topic=topic)

    # 预算: 目标 token - 模板开销 - system - question, 余下给 answer
    sys_q_tok = CHATML_OVERHEAD + int(len(system) * TOK_PER_CHAR) + int(len(question) * TOK_PER_CHAR)
    ans_tok_budget = max(4, target_tokens - sys_q_tok)
    ans_chars = int(ans_tok_budget / TOK_PER_CHAR)

    # 迭代 3 次微调命中 target
    answer = build_text_to_chars(rng, ans_chars)
    for _ in range(4):
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        ids = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
        cur = len(ids)
        if abs(cur - target_tokens) <= max(2, int(0.02 * target_tokens)):
            break
        # 按差值比例调整字符数
        diff_tok = target_tokens - cur
        ans_chars = max(1, ans_chars + int(diff_tok / TOK_PER_CHAR))
        answer = build_text_to_chars(rng, ans_chars)

    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }, cur


def main():
    ap = argparse.ArgumentParser(description="Generate Hulk-distribution Chinese SFT data (messages jsonl).")
    ap.add_argument("--output", required=True, help="输出 jsonl 路径")
    ap.add_argument("--num-samples", type=int, default=4000, help="生成样本数")
    ap.add_argument("--tokenizer", default="/data/sejin/models/Qwen3-30B-A3B-Base")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stats", action="store_true", help="生成后打印长度分布统计")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    import numpy as np

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    rng = random.Random(args.seed)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lens = []
    with out_path.open("w", encoding="utf-8") as f:
        for i in range(args.num_samples):
            target = sample_target_tokens(rng)
            sample, actual = build_sample(rng, target, tok)
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            lens.append(actual)
            if (i + 1) % 500 == 0:
                print(f"  生成 {i+1}/{args.num_samples} ...", flush=True)

    print(f"已写出 {len(lens)} 条 -> {out_path}")
    if args.stats:
        a = np.array(lens)
        print(f"实际分布: min={a.min()} max={a.max()} mean={a.mean():.1f} median={np.median(a):.0f} std={a.std():.1f}")
        for p in [10, 25, 50, 75, 90, 95, 99]:
            print(f"  P{p}: {np.percentile(a, p):.0f}")
        edges = [0, 100, 200, 500, 1000, 1500, 2000, 2049]
        print("区间占比:")
        for lo, hi in zip(edges[:-1], edges[1:]):
            n = int(((a >= lo) & (a < hi)).sum())
            print(f"  [{lo}-{hi}): {n} ({100*n/len(a):.2f}%)")


if __name__ == "__main__":
    main()
