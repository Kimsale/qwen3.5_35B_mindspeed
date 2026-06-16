#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按团队 7 数据集语音长度分布，构造同分布的音频 SFT 训练数据。

设计依据（团队提供的全局分布）：
  总样本 667,603；p5=0.5s p25=2.2s p50=5.0s mean=6.0s p75=9.9s p90=11.3s p95=14.2s max=226.7s
  按比例混合 7 个子集（短音频<3s 约占 40%，来自两个 AED 集）。

关键约束（已对照 mindspeed-mm 源码确认）：
  - Whisper feature_extractor 用 padding="max_length"，音频被截/补到 30s 上限(3000 mel帧)。
    => 时长上限设 30s（团队 max 226s 的极长尾本就会被 Whisper 截断到 30s）。
  - 音频 token 数 ≈ 时长(s) × 25，cutoff_len=4096 余量充足。
  - librosa.load(sr=16000) 自动重采样 => 直接生成 16kHz mono wav。

数据性质：真实时长分布(算力/ token 数真实) + 合成波形与合成转写文本。
  本任务目标是验证音频训练流程并采集性能，不是训练高质量 ASR，故内容用合成。
"""
import argparse
import json
import os
import random
import wave

import numpy as np

# 7 子集：比例、时长采样参数（对数正态贴合各自中位/范围；截到 30s 上限）
# (name, proportion, sampler)
SR = 16000
MAX_SEC = 30.0  # Whisper 上限
MIN_SEC = 0.2


def lognorm(median, sigma, lo, hi):
    """对数正态：median 为中位数，sigma 控制离散度，截断到 [lo, hi]。"""
    def _s():
        v = median * np.exp(np.random.randn() * sigma)
        return float(np.clip(v, lo, hi))
    return _s


def fixed(val):
    return lambda: float(val)


SUBSETS = [
    # name,            prop,  sampler
    # 修正点：两个 AED 子集按团队口径应贡献约 40% <3s 样本，旧参数虽有 40%
    # AED 配比，但 AED_event_2 长尾过多，导致全局 <3s 只有约 30.7%。
    # 因此将 AED 子集截断到 2.95s，并上调中等长度子集，保持 p50/mean 接近目标。
    ("AED_event_2",    0.32, lognorm(1.9, 0.65, 0.2, 2.95)),  # 短事件，主力，大量<3s
    ("mulv18",         0.20, lognorm(8.9, 0.45, 5.0, 22.0)),  # 多语言长段
    ("aishell1",       0.18, lognorm(6.0, 0.30, 3.0, 8.0)),   # 标准 ASR
    ("CochlScene",     0.09, fixed(10.0)),                    # 固定 10s
    ("pretrain_cap",   0.08, lognorm(11.3, 0.40, 5.0, 22.0)), # 中长描述
    ("AED_event_0",    0.08, lognorm(0.8, 0.65, 0.2, 2.95)),  # 极短，大量<1s
    ("ChildMandarin",  0.05, lognorm(3.8, 0.45, 1.0, 6.0)),   # 儿童语音
]

# 多样化的 ASR/语音理解指令
INSTRUCTIONS = [
    "<|AUDIO|>\n请把这段语音转写成文字。",
    "<|AUDIO|>\n这段音频说了什么？",
    "<|AUDIO|>\n请转写以下语音内容。",
    "<|AUDIO|>\n听完这段语音，请总结主要内容。",
    "<|AUDIO|>\n请识别这段音频中的语音。",
    "<|AUDIO|>\n把录音内容写出来。",
]

# 合成转写文本片段（中文，长度随机拼接以贴近真实回复）
TEXT_POOL = [
    "今天天气很好，我们一起去公园散步吧。",
    "会议定于下周三上午十点在三号会议室召开。",
    "请帮我查询最近的航班信息。",
    "这段录音里说话人讨论了项目的进度安排。",
    "麻烦把空调温度调低两度，谢谢。",
    "他说明天的活动改到下午三点开始。",
    "系统检测到一处异常，已自动记录日志。",
    "孩子们在操场上开心地玩耍。",
    "这首歌的旋律非常优美动听。",
    "客户希望尽快得到退款处理结果。",
]


def synth_waveform(duration_sec: float) -> np.ndarray:
    """合成类语音包络的波形：几个谐波正弦 + 轻噪声 + 幅度包络。"""
    n = max(1, int(duration_sec * SR))
    t = np.arange(n) / SR
    # 基频在人声范围随机
    f0 = random.uniform(110, 260)
    sig = np.zeros(n, dtype=np.float64)
    for k, amp in enumerate([1.0, 0.5, 0.3, 0.15], start=1):
        sig += amp * np.sin(2 * np.pi * f0 * k * t + random.random() * 6.28)
    # 音节包络：几个高斯包，模拟语音断续
    env = np.zeros(n)
    n_syl = max(1, int(duration_sec * random.uniform(2.5, 4.0)))
    for _ in range(n_syl):
        c = random.uniform(0, duration_sec)
        w = random.uniform(0.05, 0.18)
        env += np.exp(-0.5 * ((t - c) / w) ** 2)
    env = env / (env.max() + 1e-8)
    sig = sig * env + 0.02 * np.random.randn(n)
    sig = sig / (np.abs(sig).max() + 1e-8) * 0.9
    return (sig * 32767).astype(np.int16)


def write_wav(path: str, pcm16: np.ndarray):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm16.tobytes())


def sample_duration():
    r = random.random()
    acc = 0.0
    for name, prop, sampler in SUBSETS:
        acc += prop
        if r <= acc:
            d = sampler()
            return name, float(np.clip(d, MIN_SEC, MAX_SEC))
    name, _, sampler = SUBSETS[-1]
    return name, float(np.clip(sampler(), MIN_SEC, MAX_SEC))


def make_transcript():
    k = random.randint(1, 3)
    return "".join(random.choice(TEXT_POOL) for _ in range(k))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3200, help="样本数（100步×global_batch32=3200）")
    ap.add_argument("--out", default="/data/sejin/baseline_26/data_audio")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--text-only-ratio", type=float, default=0.05,
                    help="纯文本样本比例（语音/文本混合训练，贴近团队含 text_only 样本）")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    audio_dir = os.path.join(args.out, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    jsonl_path = os.path.join(args.out, "train.jsonl")

    durs = []
    subset_count = {}
    n_text_only = 0
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for i in range(args.n):
            sid = f"sample_{i:06d}"
            if random.random() < args.text_only_ratio:
                n_text_only += 1
                rec = {
                    "id": sid,
                    "messages": [
                        {"role": "user", "content": "用一句话解释什么是语音识别。"},
                        {"role": "assistant", "content": "语音识别是把人说的话自动转换成对应文字的技术。"},
                    ],
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            name, dur = sample_duration()
            durs.append(dur)
            subset_count[name] = subset_count.get(name, 0) + 1
            wav_path = os.path.join(audio_dir, f"{sid}.wav")
            write_wav(wav_path, synth_waveform(dur))
            rec = {
                "id": sid,
                "audios": [wav_path],
                "messages": [
                    {"role": "user", "content": random.choice(INSTRUCTIONS)},
                    {"role": "assistant", "content": make_transcript()},
                ],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (i + 1) % 500 == 0:
                print(f"  ...{i+1}/{args.n}")

    d = np.array(durs)
    print("\n=== 生成完成 ===")
    print(f"JSONL: {jsonl_path}")
    print(f"总样本: {args.n}  (含音频 {len(durs)}, 纯文本 {n_text_only})")
    print(f"音频目录: {audio_dir}")
    print("\n=== 时长分布(秒) vs 团队目标 ===")
    print(f"  p5  ={np.percentile(d,5):.2f} (目标0.5)")
    print(f"  p25 ={np.percentile(d,25):.2f} (目标2.2)")
    print(f"  p50 ={np.percentile(d,50):.2f} (目标5.0)")
    print(f"  mean={d.mean():.2f} (目标6.0)")
    print(f"  p75 ={np.percentile(d,75):.2f} (目标9.9)")
    print(f"  p90 ={np.percentile(d,90):.2f} (目标11.3)")
    print(f"  p95 ={np.percentile(d,95):.2f} (目标14.2)")
    print(f"  max ={d.max():.2f} (目标226.7,Whisper截至30)")
    print(f"  <3s 占比={np.mean(d<3)*100:.1f}% (目标~40%)")
    print("\n=== 子集分布 ===")
    for name, prop, _ in SUBSETS:
        c = subset_count.get(name, 0)
        print(f"  {name:14s} {c:5d} ({c/max(1,len(durs))*100:.1f}%, 目标{prop*100:.0f}%)")
    total_sec = d.sum()
    print(f"\n总音频时长: {total_sec/3600:.2f} 小时, 预估磁盘 ~{total_sec*SR*2/1e9:.2f} GB")


if __name__ == "__main__":
    main()
