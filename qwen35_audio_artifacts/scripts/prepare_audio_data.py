#!/usr/bin/env python3
"""
音频 SFT 数据转换脚本
将常见音频数据集格式转换为 Qwen3.5 音频训练所需的 JSONL 格式
"""

import json
import os
from pathlib import Path
from typing import List, Dict

def create_audio_sft_sample(
    sample_id: str,
    audio_path: str,
    transcription: str,
    instruction: str = "请转写这段语音。"
) -> Dict:
    """
    创建单个音频 SFT 样本

    Args:
        sample_id: 样本唯一标识
        audio_path: 音频文件路径（相对或绝对）
        transcription: 音频内容的文字转写
        instruction: 用户指令（可自定义）

    Returns:
        符合 Qwen3.5 格式的样本字典
    """
    return {
        "id": sample_id,
        "audios": [audio_path],
        "messages": [
            {
                "role": "user",
                "content": f"<|AUDIO|>\n{instruction}"
            },
            {
                "role": "assistant",
                "content": transcription
            }
        ]
    }


def convert_csv_to_jsonl(
    csv_path: str,
    output_path: str,
    audio_base_dir: str = None,
    instruction: str = "请转写这段语音。"
):
    """
    从 CSV 格式转换（格式：audio_path,transcription）

    Args:
        csv_path: 输入 CSV 文件路径
        output_path: 输出 JSONL 文件路径
        audio_base_dir: 音频文件根目录（如果 CSV 中是相对路径）
        instruction: 默认指令
    """
    import csv

    with open(csv_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:

        reader = csv.reader(f_in)
        next(reader, None)  # 跳过表头（如果有）

        for i, row in enumerate(reader):
            if len(row) < 2:
                continue

            audio_path, transcription = row[0], row[1]

            # 如果指定了 audio_base_dir，转为绝对路径
            if audio_base_dir:
                audio_path = str(Path(audio_base_dir) / audio_path)

            # 检查音频文件是否存在
            if not os.path.exists(audio_path):
                print(f"⚠️  警告: 音频文件不存在，跳过: {audio_path}")
                continue

            sample = create_audio_sft_sample(
                sample_id=f"asr_{i:06d}",
                audio_path=audio_path,
                transcription=transcription,
                instruction=instruction
            )

            f_out.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"✅ 转换完成: {output_path}")


def convert_directory_to_jsonl(
    audio_dir: str,
    transcription_file: str,
    output_path: str,
    instruction: str = "请转写这段语音。"
):
    """
    从目录格式转换（音频文件在一个目录，转写文本在单独文件）

    Args:
        audio_dir: 音频文件目录
        transcription_file: 转写文本文件（格式：audio_filename TAB transcription）
        output_path: 输出 JSONL 文件路径
        instruction: 默认指令
    """
    # 读取转写文本
    transcriptions = {}
    with open(transcription_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                filename, text = parts[0], parts[1]
                transcriptions[filename] = text

    # 生成 JSONL
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for i, filename in enumerate(sorted(transcriptions.keys())):
            audio_path = str(Path(audio_dir) / filename)

            if not os.path.exists(audio_path):
                print(f"⚠️  警告: 音频文件不存在，跳过: {audio_path}")
                continue

            sample = create_audio_sft_sample(
                sample_id=f"asr_{i:06d}",
                audio_path=audio_path,
                transcription=transcriptions[filename],
                instruction=instruction
            )

            f_out.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"✅ 转换完成: {output_path}")


def create_demo_data(output_dir: str, num_samples: int = 10):
    """
    创建演示数据（用于快速测试，需要手动提供真实音频文件）

    Args:
        output_dir: 输出目录
        num_samples: 生成样本数量
    """
    os.makedirs(output_dir, exist_ok=True)

    demo_texts = [
        "今天天气很好，我们一起去公园散步吧。",
        "会议定于下周三上午十点在三号会议室召开。",
        "请帮我预订明天晚上七点的餐厅位置。",
        "这个项目的进度符合预期，下周可以进入测试阶段。",
        "早上好，今天的日程安排已经发送到您的邮箱。",
        "感谢您的来电，我们会尽快处理您的问题。",
        "这本书非常有趣，我强烈推荐你读一读。",
        "明天的会议改到下午三点，请注意时间变更。",
        "这道菜味道很好，你要不要尝一尝？",
        "周末我们去爬山怎么样？听说那里的风景很美。"
    ]

    output_path = str(Path(output_dir) / "train.jsonl")

    with open(output_path, 'w', encoding='utf-8') as f:
        for i in range(min(num_samples, len(demo_texts))):
            sample = create_audio_sft_sample(
                sample_id=f"demo_{i:04d}",
                audio_path=f"{output_dir}/audio/sample_{i:04d}.wav",
                transcription=demo_texts[i],
                instruction="请转写这段语音。"
            )
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"✅ 演示数据已生成: {output_path}")
    print(f"⚠️  注意: 需要手动准备音频文件到 {output_dir}/audio/ 目录")
    print(f"   或将 JSONL 中的路径改为你的真实音频文件路径")


def validate_jsonl(jsonl_path: str) -> bool:
    """
    验证 JSONL 文件格式和音频文件是否存在

    Args:
        jsonl_path: JSONL 文件路径

    Returns:
        验证是否通过
    """
    print(f"验证文件: {jsonl_path}")

    valid_count = 0
    missing_audio = []

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            try:
                sample = json.loads(line)

                # 检查必需字段
                assert 'id' in sample, f"行 {i}: 缺少 'id' 字段"
                assert 'audios' in sample, f"行 {i}: 缺少 'audios' 字段"
                assert 'messages' in sample, f"行 {i}: 缺少 'messages' 字段"
                assert len(sample['audios']) > 0, f"行 {i}: 'audios' 不能为空"
                assert len(sample['messages']) >= 2, f"行 {i}: 'messages' 至少需要 2 条（user + assistant）"

                # 检查音频文件
                for audio_path in sample['audios']:
                    if not os.path.exists(audio_path):
                        missing_audio.append((i, audio_path))

                valid_count += 1

            except (json.JSONDecodeError, AssertionError) as e:
                print(f"❌ 行 {i}: {e}")
                return False

    print(f"✅ 格式验证通过: {valid_count} 个样本")

    if missing_audio:
        print(f"⚠️  警告: {len(missing_audio)} 个音频文件不存在:")
        for line_num, path in missing_audio[:10]:  # 只显示前10个
            print(f"   行 {line_num}: {path}")
        if len(missing_audio) > 10:
            print(f"   ... 还有 {len(missing_audio) - 10} 个")
        return False

    print("✅ 所有音频文件都存在")
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="音频 SFT 数据转换工具")
    parser.add_argument("--mode", choices=["csv", "dir", "demo", "validate"], required=True,
                        help="转换模式: csv=从CSV转换, dir=从目录转换, demo=生成演示数据, validate=验证JSONL")
    parser.add_argument("--input", help="输入文件/目录路径")
    parser.add_argument("--output", help="输出JSONL文件路径")
    parser.add_argument("--audio-dir", help="音频文件目录（csv/dir模式）")
    parser.add_argument("--transcription-file", help="转写文本文件（dir模式）")
    parser.add_argument("--num-samples", type=int, default=10, help="演示数据样本数（demo模式）")

    args = parser.parse_args()

    if args.mode == "csv":
        if not args.input or not args.output:
            parser.error("csv 模式需要 --input 和 --output")
        convert_csv_to_jsonl(args.input, args.output, args.audio_dir)

    elif args.mode == "dir":
        if not args.audio_dir or not args.transcription_file or not args.output:
            parser.error("dir 模式需要 --audio-dir, --transcription-file 和 --output")
        convert_directory_to_jsonl(args.audio_dir, args.transcription_file, args.output)

    elif args.mode == "demo":
        if not args.output:
            parser.error("demo 模式需要 --output（输出目录）")
        create_demo_data(args.output, args.num_samples)

    elif args.mode == "validate":
        if not args.input:
            parser.error("validate 模式需要 --input")
        validate_jsonl(args.input)
