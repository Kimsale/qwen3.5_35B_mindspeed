# Copyright 2025 Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""把 Qwen3.5-MoE 基座 + Whisper-large-v3 encoder + 初始化的 projector
合并成一个 DCP 权重，供 FSDP2 栈的 meta-device 初始化 + DCP 加载使用。

为什么需要这一步
----------------
FSDP2 训练入口用 ``init_model_with_meta_device=True`` 构建模型（空壳），随后用
``training.load`` 指定的 DCP 权重去填充。DCP 加载是 ``allow_partial_load`` 模式：
checkpoint 里缺失的参数 **不会报错**，但会保持 ``to_empty`` 后的未初始化显存
（垃圾值）。因此若不把 audio_tower / audio_projector 的权重写进 DCP，Whisper
就会是垃圾权重 —— 训练在数学上无意义。

本脚本产出的 DCP 同时包含三部分，key 前缀与模型 ``named_parameters`` 完全对应：
  * ``model.*``            —— Qwen3.5-MoE 基座（含 MoE expert 的 reshape/permute）
  * ``audio_tower.encoder.*`` —— Whisper-large-v3 的 encoder 权重
  * ``audio_projector.*``  —— 新增 projector，按指定 init 方式初始化（默认 kaiming）

用法
----
    bash 进入 CANN8.5 环境后执行：
    python -m mindspeed_mm.fsdp.models.qwen3_5_audio.convert_weights \
        --qwen_hf_dir   /mnt/shared_data_196/sejin/models/Qwen3.5-35B-A3B \
        --whisper_hf_dir /mnt/shared_data_196/sejin/models/whisper-large-v3 \
        --dcp_dir       /mnt/shared_data_196/sejin/models/Qwen3.5-35B-A3B-audio-dcp \
        --llm_hidden_size 2048
"""

import argparse
import re
from pathlib import Path

import torch
from torch import nn
from safetensors.torch import load_file
from transformers.models.whisper.configuration_whisper import WhisperConfig

from checkpoint.common.hf_to_dcp import hf_to_dcp_sharded
from .projector import AudioProjector


# 与 Qwen35Converter 一致：MoE expert 权重需要 reshape + permute 以适配 gemm。
EXPERT_PATTERNS = [
    r"model\.language_model\.layers\.\d+\.mlp\.experts\.gate_up_proj",
    r"model\.language_model\.layers\.\d+\.mlp\.experts\.down_proj",
]
TIE_MAPPING = {"lm_head.weight": "model.language_model.embed_tokens.weight"}


def _convert_qwen_shard(state_dict):
    """对单个 Qwen safetensors 分片做 tie / reshape / permute（沿用官方 Qwen35Converter 逻辑）。"""
    for tgt, src in TIE_MAPPING.items():
        if src in state_dict:
            state_dict[tgt] = state_dict[src]

    for key in list(state_dict.keys()):
        value = state_dict.pop(key)
        for pattern in EXPERT_PATTERNS:
            if re.fullmatch(pattern, key):
                # (E, 2I, H) -> (E, H, 2I)
                value = value.permute(0, 2, 1).contiguous()
        state_dict[key] = value
    return state_dict


def build_audio_state_dict(whisper_hf_dir: str, llm_hidden_size: int, init_lora_weights: str = "kaiming"):
    """构建 audio_tower.encoder.* 和 audio_projector.* 的 state_dict。"""
    whisper_config = WhisperConfig.from_pretrained(whisper_hf_dir)

    # ---- whisper encoder 权重 ----
    enc_sd = {}
    for f in sorted(Path(whisper_hf_dir).glob("*.safetensors")):
        sd = load_file(str(f), device="cpu")
        for k, v in sd.items():
            # whisper 完整权重里 encoder 子模块前缀为 "model.encoder." 或 "encoder."
            if k.startswith("model.encoder."):
                enc_sd["audio_tower.encoder." + k[len("model.encoder."):]] = v
            elif k.startswith("encoder."):
                enc_sd["audio_tower.encoder." + k[len("encoder."):]] = v
    if not enc_sd:
        raise RuntimeError(f"未在 {whisper_hf_dir} 找到 encoder 权重，请确认 whisper-large-v3 safetensors 完整。")

    # ---- projector 权重（新增模块，初始化后写入）----
    projector = AudioProjector(
        audio_hidden_size=whisper_config.d_model,
        llm_hidden_size=llm_hidden_size,
    )
    # LayerNorm/Linear 默认初始化即可；保持 fp32。
    proj_sd = {f"audio_projector.{k}": v for k, v in projector.state_dict().items()}

    merged = {}
    merged.update(enc_sd)
    merged.update(proj_sd)
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen_hf_dir", required=True)
    parser.add_argument("--whisper_hf_dir", required=True)
    parser.add_argument("--dcp_dir", required=True)
    parser.add_argument("--llm_hidden_size", type=int, default=2048)
    args = parser.parse_args()

    audio_sd = build_audio_state_dict(args.whisper_hf_dir, args.llm_hidden_size)
    print(f"[convert] audio 权重张量数: {len(audio_sd)}")

    # 标记：第一个分片附加 audio 权重；其余分片只过 qwen 转换。
    state = {"_audio_injected": False}

    def convert_func(shard_sd):
        shard_sd = _convert_qwen_shard(shard_sd)
        if not state["_audio_injected"]:
            shard_sd.update(audio_sd)
            state["_audio_injected"] = True
        return shard_sd

    hf_to_dcp_sharded(
        hf_dir=args.qwen_hf_dir,
        dcp_dir=args.dcp_dir,
        state_dict_convert_func=convert_func,
    )
    if not state["_audio_injected"]:
        raise RuntimeError("audio 权重未被注入，qwen_hf_dir 下没有 safetensors 分片？")
    print(f"[convert] 完成，DCP 已写入 {args.dcp_dir}")


if __name__ == "__main__":
    main()
