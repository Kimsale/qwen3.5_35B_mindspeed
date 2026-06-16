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

"""Whisper-large-v3 audio encoder wrapper for the Qwen3.5-Audio multimodal model.

设计要点
--------
1. 直接复用 HuggingFace transformers 的 ``WhisperEncoder``，它本身就是
   openai/whisper-large-v3 的 encoder 部分（2 层 Conv1d + 32 层 Transformer，
   d_model=1280）。这样权重可以零改名地从官方 ``openai/whisper-large-v3``
   safetensors 直接加载，避免自己重写 attention 带来的数学不一致风险。
2. 输入是 ``input_features``（mel 谱，shape ``(B, num_mel_bins, T_mel)``），
   由数据侧的 Whisper ``feature_extractor`` 产出，与 mm_plugin 完全一致。
3. encoder 内部 2 个 Conv1d 做 2x 下采样（stride=2），输出
   ``(B, T_mel/2, 1280)``。后续的 4x 总下采样里，剩余的 2x 由 projector 的
   ``AvgPool1d`` 完成，以严格对齐数据侧 token 数公式（见 projector.py）。
"""

from typing import Optional

import torch
from torch import nn
from transformers.models.whisper.modeling_whisper import WhisperEncoder
from transformers.models.whisper.configuration_whisper import WhisperConfig

from mindspeed.fsdp.utils.log import print_rank
import logging

logger = logging.getLogger(__name__)


class WhisperAudioTower(nn.Module):
    """对 transformers ``WhisperEncoder`` 的薄封装。

    Args:
        config: ``WhisperConfig`` 实例（whisper-large-v3 的标准配置）。
    """

    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.config = config
        # 仅取 encoder，丢弃 whisper 的 decoder（ASR 解码头我们不需要）。
        self.encoder = WhisperEncoder(config)
        # 暴露隐藏维度，供 projector 对齐使用。whisper-large-v3 为 1280。
        self.hidden_size = config.d_model

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def forward(
        self,
        input_features: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        """
        Args:
            input_features: ``(B, num_mel_bins, T_mel)`` 的 mel 谱。
            attention_mask: 透传给 WhisperEncoder（一般可为 None，
                whisper encoder 内部按固定长度处理）。

        Returns:
            ``(B, T_mel/2, d_model)`` 的声学特征。
        """
        input_features = input_features.to(self.dtype)
        encoder_outputs = self.encoder(
            input_features=input_features,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return encoder_outputs.last_hidden_state

    @classmethod
    def from_pretrained_encoder(cls, whisper_path: str, dtype: torch.dtype = torch.float32):
        """从官方 whisper-large-v3 目录加载 encoder 权重。

        只加载 encoder 子模块的权重，decoder 权重会被忽略。
        """
        config = WhisperConfig.from_pretrained(whisper_path)
        module = cls(config)
        # 用 from_pretrained 把完整 whisper 权重载入临时模型，再抽 encoder。
        full = WhisperEncoder.from_pretrained(
            whisper_path,
            config=config,
            dtype=dtype,
        )
        missing, unexpected = module.encoder.load_state_dict(full.state_dict(), strict=False)
        print_rank(
            logger.info,
            f"[WhisperAudioTower] loaded encoder from {whisper_path}; "
            f"missing={len(missing)} unexpected={len(unexpected)}",
        )
        return module
