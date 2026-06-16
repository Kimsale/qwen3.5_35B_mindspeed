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

"""Audio adapter / projector：把 Whisper encoder 的声学特征对齐到 LLM 嵌入空间。

下采样数学一致性（关键！）
--------------------------
数据侧 ``mm_plugin.py`` 计算每条音频展开成多少个 ``<|audio_pad|>`` token：

    input_lengths = (mel_valid_frames - 1) // 2 + 1     # Whisper 2 层 Conv1d(stride=2) 的输出长度
    audio_lengths = (input_lengths      - 2) // 2 + 1     # 本 projector 的 AvgPool1d(k=2,s=2) 输出长度

模型这一侧产出的音频向量数必须 **逐条等于** ``audio_lengths``，否则顶层
``masked_scatter`` 会因元素数不匹配而报错。因此：

  * Whisper encoder（whisper_encoder.py）负责第一个 2x（卷积）；
  * 本 projector 的 ``AvgPool1d(kernel_size=2, stride=2)`` 负责第二个 2x；

两者相乘即数据侧约定的 ~4x 总下采样，数量严格自洽。
``AvgPool1d`` 无参数，纯线性平均，改动前后在数学上等价于对相邻两帧取均值，
不破坏声学语义。
"""

import torch
from torch import nn


def get_feat_lengths_after_conv(mel_valid_frames: torch.Tensor) -> torch.Tensor:
    """Whisper 2 层 stride=2 卷积后的有效帧数：``(L - 1) // 2 + 1``。"""
    return (mel_valid_frames - 1) // 2 + 1


def get_audio_lengths_after_pool(conv_lengths: torch.Tensor) -> torch.Tensor:
    """AvgPool1d(k=2,s=2) 后的有效帧数：``(L - 2) // 2 + 1``。"""
    return (conv_lengths - 2) // 2 + 1


class AudioProjector(nn.Module):
    """Whisper(1280) → AvgPool 2x → LayerNorm → MLP → LLM hidden(2048)。

    Args:
        audio_hidden_size: Whisper encoder 输出维度（whisper-large-v3 = 1280）。
        llm_hidden_size:   LLM 文本塔嵌入维度（Qwen3.5-35B-A3B = 2048）。
        projector_hidden_act: 中间激活函数名（默认 gelu）。
    """

    def __init__(
        self,
        audio_hidden_size: int = 1280,
        llm_hidden_size: int = 2048,
        projector_hidden_act: str = "gelu",
    ):
        super().__init__()
        self.audio_hidden_size = audio_hidden_size
        self.llm_hidden_size = llm_hidden_size

        # 第二个 2x 下采样，与数据侧 token 数公式严格对齐。
        self.avg_pooler = nn.AvgPool1d(kernel_size=2, stride=2)
        self.ln_post = nn.LayerNorm(audio_hidden_size)
        # 两层 MLP 做维度对齐（SLAM-LLM 风格的轻量 projector）。
        self.linear_1 = nn.Linear(audio_hidden_size, llm_hidden_size, bias=True)
        self.act = nn.GELU() if projector_hidden_act == "gelu" else nn.SiLU()
        self.linear_2 = nn.Linear(llm_hidden_size, llm_hidden_size, bias=True)

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio_features: ``(num_frames_conv, audio_hidden_size)`` —— 单条音频
                经 Whisper encoder 后、且已按有效长度裁剪好的特征。

        Returns:
            ``(num_frames_pool, llm_hidden_size)``，其中
            ``num_frames_pool = (num_frames_conv - 2) // 2 + 1``。
        """
        # AvgPool1d 作用在时间维：(T, C) -> (C, T) -> pool -> (T', C)
        x = audio_features.transpose(0, 1).unsqueeze(0)   # (1, C, T)
        x = self.avg_pooler(x)                             # (1, C, T')
        x = x.squeeze(0).transpose(0, 1)                   # (T', C)

        x = self.ln_post(x)
        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        return x
