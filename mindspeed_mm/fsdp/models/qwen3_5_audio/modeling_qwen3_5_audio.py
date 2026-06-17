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

"""Qwen3.5-35B-A3B + Whisper-large-v3 语音多模态模型（FSDP2 栈）。

整体思路
--------
复用框架已注册的 ``Qwen3_5MoeForConditionalGeneration``（MoE 文本塔 + 视觉塔），
通过 **子类化** 的方式挂上 Whisper audio encoder 与 projector，并重写 ``forward``：

    input_features ─► WhisperAudioTower(冻结) ─► AudioProjector(可训) ─► audio_embeds
                                                                            │ masked_scatter
    input_ids ─► embed_tokens ─► text_embeds ───────────────────────────────┘
                                       │
                                 融合后的 inputs_embeds ─► super().forward(...)

为什么委托给 ``super().forward``：父类已经实现了 chunk-loss、MoE aux-loss、CP-loss
聚合等逻辑，我们只在它之前把音频特征融进 ``inputs_embeds``，避免重写损失链路引入
数学不一致。视觉塔保留在父类里但不喂 ``pixel_values`` 即等于关闭，符合“丢弃 vision”
的需求；如需彻底省显存可在 YAML 中 freeze ``model.visual``。

数量一致性
----------
音频 token 数严格遵循数据侧 ``mm_plugin.py`` 的公式：
    conv_len  = (mel_valid_frames - 1) // 2 + 1     # Whisper 卷积 2x
    audio_len = (conv_len          - 2) // 2 + 1     # projector AvgPool 2x
``masked_scatter`` 要求被替换元素数 == 替换源元素数，故此处裁剪逻辑必须与之一致。
"""

import logging
import time
from typing import Optional

import torch
from torch import nn
from transformers.models.whisper.configuration_whisper import WhisperConfig

from mindspeed.fsdp.utils.log import print_rank
from mindspeed_mm.fsdp.utils.register import model_register
# 注意：框架的 @model_register.register 装饰器不返回类（返回 None），
# 因此模块里的 `Qwen3_5MoeForConditionalGeneration` 名字会被重绑为 None，不能直接 import。
# 必须先触发该模块导入（完成注册），再从 registry 取回真正的类对象。
import mindspeed_mm.fsdp.models.qwen3_5_moe.modeling_qwen3_5_moe  # noqa: F401  触发注册
Qwen3_5MoeForConditionalGeneration = model_register.get("qwen3_5_moe")

from .whisper_encoder import WhisperAudioTower
from .projector import (
    AudioProjector,
    get_feat_lengths_after_conv,
    get_audio_lengths_after_pool,
)

logger = logging.getLogger(__name__)

# tokenizer 自带的音频占位 token id（来自 Qwen3.5-35B-A3B 的 tokenizer_config.json）。
# <|audio_pad|> 是真正用于 masked_scatter 填充声学特征的占位符。
DEFAULT_AUDIO_TOKEN_ID = 248076


@model_register.register("qwen3_5_audio")
class Qwen3_5AudioForConditionalGeneration(Qwen3_5MoeForConditionalGeneration):
    """在 Qwen3.5-MoE 文本塔上接入 Whisper-large-v3 的语音多模态模型。"""

    def __init__(self, config):
        super().__init__(config)

        # ---- audio 配置（由 overwrite_transformer_config 注入到 config）----
        whisper_path = getattr(config, "whisper_path", None)
        self.audio_token_id = getattr(config, "audio_token_id", DEFAULT_AUDIO_TOKEN_ID)
        projector_act = getattr(config, "audio_projector_act", "gelu")

        if whisper_path is None:
            raise ValueError(
                "`whisper_path` 未设置。请在 YAML 的 model 段提供 whisper_path，"
                "指向本地 whisper-large-v3 目录（含 config.json）。"
            )

        # 仅用 config.json 构建结构（不加载权重），meta-device 初始化也安全。
        whisper_config = WhisperConfig.from_pretrained(whisper_path)
        self.audio_tower = WhisperAudioTower(whisper_config)

        llm_hidden = config.text_config.hidden_size  # Qwen3.5-35B-A3B = 2048
        self.audio_projector = AudioProjector(
            audio_hidden_size=whisper_config.d_model,  # 1280
            llm_hidden_size=llm_hidden,
            projector_hidden_act=projector_act,
        )

        self._whisper_path = whisper_path

    @staticmethod
    def overwrite_transformer_config(transformer_config, model_args):
        # 先沿用父类对 text_config 的覆盖（triton gdn / grouped expert matmul）。
        transformer_config = Qwen3_5MoeForConditionalGeneration.overwrite_transformer_config(
            transformer_config, model_args
        )
        # 再把 audio 相关参数从 model_args 透传到 config，供 __init__ 读取。
        transformer_config.whisper_path = getattr(model_args, "whisper_path", None)
        transformer_config.audio_token_id = getattr(
            model_args, "audio_token_id", DEFAULT_AUDIO_TOKEN_ID
        )
        transformer_config.audio_projector_act = getattr(
            model_args, "audio_projector_act", "gelu"
        )
        return transformer_config

    def load_whisper_encoder(self, dtype: torch.dtype = torch.float32):
        """从 HF whisper-large-v3 目录把 encoder 权重灌入 audio_tower。

        在 meta-device 构建 + 基座 DCP 权重加载完成之后调用一次。projector 为新增
        模块，保持随机初始化（SFT 对齐阶段从零训练）。
        """
        loaded = WhisperAudioTower.from_pretrained_encoder(self._whisper_path, dtype=dtype)
        self.audio_tower.load_state_dict(loaded.state_dict())
        print_rank(logger.info, f"[Qwen3_5Audio] whisper encoder 权重已从 {self._whisper_path} 载入")

    def initialize_weights(self):
        """Fast-path initialization for the composite audio model.

        The default HF implementation walks the full multimodal module tree. On this legacy
        pack path that turns into a very expensive traversal after checkpoint loading, while
        only the newly introduced trainable audio projector really needs local initialization
        here. Base text / vision weights are already loaded from the checkpoint, and the
        Whisper tower is populated later by ``load_whisper_encoder()``.
        """
        if not hasattr(self, "audio_projector"):
            return super().initialize_weights()

        init_start = time.time()
        print_rank(logger.info, "[Qwen3_5Audio] initialize_weights fast-path start: audio_projector only")
        self.audio_projector.apply(self._initialize_weights)
        print_rank(
            logger.info,
            f"[Qwen3_5Audio] initialize_weights fast-path finished in {time.time() - init_start:.3f}s",
        )

    def _get_audio_features(
        self,
        input_features: torch.FloatTensor,
        feature_attention_mask: Optional[torch.Tensor],
    ) -> torch.FloatTensor:
        """编码音频并展平成 ``(total_audio_tokens, llm_hidden)``。

        每条样本按有效长度裁剪后再过 projector，确保产出的 token 数逐条等于数据侧
        ``<|audio_pad|>`` 的数量。
        """
        # WhisperEncoder 处理整段（max_length 已 pad 到 3000 mel 帧）→ (B, 1500, 1280)
        hidden = self.audio_tower(input_features)  # (B, T_conv, 1280)
        bsz, t_conv, _ = hidden.shape

        if feature_attention_mask is not None:
            mel_valid = feature_attention_mask.sum(-1)  # (B,) 每条有效 mel 帧数
            conv_lens = get_feat_lengths_after_conv(mel_valid).clamp(max=t_conv)
        else:
            conv_lens = torch.full((bsz,), t_conv, device=hidden.device, dtype=torch.long)

        per_sample = []
        for i in range(bsz):
            valid = int(conv_lens[i].item())
            feat = hidden[i, :valid, :]            # (valid, 1280)
            projected = self.audio_projector(feat)  # (audio_len, 2048)
            per_sample.append(projected)

        return torch.cat(per_sample, dim=0)  # (total_audio_tokens, 2048)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor = None,
        labels: torch.LongTensor = None,
        input_features: torch.FloatTensor = None,
        feature_attention_mask: torch.Tensor = None,
        pixel_values: torch.Tensor = None,
        pixel_values_videos: torch.FloatTensor = None,
        image_grid_thw: torch.LongTensor = None,
        video_grid_thw: torch.LongTensor = None,
        cache_position: torch.LongTensor = None,
        logits_to_keep=0,
        cu_seqlens: torch.LongTensor = None,  # Pack format: cumulative sequence lengths
        **kwargs,
    ):
        """Forward supporting both Pack and Pad formats.

        Pack format (when cu_seqlens is provided):
          - input_ids: (1, total_len) — packed sequence with leading batch dim (set by collator)
          - position_ids: (1, total_len) — per-sample independent position IDs (reset per sample)
          - labels: (1, total_len) — packed labels with leading batch dim
          - cu_seqlens: (batch_size+1,) — cumulative lengths [0, len1, len1+len2, ...]
          - attention_mask: None (triggers native FA2 varlen path via position_ids resets)

        Pad format (legacy, when cu_seqlens is None):
          - input_ids: (batch_size, max_len) padded
          - attention_mask: (batch_size, max_len)
          - position_ids: auto-generated or provided
        """
        # Detect pack vs pad format
        is_packed = cu_seqlens is not None

        # Pack format: enforce attention_mask=None so transformers routes to FA2 varlen
        # via position_ids (the _is_packed_sequence detection in modeling_flash_attention_utils).
        if is_packed:
            attention_mask = None

        # 先取文本 embedding，再把音频特征 scatter 进 <|audio_pad|> 位置。
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
            # Shape after embed:
            #   pack: (1, total_len, hidden)   — collator already added batch dim
            #   pad:  (batch, seq_len, hidden) — original 2D input_ids -> 3D embeds

        if input_features is not None:
            audio_embeds = self._get_audio_features(input_features, feature_attention_mask)
            audio_embeds = audio_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

            if is_packed:
                # Pack format: replace audio tokens sample-by-sample using cu_seqlens boundaries.
                # Both input_ids and inputs_embeds carry leading batch dim of size 1.
                inputs_embeds = self._replace_audio_tokens_packed(
                    inputs_embeds, audio_embeds, input_ids, cu_seqlens
                )
            else:
                # Pad format: original masked_scatter logic
                audio_mask = (input_ids == self.audio_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                n_audio_pos = audio_mask[..., 0].sum()
                if n_audio_pos != audio_embeds.shape[0]:
                    raise ValueError(
                        f"音频 token 数不匹配：input_ids 中 <|audio_pad|> 有 {n_audio_pos} 个，"
                        f"但 projector 产出 {audio_embeds.shape[0]} 个向量。请检查下采样公式一致性。"
                    )
                inputs_embeds = inputs_embeds.masked_scatter(
                    audio_mask.to(inputs_embeds.device), audio_embeds
                )

        # 委托父类：input_ids 置 None（与 inputs_embeds 互斥），保留全部 loss 逻辑。
        # 不传 pixel_values/grid_thw，即关闭视觉分支。
        # Pack format reaches here with shapes:
        #   inputs_embeds: (1, total_len, hidden), position_ids/labels: (1, total_len),
        #   attention_mask: None — triggers FA2 varlen via position_ids resets.
        return super().forward(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

    def _replace_audio_tokens_packed(
        self,
        inputs_embeds: torch.Tensor,  # (1, total_len, hidden)
        audio_embeds: torch.Tensor,   # (total_audio_tokens, hidden)
        input_ids: torch.Tensor,      # (1, total_len)
        cu_seqlens: torch.Tensor,     # (batch_size+1,)
    ) -> torch.Tensor:
        """Replace audio tokens in pack format, sample by sample via cu_seqlens.

        Operates on tensors that carry a leading batch dim of size 1 (set by collator),
        which the framework expects throughout. We index via [0, start:end] so the
        output retains the batch dim for downstream FSDP/loss handling.
        """
        batch_size = len(cu_seqlens) - 1
        audio_offset = 0

        for b in range(batch_size):
            start = cu_seqlens[b].item()
            end = cu_seqlens[b + 1].item()
            sample_ids = input_ids[0, start:end]               # (sample_len,)
            sample_embeds = inputs_embeds[0, start:end]        # (sample_len, hidden)

            audio_positions = (sample_ids == self.audio_token_id).nonzero(as_tuple=True)[0]
            n_audio = len(audio_positions)

            if n_audio > 0:
                sample_audio = audio_embeds[audio_offset : audio_offset + n_audio]
                if sample_audio.shape[0] != n_audio:
                    raise ValueError(
                        f"Sample {b}: audio token count mismatch. "
                        f"input_ids has {n_audio} <|audio_pad|>, but audio_embeds provides {sample_audio.shape[0]}."
                    )
                sample_embeds[audio_positions] = sample_audio
                inputs_embeds[0, start:end] = sample_embeds
                audio_offset += n_audio

        if audio_offset != audio_embeds.shape[0]:
            raise ValueError(
                f"Total audio token mismatch: processed {audio_offset}, but audio_embeds has {audio_embeds.shape[0]}."
            )

        return inputs_embeds
