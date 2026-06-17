"""Legacy custom-registry adapters for Qwen3.5 audio models.

The legacy Megatron entry discovers models through the local custom registry.
The actual implementations are already registered on the FSDP side, so we
reuse those registrations here instead of re-importing the implementation
classes directly.
"""

from mindspeed_mm.fsdp.utils.register import model_register
from mindspeed_mm.models.transformers.custom_model_registry import register_model

# Import for registration side effects, then fetch the real classes from the
# FSDP registry.
from mindspeed_mm.fsdp.models.qwen3_5_audio import manual_ep as _manual_ep  # noqa: F401
from mindspeed_mm.fsdp.models.qwen3_5_audio import modeling_qwen3_5_audio as _modeling  # noqa: F401


Qwen3_5AudioForConditionalGeneration = model_register.get("qwen3_5_audio")
Qwen3_5AudioManualEPForConditionalGeneration = model_register.get("qwen3_5_audio_manual_ep")


register_model("qwen3_5_audio")(Qwen3_5AudioForConditionalGeneration)
register_model("qwen3_5_audio_manual_ep")(Qwen3_5AudioManualEPForConditionalGeneration)
