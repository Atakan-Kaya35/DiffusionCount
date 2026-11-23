"""Frozen generator G and attention capture (infrastructure, not part of the novel surface)."""
from .attention import (
    AttentionRequest,
    AttentionStore,
    CaptureAttnProcessor,
    LayerInfo,
    concat_self_features,
    mean_self_attention,
    minmax_normalize,
    object_attention_map,
)
from .turbo import FrozenTurboGenerator, PromptEncoding, StepOutput, load_vae, vae_decode, vae_encode
