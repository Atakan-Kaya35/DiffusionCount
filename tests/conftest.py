"""Tiny random SD-style components so the whole pipeline runs on CPU without downloads."""
from __future__ import annotations

import sys
import zlib
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from countsampler.generator import FrozenTurboGenerator  # noqa: E402
from countsampler.utils.config import load_config  # noqa: E402

TINY_SELF = "down_blocks.0.attentions.0.transformer_blocks.0.attn1"
TINY_CROSS = "down_blocks.0.attentions.0.transformer_blocks.0.attn2"


class FakeTokenizer:
    """Whitespace tokenizer with CLIP-like BOS/EOS/padding behaviour."""

    model_max_length = 77

    def __init__(self, vocab_size: int = 1000):
        self.bos, self.eos = vocab_size - 2, vocab_size - 1
        self.vocab = vocab_size - 2

    def _ids(self, text):
        return [zlib.crc32(w.encode()) % self.vocab for w in text.lower().replace(",", " ").split()]

    def __call__(self, text, padding=None, max_length=None, truncation=False, return_tensors=None,
                 add_special_tokens=True):
        single = isinstance(text, str)
        rows = []
        for t in ([text] if single else text):
            ids = self._ids(t)
            if add_special_tokens:
                ids = [self.bos] + ids + [self.eos]
            if truncation and max_length:
                ids = ids[:max_length]
            if padding == "max_length":
                ids = ids + [self.eos] * (max_length - len(ids))
            rows.append(ids)

        class Out:
            pass

        out = Out()
        out.input_ids = torch.tensor(rows) if return_tensors == "pt" else (rows[0] if single else rows)
        return out


def make_tiny_generator(gradient_checkpointing: bool = True) -> FrozenTurboGenerator:
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from transformers import CLIPTextConfig, CLIPTextModel

    torch.manual_seed(0)
    unet = UNet2DConditionModel(
        sample_size=16, in_channels=4, out_channels=4, layers_per_block=1, block_out_channels=(32, 64),
        down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D"), cross_attention_dim=32,
        attention_head_dim=8, norm_num_groups=8)
    vae = AutoencoderKL(in_channels=3, out_channels=3, down_block_types=["DownEncoderBlock2D"] * 2,
                        up_block_types=["UpDecoderBlock2D"] * 2, block_out_channels=[16, 32], latent_channels=4,
                        norm_num_groups=8, sample_size=32)
    te = CLIPTextModel(CLIPTextConfig(vocab_size=1000, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                      num_attention_heads=4, max_position_embeddings=77, bos_token_id=998,
                                      eos_token_id=999, pad_token_id=999, projection_dim=32))
    sched = DDPMScheduler(beta_schedule="scaled_linear", beta_start=0.00085, beta_end=0.012)
    return FrozenTurboGenerator(unet=unet, vae=vae, text_encoders=[te], tokenizers=[FakeTokenizer()],
                                alphas_cumprod=sched.alphas_cumprod, prediction_type="epsilon", family="sd",
                                device="cpu", dtype=torch.float32, resolution=32,
                                gradient_checkpointing=gradient_checkpointing)


@pytest.fixture(scope="session")
def tiny_gen() -> FrozenTurboGenerator:
    return make_tiny_generator()


def tiny_cfg(method_file: str, tmp_path, extra=()):
    overrides = [
        f"experiment.output_dir={tmp_path}", "experiment.device=cpu",
        "train.eps_per_batch=2", "train.counts_per_eps=3", "train.log_every=1", "train.ckpt_every=0",
        "train.preview.every=0", "train.warmup_steps=0", "train.lr=1e-3",
        "data.colors=[red,blue]", "data.train_shapes=[circle,square]", "data.heldout_shapes=[star]",
        f"method.attention.self_layers=[{TINY_SELF}]", f"method.attention.cross_layers=[{TINY_CROSS}]",
        "method.spectral.K=64", "method.spectral.min_foreground=2", "method.spectral.tau_a=0.0",
        "sampler.base_channels=16", "sampler.channel_mult=[1,2]",
        *extra,
    ]
    return load_config(ROOT / "configs" / method_file, overrides)
