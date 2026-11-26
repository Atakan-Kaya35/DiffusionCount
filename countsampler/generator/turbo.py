"""Frozen few-step text-to-image generator G (SD-Turbo / SDXL-Turbo). Spec §1.

Only the sampler is trained. Text encoder(s), U-Net and VAE are frozen (requires_grad=False,
eval mode) but the U-Net is *not* wrapped in no_grad, so gradients reach z_n.

Clean estimate at every step (Eq. 1):
    x̂0 = (x_t − √(1−ᾱ_t) ε̂) / √ᾱ_t
First-step input, `input_scaling` (generator.input_scaling in the config):
    "pipeline" (default): x_t = √(1−ᾱ_T) · z_n. This is exactly what diffusers' SD-Turbo /
                SDXL-Turbo Euler "trailing" pipelines feed the U-Net from initial noise z_n
                (√(1−ᾱ_999) ≈ 0.9977), so z_n = ε reproduces the official frozen baseline.
    "identity":   x_t = z_n, the literal reading of the spec's "noisy input x_t = z_n".
Multi-step unrolls (e.g. timesteps [999, 499]) re-noise deterministically:
    x_{t'} = √ᾱ_{t'} x̂0 + √(1−ᾱ_{t'}) ε̂,
which equals diffusers' deterministic Euler step (SD-Turbo's scheduler) exactly.
"""
from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass, field

import torch

from ..utils.misc import resolve_dtype
from .attention import (
    AttentionRequest,
    AttentionStore,
    LayerInfo,
    install_capture_processors,
    is_attention_module,
    restore_processors,
)


@dataclass
class PromptEncoding:
    hidden: torch.Tensor                 # [B, L, Dh] token embeddings c(p_n) (cross-attention keys)
    pooled: torch.Tensor                 # [B, Dp]   pooled embedding c̄ (sampler FiLM input)
    time_ids: torch.Tensor | None = None  # SDXL micro-conditioning [B, 6]

    def __len__(self) -> int:
        return self.hidden.shape[0]


@dataclass
class StepOutput:
    index: int                        # position in the unroll
    t: int                            # diffusion timestep
    alpha_bar: torch.Tensor           # scalar ᾱ_t (clamped)
    x_t: torch.Tensor                 # U-Net input (float32)
    eps_hat: torch.Tensor             # ε̂ (float32)
    x0_hat: torch.Tensor              # x̂0, Eq. (1) (float32, scaled-latent space)
    self_attn: dict[str, torch.Tensor] = field(default_factory=dict)   # name -> [B, N, N]
    cross_attn: dict[str, torch.Tensor] = field(default_factory=dict)  # name -> [B, N, L]
    self_feats: dict[str, torch.Tensor] = field(default_factory=dict)  # name -> [B, N, C]


class FrozenTurboGenerator:
    def __init__(self, *, unet, vae, text_encoders, tokenizers, alphas_cumprod: torch.Tensor,
                 prediction_type: str = "epsilon", family: str = "sd", device="cuda",
                 dtype: torch.dtype = torch.bfloat16, resolution: int = 512,
                 min_alpha_bar: float = 1e-4, gradient_checkpointing: bool = True,
                 decode_chunk: int = 4,
                 default_timesteps: tuple[int, ...] = (999,), input_scaling: str = "pipeline"):
        if family not in ("sd", "sdxl"):
            raise ValueError("family must be 'sd' or 'sdxl'")
        if input_scaling not in ("pipeline", "identity"):
            raise ValueError("input_scaling must be 'pipeline' or 'identity'")
        self.input_scaling = input_scaling
        self.family = family
        self.device = torch.device(device)
        self.dtype = dtype
        self.unet = unet.to(device=self.device, dtype=dtype)
        self.vae = vae.to(device=self.device, dtype=dtype) if vae is not None else None
        self.text_encoders = [te.to(device=self.device, dtype=dtype) for te in text_encoders]
        self.tokenizers = list(tokenizers)
        self.alphas_cumprod = alphas_cumprod.to(device=self.device, dtype=torch.float32)
        self.prediction_type = prediction_type
        self.resolution = int(resolution)
        self.min_alpha_bar = float(min_alpha_bar)
        self.default_timesteps = tuple(int(t) for t in default_timesteps)
        self.vae_scale_factor = (2 ** (len(vae.config.block_out_channels) - 1)) if vae is not None else 8
        self.latent_size = self.resolution // self.vae_scale_factor
        self.latent_channels = int(self.unet.config.in_channels)
        self.store = AttentionStore()
        self._originals: dict[str, object] = {}
        self._request: AttentionRequest | None = None
        self._layer_info: dict[str, LayerInfo] | None = None
        self._prompt_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._token_cache: dict[tuple[str, str], list[int]] = {}
        self.freeze()
        self.decode_chunk = int(decode_chunk)
        if gradient_checkpointing and hasattr(self.unet, "enable_gradient_checkpointing"):
            self.unet.enable_gradient_checkpointing()

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_config(cls, gcfg, device) -> "FrozenTurboGenerator":
        return cls.from_pretrained(
            gcfg.model_id,
            family=gcfg.get("family", "auto"),
            device=device,
            dtype=resolve_dtype(gcfg.get("dtype", "bfloat16")),
            resolution=gcfg.get("resolution", 512),
            gradient_checkpointing=gcfg.get("gradient_checkpointing", True),
            decode_chunk=int(gcfg.get("decode_chunk", 4)),
            min_alpha_bar=gcfg.get("min_alpha_bar", 1e-4),
            default_timesteps=tuple(gcfg.get("timesteps", [999])),
            variant=gcfg.get("variant", "fp16"),
            vae_id=gcfg.get("vae_id"),
            input_scaling=gcfg.get("input_scaling", "pipeline"),
        )

    @classmethod
    def from_pretrained(cls, model_id: str, *, family: str = "auto", device="cuda",
                        dtype: torch.dtype = torch.bfloat16, resolution: int = 512,
                        gradient_checkpointing: bool = True, min_alpha_bar: float = 1e-4,
                        decode_chunk: int = 4,
                        default_timesteps=(999,), variant: str | None = "fp16",
                        vae_id: str | None = None, input_scaling: str = "pipeline") -> "FrozenTurboGenerator":
        from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

        def load(loader, sub, repo=model_id):
            if variant:
                try:
                    return loader.from_pretrained(repo, subfolder=sub, variant=variant)
                except (OSError, ValueError):
                    pass
            return loader.from_pretrained(repo, subfolder=sub)

        if family == "auto":
            family = "sdxl" if "xl" in model_id.lower() else "sd"
        unet = load(UNet2DConditionModel, "unet")
        if vae_id:
            vae = AutoencoderKL.from_pretrained(vae_id)
        else:
            vae = load(AutoencoderKL, "vae")
        sched = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
        tokenizers = [CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")]
        text_encoders = [load(CLIPTextModel, "text_encoder")]
        if family == "sdxl":
            tokenizers.append(CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer_2"))
            text_encoders.append(load(CLIPTextModelWithProjection, "text_encoder_2"))
        return cls(unet=unet, vae=vae, text_encoders=text_encoders, tokenizers=tokenizers,
                   alphas_cumprod=sched.alphas_cumprod,
                   prediction_type=sched.config.get("prediction_type", "epsilon"),
                   family=family, device=device, dtype=dtype, resolution=resolution,
                   min_alpha_bar=min_alpha_bar, gradient_checkpointing=gradient_checkpointing,
                   decode_chunk=decode_chunk,
                   default_timesteps=default_timesteps, input_scaling=input_scaling)

    def freeze(self) -> None:
        for m in [self.unet, self.vae, *self.text_encoders]:
            if m is not None:
                m.requires_grad_(False)
                m.eval()

    # ------------------------------------------------------------------ properties
    @property
    def pooled_dim(self) -> int:
        if self.family == "sdxl":
            return int(self.text_encoders[1].config.projection_dim)
        return int(self.text_encoders[0].config.hidden_size)

    @property
    def latent_shape(self) -> tuple[int, int, int]:
        return (self.latent_channels, self.latent_size, self.latent_size)

    def alpha_bar(self, t: int) -> torch.Tensor:
        return self.alphas_cumprod[int(t)].clamp_min(self.min_alpha_bar)

    # ------------------------------------------------------------------ text
    @torch.no_grad()
    def _encode_uncached(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        def ids_for(tok):
            return tok(prompts, padding="max_length", max_length=tok.model_max_length,
                       truncation=True, return_tensors="pt").input_ids.to(self.device)

        if self.family == "sd":
            out = self.text_encoders[0](ids_for(self.tokenizers[0]))
            return out[0], out.pooler_output
        hidden, pooled = [], None
        for i, (tok, te) in enumerate(zip(self.tokenizers, self.text_encoders)):
            out = te(ids_for(tok), output_hidden_states=True)
            hidden.append(out.hidden_states[-2])
            if i == 1:
                pooled = out[0]  # projected text_embeds of text_encoder_2
        return torch.cat(hidden, dim=-1), pooled

    def encode_prompts(self, prompts: list[str]) -> PromptEncoding:
        """c(p_n) and c̄ for a list of prompts, cached per unique prompt string."""
        missing = [p for p in dict.fromkeys(prompts) if p not in self._prompt_cache]
        for i in range(0, len(missing), 32):
            chunk = missing[i:i + 32]
            hidden, pooled = self._encode_uncached(chunk)
            for j, p in enumerate(chunk):
                self._prompt_cache[p] = (hidden[j], pooled[j])
        hidden = torch.stack([self._prompt_cache[p][0] for p in prompts])
        pooled = torch.stack([self._prompt_cache[p][1] for p in prompts])
        time_ids = None
        if self.family == "sdxl":
            r = float(self.resolution)
            time_ids = torch.tensor([[r, r, 0.0, 0.0, r, r]], device=self.device).repeat(len(prompts), 1)
        return PromptEncoding(hidden=hidden, pooled=pooled, time_ids=time_ids)

    def token_indices(self, prompts: list[str], words: list[str]) -> list[list[int]]:
        """Positions of each prompt's object word in the (BOS-prefixed) 77-token sequence."""
        out = []
        for p, w in zip(prompts, words):
            key = (p, w)
            if key not in self._token_cache:
                self._token_cache[key] = self._find_token_positions(p, w)
            out.append(self._token_cache[key])
        return out

    def _find_token_positions(self, prompt: str, word: str) -> list[int]:
        tok = self.tokenizers[0]
        ids = list(tok(prompt, padding="max_length", max_length=tok.model_max_length, truncation=True).input_ids)
        wid = list(tok(word, add_special_tokens=False).input_ids)
        for start in range(len(ids) - len(wid), -1, -1):  # last occurrence
            if ids[start:start + len(wid)] == wid:
                return list(range(start, start + len(wid)))
        raise ValueError(f"object word '{word}' not found in tokenised prompt '{prompt}'")

    # ------------------------------------------------------------------ attention layers
    def attention_layers(self) -> dict[str, LayerInfo]:
        """All attention layers of the U-Net with kind (self/cross) and token grid (one dry run)."""
        if self._layer_info is not None:
            return self._layer_info
        records: dict[str, tuple[str, int]] = {}
        handles = []
        for name, module in self.unet.named_modules():
            if is_attention_module(module):
                def hook(mod, args, kwargs, _name=name):
                    hs = args[0] if args else kwargs["hidden_states"]
                    tokens = hs.shape[1] if hs.ndim == 3 else hs.shape[2] * hs.shape[3]
                    kind = "cross" if kwargs.get("encoder_hidden_states") is not None else "self"
                    records[_name] = (kind, int(tokens))
                handles.append(module.register_forward_pre_hook(hook, with_kwargs=True))
        try:
            with torch.no_grad():
                enc = self.encode_prompts([""])
                z = torch.zeros(1, *self.latent_shape, device=self.device)
                self._model_output(z, self.default_timesteps[0], enc)
        finally:
            for h in handles:
                h.remove()
        info = {}
        lat = self.latent_size
        for name, (kind, tokens) in records.items():
            side = int(round(lat / math.sqrt(lat * lat / tokens)))
            info[name] = LayerInfo(name=name, kind=kind, tokens=tokens, grid=(side, side))
        self._layer_info = info
        return info

    def layer_grid(self, name: str) -> tuple[int, int]:
        return self.attention_layers()[name].grid

    def layer_grids(self) -> dict[str, tuple[int, int]]:
        return {n: i.grid for n, i in self.attention_layers().items()}

    def watch(self, request: AttentionRequest | None) -> None:
        """Install capture processors for a method's attention request (replaces any previous one)."""
        self.unwatch()
        if request is None or request.is_empty():
            return
        info = self.attention_layers()
        for name in request.self_layers:
            if name not in info or info[name].kind != "self":
                raise ValueError(f"'{name}' is not a self-attention layer (see scripts/list_layers.py)")
        for name in request.cross_layers:
            if name not in info or info[name].kind != "cross":
                raise ValueError(f"'{name}' is not a cross-attention layer (see scripts/list_layers.py)")
        self._originals = install_capture_processors(self.unet, self.store, request)
        self._request = request

    def unwatch(self) -> None:
        if self._originals:
            restore_processors(self.unet, self._originals)
        self._originals = {}
        self._request = None

    # ------------------------------------------------------------------ denoising
    def _model_output(self, x_t: torch.Tensor, t: int, enc: PromptEncoding) -> torch.Tensor:
        b = x_t.shape[0]
        timesteps = torch.full((b,), int(t), device=self.device, dtype=torch.long)
        kwargs = {}
        if self.family == "sdxl":
            kwargs["added_cond_kwargs"] = {"text_embeds": enc.pooled.to(self.dtype),
                                           "time_ids": enc.time_ids.to(self.dtype)}
        out = self.unet(x_t.to(self.dtype), timesteps, encoder_hidden_states=enc.hidden.to(self.dtype),
                        return_dict=False, **kwargs)[0]
        return out.float()

    def forward_at(self, x_t: torch.Tensor, t: int, enc: PromptEncoding, capture: bool = False,
                   index: int = 0) -> StepOutput:
        """One U-Net evaluation at timestep t; returns ε̂, x̂0 (Eq. 1) and captured attention."""
        x_t = x_t.float()
        self.store.begin(capture and self._request is not None)
        try:
            model_out = self._model_output(x_t, t, enc)
        finally:
            caps = self.store.end()
        ab = self.alpha_bar(t)
        sa, s1a = ab.sqrt(), (1.0 - ab).sqrt()
        if self.prediction_type == "epsilon":
            eps_hat = model_out
            x0_hat = (x_t - s1a * eps_hat) / sa
        elif self.prediction_type == "v_prediction":
            x0_hat = sa * x_t - s1a * model_out
            eps_hat = sa * model_out + s1a * x_t
        elif self.prediction_type == "sample":
            x0_hat = model_out
            eps_hat = (x_t - sa * x0_hat) / s1a
        else:
            raise ValueError(f"unsupported prediction_type {self.prediction_type}")
        return StepOutput(index=index, t=int(t), alpha_bar=ab, x_t=x_t, eps_hat=eps_hat, x0_hat=x0_hat,
                          self_attn=caps["self_attn"], cross_attn=caps["cross_attn"],
                          self_feats=caps["self_feats"])

    def initial_input(self, z: torch.Tensor, t: int) -> torch.Tensor:
        """Map initial noise z_n to the first U-Net input x_T (see module docstring)."""
        if self.input_scaling == "pipeline":
            return (1.0 - self.alphas_cumprod[int(t)]).sqrt() * z
        return z

    def unroll(self, z: torch.Tensor, enc: PromptEncoding, timesteps=None,
               capture: bool = False) -> list[StepOutput]:
        """Run the frozen generator from initial noise z for the given timesteps (default: config)."""
        timesteps = list(timesteps or self.default_timesteps)
        outs: list[StepOutput] = []
        x = self.initial_input(z, timesteps[0])
        for i, t in enumerate(timesteps):
            out = self.forward_at(x, t, enc, capture=capture, index=i)
            outs.append(out)
            if i + 1 < len(timesteps):
                ab_next = self.alpha_bar(timesteps[i + 1])
                x = ab_next.sqrt() * out.x0_hat + (1.0 - ab_next).sqrt() * out.eps_hat
        return outs

    @torch.no_grad()
    def generate_latents(self, z: torch.Tensor, enc: PromptEncoding, timesteps=None,
                         chunk: int = 8) -> torch.Tensor:
        """Final x̂0 latents (no grad, no capture), chunked over the batch."""
        res = []
        for i in range(0, z.shape[0], chunk):
            sub = PromptEncoding(enc.hidden[i:i + chunk], enc.pooled[i:i + chunk],
                                 None if enc.time_ids is None else enc.time_ids[i:i + chunk])
            res.append(self.unroll(z[i:i + chunk], sub, timesteps, capture=False)[-1].x0_hat)
        return torch.cat(res)

    # ------------------------------------------------------------------ VAE
    def decode(self, latents: torch.Tensor, grad: bool = False, chunk: int | None = None) -> torch.Tensor:
        """Scaled latents -> images in [-1, 1] (float32). Keep `grad=True` to backprop (image counter).

        `chunk` defaults to `generator.decode_chunk`. The size matters far more than it looks: at
        512 px on an 8 GB card, decoding 8 latents at once reserves 12.3 GB, spills to shared
        memory and takes **11.98 s**, against **1.16 s** at chunk 4 and 1.04 s at chunk 1 - a 10x
        penalty paid silently by every evaluation in this project.
        """
        return vae_decode(self.vae, latents, grad=grad,
                          chunk=self.decode_chunk if chunk is None else chunk)

    def encode_images(self, images: torch.Tensor, chunk: int = 8) -> torch.Tensor:
        """Images in [-1, 1] -> scaled latents (posterior mean)."""
        return vae_encode(self.vae, images, chunk=chunk)


# ---------------------------------------------------------------------------- standalone VAE helpers
def _vae_factors(vae) -> tuple[float, float]:
    sf = float(vae.config.scaling_factor)
    shift = getattr(vae.config, "shift_factor", None) or 0.0
    return sf, float(shift)


def vae_decode(vae, latents: torch.Tensor, grad: bool = False, chunk: int | None = 8) -> torch.Tensor:
    sf, shift = _vae_factors(vae)
    x = latents / sf + shift
    ctx = nullcontext() if grad else torch.no_grad()
    parts = x.split(chunk) if (chunk and not grad) else [x]
    with ctx:
        imgs = [vae.decode(p.to(vae.dtype)).sample.float() for p in parts]
    img = torch.cat(imgs)
    return img if grad else img.clamp(-1, 1)


@torch.no_grad()
def vae_encode(vae, images: torch.Tensor, chunk: int = 8) -> torch.Tensor:
    sf, shift = _vae_factors(vae)
    dev = next(vae.parameters()).device
    out = []
    for p in images.split(chunk):
        lat = vae.encode(p.to(dev, vae.dtype)).latent_dist.mean.float()
        out.append((lat - shift) * sf)
    return torch.cat(out)


def load_vae(gcfg, device) -> torch.nn.Module:
    """Only the frozen VAE (for building the counter's latent cache without loading the U-Net)."""
    from diffusers import AutoencoderKL

    dtype = resolve_dtype(gcfg.get("dtype", "bfloat16"))
    if gcfg.get("vae_id"):
        vae = AutoencoderKL.from_pretrained(gcfg.vae_id)
    else:
        vae = None
        if gcfg.get("variant", "fp16"):
            try:
                vae = AutoencoderKL.from_pretrained(gcfg.model_id, subfolder="vae", variant=gcfg.get("variant", "fp16"))
            except (OSError, ValueError):
                vae = None
        if vae is None:
            vae = AutoencoderKL.from_pretrained(gcfg.model_id, subfolder="vae")
    vae = vae.to(device=device, dtype=dtype)
    vae.requires_grad_(False)
    return vae.eval()
