"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║  ★ NOVEL ★  SHARED COMPONENTS (spec §2). Used by Methods A, B and C.             ║
║  Kept in one file on purpose, so the whole shared part can be read top to bottom.║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  2.1  CountNoiseSampler   z_n = ε + w·Δφ(ε, n, c̄)                      Eq. (2)  ║
║  2.2  normalized_distance d(a,b) = ‖a−b‖₂ / √D                         Eq. (3)  ║
║       g_log / g_lin       target metric between counts                 Eq. (4)  ║
║       alpha_scale         α = 2δ / g_max  (feasibility)                Eq. (8)  ║
║       ordering_loss       regression form                              Eq. (5)  ║
║       ranking_loss        triplet alternative                          Eq. (6)  ║
║       anchor_loss         keep z_n inside a δ-ball around ε            Eq. (7)  ║
║       prior_loss          keep ‖z_n‖²/D ≈ 1                            Eq. (9)  ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  Provenance (§10): a small net whose output modifies the initial noise, plus a   ║
║  mixing weight and a norm prior, follows D2D (which builds on ReNO). The sampler ║
║  form (2) and the geometry losses are proposed here and untested.                ║
║  Remark 1: L_ord alone is satisfied by a collinear embedding, so it only adds    ║
║  smoothness. The count signal must come from Method A, B or C.                   ║
╚══════════════════════════════════════════════════════════════════════════════════╝

KNOBS (configs/base.yaml → `sampler:` and `geometry:`):
    sampler.w, sampler.base_channels, sampler.channel_mult, sampler.mid_attention, sampler.use_text,
    geometry.delta (δ), geometry.g ("log" | "lin"), geometry.ord_form ("regression" | "rank"),
    geometry.rank_margin (μ), geometry.over ("sampled" = §7 step 5 | "all_counts" = Eq. 5 literally)
"""
from __future__ import annotations

import itertools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================================
# 2.1  Sampler Δφ  —  z_n = ε + w Δφ(ε, n, c̄)                                  Eq. (2)
# =====================================================================================


def sinusoidal_count_encoding(n: torch.Tensor, dim: int, max_freq: float = 64.0) -> torch.Tensor:
    """Sinusoidal encoding of log n. n: [B] -> [B, dim]."""
    x = torch.log(n.float().clamp_min(1.0))
    half = dim // 2
    freqs = torch.exp(torch.linspace(0.0, math.log(max_freq), half, device=n.device))
    ang = x[:, None] * freqs[None, :]
    enc = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
    return F.pad(enc, (0, dim - enc.shape[-1]))


class FiLMResBlock(nn.Module):
    """GroupNorm → SiLU → Conv, FiLM (γ, β) from the condition vector, → SiLU → Conv, residual."""

    def __init__(self, ch_in: int, ch_out: int, cond_dim: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(groups, ch_in), ch_in)
        self.conv1 = nn.Conv2d(ch_in, ch_out, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(groups, ch_out), ch_out)
        self.film = nn.Linear(cond_dim, 2 * ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, 3, padding=1)
        self.skip = nn.Conv2d(ch_in, ch_out, 1) if ch_in != ch_out else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        gamma, beta = self.film(cond).chunk(2, dim=-1)
        h = self.norm2(h) * (1 + gamma[..., None, None]) + beta[..., None, None]
        h = self.conv2(F.silu(h))
        return h + self.skip(x)


class SelfAttention2d(nn.Module):
    """Global mixing at the lowest resolution, so the sampler can reason about the whole layout."""

    def __init__(self, ch: int, heads: int = 4, groups: int = 8):
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(min(groups, ch), ch)
        self.qkv = nn.Conv2d(ch, 3 * ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, h * w).unbind(1)
        out = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2))
        return x + self.proj(out.transpose(-1, -2).reshape(b, c, h, w))


class CountNoiseSampler(nn.Module):
    """Δφ: a small conv U-Net with FiLM conditioning on [sin-enc(log n), proj(c̄)].

    The last layer is zero-initialised, so training starts at the identity map z_n = ε.
    """

    def __init__(self, latent_channels: int = 4, text_dim: int = 1024, base_channels: int = 64,
                 channel_mult=(1, 2, 2), blocks_per_level: int = 1, mid_attention: bool = True,
                 cond_dim: int = 256, count_embed_dim: int = 64, count_embed_max_freq: float = 64.0,
                 text_embed_dim: int = 128, use_text: bool = True, w: float = 1.0, groups: int = 8):
        super().__init__()
        self.w = float(w)
        self.use_text = use_text
        self.count_embed_dim = count_embed_dim
        self.count_embed_max_freq = count_embed_max_freq

        in_cond = count_embed_dim
        if use_text:
            self.text_norm = nn.LayerNorm(text_dim)
            self.text_proj = nn.Linear(text_dim, text_embed_dim)
            in_cond += text_embed_dim
        self.cond_mlp = nn.Sequential(nn.Linear(in_cond, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))

        chs = [base_channels * m for m in channel_mult]
        self.conv_in = nn.Conv2d(latent_channels, chs[0], 3, padding=1)
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        prev = chs[0]
        for i, c in enumerate(chs):
            blocks = nn.ModuleList()
            for _ in range(blocks_per_level):
                blocks.append(FiLMResBlock(prev, c, cond_dim, groups))
                prev = c
            self.enc_blocks.append(blocks)
            if i < len(chs) - 1:
                self.downs.append(nn.Conv2d(c, c, 3, stride=2, padding=1))
        self.mid = FiLMResBlock(prev, prev, cond_dim, groups)
        self.mid_attn = SelfAttention2d(prev, groups=groups) if mid_attention else None
        self.dec_blocks = nn.ModuleList()
        self.ups = nn.ModuleList()
        for i in reversed(range(len(chs))):
            if i < len(chs) - 1:
                self.ups.append(nn.Conv2d(prev, prev, 3, padding=1))
            blocks = nn.ModuleList()
            for j in range(blocks_per_level):
                blocks.append(FiLMResBlock(prev + (chs[i] if j == 0 else 0), chs[i], cond_dim, groups))
                prev = chs[i]
            self.dec_blocks.append(blocks)
        self.norm_out = nn.GroupNorm(min(groups, prev), prev)
        self.conv_out = nn.Conv2d(prev, latent_channels, 3, padding=1)
        nn.init.zeros_(self.conv_out.weight)  # zero init → z_n = ε at step 0
        nn.init.zeros_(self.conv_out.bias)

    def condition(self, n: torch.Tensor, c_bar: torch.Tensor | None) -> torch.Tensor:
        h = sinusoidal_count_encoding(n, self.count_embed_dim, self.count_embed_max_freq)
        if self.use_text:
            if c_bar is None:
                raise ValueError("sampler built with use_text=True needs the pooled text embedding c̄")
            h = torch.cat([h, self.text_proj(self.text_norm(c_bar.float()))], dim=-1)
        return self.cond_mlp(h)

    def delta(self, eps: torch.Tensor, n: torch.Tensor, c_bar: torch.Tensor | None = None) -> torch.Tensor:
        cond = self.condition(n, c_bar)
        h = self.conv_in(eps)
        skips = []
        for i, blocks in enumerate(self.enc_blocks):
            for blk in blocks:
                h = blk(h, cond)
            skips.append(h)
            if i < len(self.downs):
                h = self.downs[i](h)
        h = self.mid(h, cond)
        if self.mid_attn is not None:
            h = self.mid_attn(h)
        up_i = 0
        for k, blocks in enumerate(self.dec_blocks):
            level = len(self.enc_blocks) - 1 - k
            if level < len(self.enc_blocks) - 1:
                h = self.ups[up_i](F.interpolate(h, scale_factor=2.0, mode="nearest"))
                up_i += 1
            h = torch.cat([h, skips[level]], dim=1)
            for blk in blocks:
                h = blk(h, cond)
        return self.conv_out(F.silu(self.norm_out(h)))

    def forward(self, eps: torch.Tensor, n: torch.Tensor, c_bar: torch.Tensor | None = None) -> torch.Tensor:
        """Eq. (2): z_n = ε + w Δφ(ε, n, c̄). eps: [B, C, H, W] float32, n: [B] long."""
        return eps + self.w * self.delta(eps, n, c_bar)


def build_sampler(scfg, latent_channels: int, text_dim: int) -> CountNoiseSampler:
    return CountNoiseSampler(
        latent_channels=latent_channels,
        text_dim=text_dim,
        base_channels=scfg.get("base_channels", 64),
        channel_mult=tuple(scfg.get("channel_mult", (1, 2, 2))),
        blocks_per_level=scfg.get("blocks_per_level", 1),
        mid_attention=scfg.get("mid_attention", True),
        cond_dim=scfg.get("cond_dim", 256),
        count_embed_dim=scfg.get("count_embed_dim", 64),
        count_embed_max_freq=scfg.get("count_embed_max_freq", 64.0),
        text_embed_dim=scfg.get("text_embed_dim", 128),
        use_text=scfg.get("use_text", True),
        w=scfg.get("w", 1.0),
    )


# =====================================================================================
# 2.2  Geometry losses
#      Shapes: z [Be, J, C, H, W] (J noises per base noise ε), counts [Be, J], eps [Be, C, H, W]
# =====================================================================================


def normalized_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Eq. (3): d(a, b) = ‖a − b‖₂ / √D over the last three dims. NaN-safe gradient at a = b."""
    diff = (a - b).flatten(-3)
    return torch.sqrt(diff.pow(2).sum(-1) + 1e-12) / math.sqrt(diff.shape[-1])


def g_log(n: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Eq. (4): |log n − log m|."""
    return (torch.log(n.float()) - torch.log(m.float())).abs()


def g_lin(n: torch.Tensor, m: torch.Tensor, n_max: int) -> torch.Tensor:
    """Eq. (4): |n − m| / N_max."""
    return (n.float() - m.float()).abs() / n_max


def g_max(kind: str, n_max: int) -> float:
    """Eq. (8): g_max^log = log N_max, g_max^lin = (N_max − 1) / N_max."""
    return math.log(n_max) if kind == "log" else (n_max - 1) / n_max


def alpha_scale(delta: float, kind: str, n_max: int) -> float:
    """Eq. (8): α = 2δ / g_max, so that α·g_max ≤ 2δ (all pairwise distances fit in the δ-ball)."""
    return 2.0 * delta / g_max(kind, n_max)


class GeometryLosses:
    def __init__(self, delta: float = 0.3, g: str = "log", n_max: int = 10, ord_form: str = "regression",
                 rank_margin: float = 0.05):
        if g not in ("log", "lin"):
            raise ValueError("geometry.g must be 'log' or 'lin'")
        if ord_form not in ("regression", "rank"):
            raise ValueError("geometry.ord_form must be 'regression' or 'rank'")
        self.delta = float(delta)
        self.kind = g
        self.n_max = int(n_max)
        self.ord_form = ord_form
        self.mu = float(rank_margin)
        self.alpha = alpha_scale(self.delta, g, self.n_max)
        self._triplets: dict[int, torch.Tensor] = {}

    def g(self, n: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        return g_log(n, m) if self.kind == "log" else g_lin(n, m, self.n_max)

    # -------------------------------------------------------------------- Eq. (5)
    def ordering_loss(self, z: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """L_ord = E_ε Σ_{n<m} (d(z_n, z_m) − α g(n, m))²."""
        i, j = torch.triu_indices(z.shape[1], z.shape[1], offset=1, device=z.device)
        d = normalized_distance(z[:, i], z[:, j])                       # [Be, P]
        target = self.alpha * self.g(counts[:, i], counts[:, j])
        return ((d - target) ** 2).sum(dim=1).mean()

    # -------------------------------------------------------------------- Eq. (6)
    def ranking_loss(self, z: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """L_ord^rank = E max(0, d(z_a, z_b) − d(z_a, z_c) + μ) over triplets with g(a,b) < g(a,c)."""
        jn = z.shape[1]
        if jn not in self._triplets:
            self._triplets[jn] = torch.tensor(list(itertools.permutations(range(jn), 3)), dtype=torch.long)
        t = self._triplets[jn].to(z.device)
        a, b, c = t[:, 0], t[:, 1], t[:, 2]
        d_ab = normalized_distance(z[:, a], z[:, b])
        d_ac = normalized_distance(z[:, a], z[:, c])
        valid = (self.g(counts[:, a], counts[:, b]) < self.g(counts[:, a], counts[:, c])).float()
        hinge = F.relu(d_ab - d_ac + self.mu)
        return (hinge * valid).sum() / valid.sum().clamp_min(1.0)

    def order_loss(self, z: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        return self.ordering_loss(z, counts) if self.ord_form == "regression" else self.ranking_loss(z, counts)

    # -------------------------------------------------------------------- Eq. (7)
    def anchor_loss(self, z: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """L_anc = E_ε Σ_n max(0, d(z_n, ε) − δ)²."""
        d = normalized_distance(z, eps[:, None])
        return (F.relu(d - self.delta) ** 2).sum(dim=1).mean()

    # -------------------------------------------------------------------- Eq. (9)
    @staticmethod
    def prior_loss(z: torch.Tensor) -> torch.Tensor:
        """L_pri = E_ε Σ_n (‖z_n‖²/D − 1)²."""
        sq = z.flatten(2).pow(2).mean(dim=-1)                            # ‖z‖²/D
        return ((sq - 1.0) ** 2).sum(dim=1).mean()

    # -------------------------------------------------------------------- diagnostics
    @torch.no_grad()
    def diagnostics(self, z: torch.Tensor, eps: torch.Tensor) -> dict[str, float]:
        d_anchor = normalized_distance(z, eps[:, None])
        out = {"geom/d_anchor_mean": d_anchor.mean().item(), "geom/d_anchor_max": d_anchor.max().item(),
               "geom/norm_sq": z.flatten(2).pow(2).mean(-1).mean().item()}
        if z.shape[1] > 1:
            i, j = torch.triu_indices(z.shape[1], z.shape[1], offset=1, device=z.device)
            out["geom/d_pair_mean"] = normalized_distance(z[:, i], z[:, j]).mean().item()
        return out
