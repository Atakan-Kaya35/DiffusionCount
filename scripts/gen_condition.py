"""The conditioning vector Dphi is given, in two variants.

"pooled" is the specification's c_bar. Measured over the six cross cells, its off-diagonal cosine
is +0.848 - the cells are nearly indistinguishable in it, which is why Dphi emitted the IDENTICAL
direction for red and yellow (+1.00) and handed unseen orange that same direction (+0.89).

"rich" appends the token embeddings at the positions that actually vary across cells. At position
6 the off-diagonal cosine is +0.518, so the colour IS recoverable there. Nothing about the count
enters either variant: the prompt is count-free and identical for every target.
"""
import torch
TOKENS = (6, 7, 8, 12)      # the positions with the highest across-cell variance, measured

def condition(enc, variant: str) -> torch.Tensor:
    pooled = enc.pooled.float()
    if variant == "pooled":
        return pooled
    if variant == "rich":
        return torch.cat([pooled] + [enc.hidden[:, k].float() for k in TOKENS], dim=1)
    raise ValueError(variant)
