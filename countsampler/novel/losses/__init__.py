"""★ NOVEL ★ Method-specific loss functions, one file per method.

    spectral.py  Method A (§3, Eq. 10-14)
    density.py   Method B (§4, Eq. 16)
    peak.py      Method C (§5, Eq. 18-21)

The shared geometry losses (§2.2) are in ../shared.py.
"""
from .density import category_targets, density_count_loss, timestep_weight
from .peak import PeakParams, PeakResult, init_centers, local_maxima, peak_losses, soft_kmeans
from .spectral import (
    SpectralParams,
    SpectralResult,
    attention_affinity,
    feature_affinity,
    normalized_laplacian,
    select_foreground,
    soft_count,
    spectral_count_losses,
    spectral_losses_from_affinity,
)
