"""
VICReg loss (Bardes, Ponce & LeCun, 2022) adapted for dense spatial embeddings.

The loss has three terms:
    L = sim_coeff  * MSE(pred, target)                      # invariance
      + std_coeff  * [hinge(1 - std(pred)) + hinge(1 - std(target))]
      + cov_coeff  * [off_diag(cov(pred))^2 + off_diag(cov(target))^2]

For dense (B, C, H, W) embeddings we flatten spatial positions into the
"sample" dimension — every spatial location becomes one VICReg sample. With
14x14=196 positions per clip and batch_size=8, that gives ~1568 samples per
batch, which is enough for stable variance/covariance estimates on C=128.

We optionally shuffle-and-chunk the flat rows before computing statistics.
This follows the reference implementation and reduces memory while keeping
each chunk's statistics valid (random subsamples are still iid from the
underlying distribution).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict

import torch
import torch.nn.functional as F
from einops import rearrange


def _off_diagonal(m: torch.Tensor) -> torch.Tensor:
    """Return a flat view of the off-diagonal entries of a square matrix."""
    n, p = m.shape
    assert n == p, "off_diagonal expects a square matrix"
    return m.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def _vicreg_core(x: torch.Tensor, y: torch.Tensor,
                 sim_coeff: float, std_coeff: float, cov_coeff: float,
                 fp32_stats: bool = True) -> Dict[str, torch.Tensor]:
    """
    Core VICReg computation on flat (N, C) embeddings.

    Returns a dict with total loss and each component (detached for logging).
    """
    n, c = x.shape

    # Invariance term in original dtype.
    repr_loss = F.mse_loss(x, y)

    # For statistical terms, optionally upcast to fp32 for stability.
    xs = x.float() if fp32_stats else x
    ys = y.float() if fp32_stats else y

    # Center features.
    xs = xs - xs.mean(dim=0)
    ys = ys - ys.mean(dim=0)

    # Variance (hinge-at-1 on per-channel std).
    std_x = torch.sqrt(xs.var(dim=0, unbiased=False) + 1e-4)
    std_y = torch.sqrt(ys.var(dim=0, unbiased=False) + 1e-4)
    std_loss = (F.relu(1.0 - std_x).mean() + F.relu(1.0 - std_y).mean()) / 2.0

    # Covariance (sum of squared off-diagonals, normalized by channel count).
    cov_x = (xs.T @ xs) / max(1, n - 1)
    cov_y = (ys.T @ ys) / max(1, n - 1)
    cov_loss = (_off_diagonal(cov_x).pow(2).sum() / c
                + _off_diagonal(cov_y).pow(2).sum() / c)

    total = sim_coeff * repr_loss + std_coeff * std_loss + cov_coeff * cov_loss

    return {
        "loss": total,
        "repr_loss": repr_loss.detach(),
        "std_loss": std_loss.detach(),
        "cov_loss": cov_loss.detach(),
    }


def vicreg_loss(pred: torch.Tensor, target: torch.Tensor,
                sim_coeff: float = 2.0, std_coeff: float = 40.0,
                cov_coeff: float = 2.0, n_chunks: int = 5,
                fp32_stats: bool = True) -> Dict[str, torch.Tensor]:
    """
    VICReg loss on dense embeddings.

    Accepts (B, C, H, W) or (B, C, T, H, W). Flattens all non-channel axes into
    a "samples" dimension, shuffles, splits into n_chunks, computes VICReg on
    each chunk, and averages.

    Args:
        pred:   predicted embeddings, same shape as target.
        target: encoder-produced target embeddings.
        sim_coeff, std_coeff, cov_coeff: VICReg coefficients.
        n_chunks: number of chunks to split the flat samples into.
        fp32_stats: compute variance/covariance in fp32 (recommended).

    Returns:
        Dict of loss components. `loss` is the backprop-ready scalar.
    """
    assert pred.shape == target.shape, (
        f"pred and target must match: {pred.shape} vs {target.shape}")

    if pred.dim() == 4:  # (B, C, H, W)
        x = rearrange(pred, "b c h w -> (b h w) c")
        y = rearrange(target, "b c h w -> (b h w) c")
    elif pred.dim() == 5:  # (B, C, T, H, W)
        x = rearrange(pred, "b c t h w -> (b t h w) c")
        y = rearrange(target, "b c t h w -> (b t h w) c")
    else:
        raise ValueError(f"Expected 4D or 5D tensor, got shape {pred.shape}")

    n = x.shape[0]
    perm = torch.randperm(n, device=x.device)
    x = x[perm]
    y = y[perm]

    n_chunks = max(1, int(n_chunks))
    x_chunks = x.chunk(n_chunks, dim=0)
    y_chunks = y.chunk(n_chunks, dim=0)

    agg: Dict[str, list] = defaultdict(list)
    for xc, yc in zip(x_chunks, y_chunks):
        out = _vicreg_core(xc, yc, sim_coeff, std_coeff, cov_coeff,
                           fp32_stats=fp32_stats)
        for k, v in out.items():
            agg[k].append(v)

    return {k: torch.stack(v).mean() for k, v in agg.items()}
