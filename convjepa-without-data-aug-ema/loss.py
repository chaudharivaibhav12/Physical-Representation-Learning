"""
EMA-JEPA loss — pure MSE between predictor output and EMA target embeddings.

The target encoder is maintained as an exponential moving average of the
online encoder in train.py. This module wraps MSE into the same dict format
used across the codebase for consistent logging.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def ema_loss(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    MSE loss between predictor output and (stop-gradient) EMA target embeddings.

    Accepts (B, C, H, W) or (B, C, T, H, W). Returns a dict with 'loss'
    (backprop-ready scalar) and 'repr_loss' (same value, detached, for logging).
    The EMA target encoder already has no gradient; target is treated as a
    constant here.
    """
    assert pred.shape == target.shape, (
        f"pred and target must match: {pred.shape} vs {target.shape}")
    loss = F.mse_loss(pred, target)
    return {"loss": loss, "repr_loss": loss.detach()}
