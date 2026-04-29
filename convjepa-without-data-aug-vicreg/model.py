"""
JEPA model components — Option A (conv-JEPA baseline).

Architecture follows the reference design from Qu et al. (2025):
- ConvNeXt-style encoder with 3D → 2D residual blocks
- Small conv predictor mapping context embedding to target embedding
- Both encoder and predictor operate on full frames (no masking)

Input/output tensor shapes (for num_frames=16, resolution=224, in_chans=11):
    encoder input  : (B, 11, 16, 224, 224)
    encoder output : (B, 128, 14, 14)          # T squeezed after last downsample
    predictor input: (B, 128, 14, 14)
    predictor output: (B, 128, 14, 14)
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from timm.layers import DropPath            # timm >= 0.9
except ImportError:                             # pragma: no cover
    from timm.models.layers import DropPath     # fallback for older timm


# -----------------------------------------------------------------------------
# Building blocks
# -----------------------------------------------------------------------------

class LayerNorm(nn.Module):
    """LayerNorm supporting both channels_last (N,...,C) and channels_first (N,C,...)."""

    def __init__(self, normalized_shape: int, eps: float = 1e-6,
                 data_format: str = "channels_last"):
        super().__init__()
        if data_format not in ("channels_last", "channels_first"):
            raise ValueError(f"Unknown data_format: {data_format}")
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        # channels_first: normalize over channel axis of (B, C, ...).
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        shape = (self.weight.shape[0],) + (1,) * (x.dim() - 2)
        return self.weight.view(shape) * x + self.bias.view(shape)


class ResidualBlock(nn.Module):
    """
    ConvNeXt block: depthwise 7x7 conv → LayerNorm → pointwise MLP (4x expansion)
    with LayerScale and optional stochastic depth. Can operate in 2D or 3D.
    """

    def __init__(self, dim: int, num_spatial_dims: int = 3,
                 layer_scale_init: float = 1e-6, drop_path: float = 0.0):
        super().__init__()
        assert num_spatial_dims in (2, 3)
        self.num_spatial_dims = num_spatial_dims

        if num_spatial_dims == 3:
            self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim)
        else:
            self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)

        self.norm = LayerNorm(dim, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (nn.Parameter(layer_scale_init * torch.ones(dim))
                      if layer_scale_init > 0 else None)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.dwconv(x)
        # (N, C, T, H, W) -> (N, T, H, W, C), or (N, C, H, W) -> (N, H, W, C)
        if self.num_spatial_dims == 3:
            x = x.permute(0, 2, 3, 4, 1)
        else:
            x = x.permute(0, 2, 3, 1)

        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x

        if self.num_spatial_dims == 3:
            x = x.permute(0, 4, 1, 2, 3)
        else:
            x = x.permute(0, 3, 1, 2)

        return identity + self.drop_path(x)


# -----------------------------------------------------------------------------
# Encoder
# -----------------------------------------------------------------------------

class ConvEncoder(nn.Module):
    """
    5-stage ConvNeXt encoder for 16-frame 224x224 physics clips.

    Stages (for dims=[16, 32, 64, 128, 128], num_frames=16):
        stage 0: stem     (B, C_in, 16, 224, 224) -> (B, 16, 16, 224, 224)
        stage 1: down3d   (B, 16,  16, 224, 224) -> (B, 32, 8, 112, 112)
        stage 2: down3d   (B, 32,   8, 112, 112) -> (B, 64, 4, 56,  56)
        stage 3: down3d   (B, 64,   4, 56,  56)  -> (B, 128, 2, 28, 28)
        stage 4: down3d   (B, 128,  2, 28,  28)  -> (B, 128, 1, 14, 14)
                           (time squeezed) -> (B, 128, 14, 14) with 2D res blocks
    """

    def __init__(self,
                 in_chans: int = 11,
                 dims: Sequence[int] = (16, 32, 64, 128, 128),
                 num_res_blocks: Sequence[int] = (3, 3, 3, 9, 3),
                 num_frames: int = 16,
                 drop_path_rate: float = 0.0):
        super().__init__()
        if num_frames != 16:
            raise NotImplementedError(
                f"This baseline encoder is configured for num_frames=16 only "
                f"(got {num_frames}). Other frame counts would need a different "
                f"stride schedule.")
        if len(dims) != 5 or len(num_res_blocks) != 5:
            raise ValueError("Expect 5-stage encoder: dims and num_res_blocks "
                             "must each have length 5.")

        self.dims = list(dims)
        self.num_res_blocks = list(num_res_blocks)

        # Stem: lift in_chans -> dims[0] without spatial downsampling.
        # kernel=(1,4,4) with padding=same preserves (T, H, W).
        stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=(1, 4, 4), padding="same"),
            LayerNorm(dims[0], data_format="channels_first"),
        )

        # Stages 1..4: each downsamples (T, H, W) by 2 in all dims using Conv3d(k=2, s=2).
        #   16x224x224 -> 8x112x112 -> 4x56x56 -> 2x28x28 -> 1x14x14
        down_layers: List[nn.Module] = [stem]
        for i in range(len(dims) - 1):
            down_layers.append(nn.Sequential(
                LayerNorm(dims[i], data_format="channels_first"),
                nn.Conv3d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            ))
        self.downsample_layers = nn.ModuleList(down_layers)

        # Stochastic depth schedule (linear from 0 -> drop_path_rate across all blocks).
        total_blocks = sum(num_res_blocks)
        dpr = [float(x) for x in torch.linspace(0, drop_path_rate, total_blocks)]

        # Residual blocks. The last stage has T=1, so we use 2D blocks there.
        res_blocks: List[nn.Module] = []
        cur = 0
        for i, nblocks in enumerate(num_res_blocks):
            spatial_dims = 3 if i < len(dims) - 1 else 2
            stage = nn.Sequential(*[
                ResidualBlock(dims[i], num_spatial_dims=spatial_dims, drop_path=dpr[cur + j])
                for j in range(nblocks)
            ])
            res_blocks.append(stage)
            cur += nblocks
        self.res_blocks = nn.ModuleList(res_blocks)

        self.out_channels = dims[-1]
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, T, H, W), e.g. (B, 11, 16, 224, 224).
        Returns:
            (B, C_out, H', W'), e.g. (B, 128, 14, 14).
        """
        for ds, res in zip(self.downsample_layers, self.res_blocks):
            x = ds(x)
            # After the last downsample, T collapses to 1 — drop it so 2D blocks work.
            if x.dim() == 5 and x.shape[2] == 1:
                x = x.squeeze(2)
            x = res(x)
        return x


# -----------------------------------------------------------------------------
# Predictor
# -----------------------------------------------------------------------------

class ConvPredictor(nn.Module):
    """
    Small conv head that maps context embedding -> predicted target embedding.

    Takes (B, C, H, W) and returns (B, C, H, W) through an intermediate widening
    (C -> 2C) with one residual block, then narrowing back (2C -> C). The
    padding/kernel choice matches the reference implementation exactly so the
    spatial shape survives the two conv2d operations.
    """

    def __init__(self, channels: int = 128, expansion: int = 2):
        super().__init__()
        hidden = channels * expansion
        self.conv = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=2, padding=1),
            ResidualBlock(hidden, num_spatial_dims=2),
            nn.Conv2d(hidden, channels, kernel_size=2),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------

def build_jepa(cfg) -> tuple[ConvEncoder, ConvPredictor]:
    """Build encoder + predictor from a config namespace with a `model` section."""
    m = cfg.model
    encoder = ConvEncoder(
        in_chans=cfg.dataset.num_chans,
        dims=list(m.dims),
        num_res_blocks=list(m.num_res_blocks),
        num_frames=cfg.dataset.num_frames,
        drop_path_rate=m.get("drop_path_rate", 0.0),
    )
    predictor = ConvPredictor(
        channels=m.dims[-1],
        expansion=m.get("predictor_expansion", 2),
    )
    return encoder, predictor


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
