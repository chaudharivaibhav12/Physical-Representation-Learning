"""
VICReg Model for Active Matter Physical Simulations
====================================================
Architecture:
  - Encoder:   3D ViT (same backbone as ViT-JEPA teammate for fair comparison)
  - Projector: MLP 384 → 2048 → 2048 (used only during training, discarded at eval)
  - Loss:      VICReg (Variance + Invariance + Covariance)

Training flow:
  view1 (B, 11, 16, 224, 224) → Encoder → pool → (B, 384) → Projector → z1 (B, 2048)
  view2 (B, 11, 16, 224, 224) → Encoder → pool → (B, 384) → Projector → z2 (B, 2048)
  Loss = VICReg(z1, z2)

Eval flow (projector discarded):
  x (B, 11, 16, 224, 224) → Encoder → pool → (B, 384)  ← linear probe / kNN

Patch config (matches ViT-JEPA for fair comparison):
  spatial patch:  32x32  → 7x7   = 49 spatial tokens
  temporal patch: 2      → 16/2  = 8  temporal tokens
  total tokens:   49 x 8 = 392
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# 1. Patch Embedding (identical to ViT-JEPA)
# ─────────────────────────────────────────────

class PatchEmbed3D(nn.Module):
    """
    Spatiotemporal patch embedding via Conv3D.
    Input:  (B, C, T, H, W)
    Output: (B, num_patches, embed_dim)
    """
    def __init__(
        self,
        in_channels: int = 11,
        embed_dim:   int = 384,
        img_size:    int = 224,
        patch_size:  int = 32,
        tubelet:     int = 2,
        num_frames:  int = 16,
    ):
        super().__init__()
        assert img_size   % patch_size == 0
        assert num_frames % tubelet    == 0

        self.num_t       = num_frames  // tubelet
        self.num_h       = img_size    // patch_size
        self.num_w       = img_size    // patch_size
        self.num_patches = self.num_t * self.num_h * self.num_w

        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=(tubelet, patch_size, patch_size),
            stride=(tubelet, patch_size, patch_size),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)           # (B, D, num_t, num_h, num_w)
        x = x.flatten(2)           # (B, D, num_patches)
        x = x.transpose(1, 2)     # (B, num_patches, D)
        return self.norm(x)


# ─────────────────────────────────────────────
# 2. Positional Embedding (3D sinusoidal)
# ─────────────────────────────────────────────

def get_3d_sincos_pos_embed(embed_dim, num_t, num_h, num_w) -> torch.Tensor:
    """Returns (1, num_t*num_h*num_w, embed_dim)."""
    assert embed_dim % 3 == 0
    d = embed_dim // 3

    def sincos_1d(length, dim):
        pos   = torch.arange(length, dtype=torch.float32)
        i     = torch.arange(dim // 2, dtype=torch.float32)
        theta = pos.unsqueeze(1) / (10000 ** (2 * i / dim)).unsqueeze(0)
        return torch.cat([theta.sin(), theta.cos()], dim=-1)

    t_emb = sincos_1d(num_t, d)
    h_emb = sincos_1d(num_h, d)
    w_emb = sincos_1d(num_w, d)

    t_emb = t_emb[:, None, None, :].expand(num_t, num_h, num_w, d)
    h_emb = h_emb[None, :, None, :].expand(num_t, num_h, num_w, d)
    w_emb = w_emb[None, None, :, :].expand(num_t, num_h, num_w, d)

    pos = torch.cat([t_emb, h_emb, w_emb], dim=-1)
    return pos.reshape(1, num_t * num_h * num_w, embed_dim)


# ─────────────────────────────────────────────
# 3. Transformer Building Blocks
# ─────────────────────────────────────────────

class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 6, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.attn_drop = dropout
        self.qkv       = nn.Linear(dim, dim * 3, bias=True)
        self.proj      = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop if self.training else 0.0)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = Attention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = MLP(dim, mlp_ratio, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ─────────────────────────────────────────────
# 4. ViT Encoder (identical backbone to ViT-JEPA)
# ─────────────────────────────────────────────

class ViTEncoder(nn.Module):
    """
    3D Vision Transformer encoder.

    forward()        → (B, N, D)  full token sequence
    forward_pooled() → (B, D)     globally mean-pooled (used for eval + projector input)
    """
    def __init__(
        self,
        in_channels: int   = 11,
        embed_dim:   int   = 384,
        depth:       int   = 6,
        num_heads:   int   = 6,
        mlp_ratio:   float = 4.0,
        dropout:     float = 0.0,
        img_size:    int   = 224,
        patch_size:  int   = 32,
        tubelet:     int   = 2,
        num_frames:  int   = 16,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed3D(in_channels, embed_dim, img_size, patch_size, tubelet, num_frames)
        num_t = num_frames // tubelet
        num_h = img_size   // patch_size
        num_w = img_size   // patch_size

        pos_embed = get_3d_sincos_pos_embed(embed_dim, num_t, num_h, num_w)
        self.register_buffer("pos_embed", pos_embed)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv3d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns full token sequence (B, N, D)."""
        x = self.patch_embed(x) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward_pooled(self, x: torch.Tensor) -> torch.Tensor:
        """Returns globally pooled embedding (B, D). Used for eval."""
        return self.forward(x).mean(dim=1)


# ─────────────────────────────────────────────
# 5. Projection MLP (training only, discarded at eval)
# ─────────────────────────────────────────────

class ProjectionMLP(nn.Module):
    """
    VICReg expander: 384 → 2048 → 2048.
    BatchNorm between layers (standard VICReg design).
    No activation on final output so covariance can capture full range.
    Discarded after training — only the encoder is used for evaluation.
    """
    def __init__(self, in_dim: int = 384, hidden_dim: int = 2048, out_dim: int = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,    hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim,   bias=False),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim,    out_dim,   bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────
# 6. VICReg Loss
# ─────────────────────────────────────────────

class VICRegLoss(nn.Module):
    """
    VICReg: Variance-Invariance-Covariance Regularization.
    Applied between projected embeddings z1 and z2 (B, D).

    Loss = sim_weight * invariance(z1, z2)
         + var_weight * variance(z1, z2)
         + cov_weight * covariance(z1, z2)
    """
    def __init__(
        self,
        sim_weight: float = 25.0,
        var_weight: float = 25.0,
        cov_weight: float = 1.0,
        eps:        float = 1e-4,
    ):
        super().__init__()
        self.sim_weight = sim_weight
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        self.eps        = eps

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> tuple:
        B, D = z1.shape

        # Invariance: push embeddings of same simulation together
        loss_inv = F.mse_loss(z1, z2)

        # Variance: prevent collapse by keeping per-dim std >= 1
        def var_loss(z):
            z   = z - z.mean(dim=0)
            std = torch.sqrt(z.var(dim=0) + self.eps)
            return F.relu(1.0 - std).mean()

        loss_var = var_loss(z1) + var_loss(z2)

        # Covariance: decorrelate embedding dimensions
        def cov_loss(z):
            z   = z - z.mean(dim=0)
            cov = (z.T @ z) / (B - 1)
            # Sum of squared off-diagonal elements, normalized by D
            off_diag = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
            return off_diag / D

        loss_cov = cov_loss(z1) + cov_loss(z2)

        loss = (
            self.sim_weight * loss_inv +
            self.var_weight * loss_var +
            self.cov_weight * loss_cov
        )

        metrics = {
            "loss_total":      loss.item(),
            "loss_invariance": (self.sim_weight * loss_inv).item(),
            "loss_variance":   (self.var_weight * loss_var).item(),
            "loss_covariance": (self.cov_weight * loss_cov).item(),
            "loss_inv_raw":    loss_inv.item(),
            "loss_var_raw":    loss_var.item(),
            "loss_cov_raw":    loss_cov.item(),
        }
        return loss, metrics


# ─────────────────────────────────────────────
# 7. Full VICReg Model
# ─────────────────────────────────────────────

class VICReg(nn.Module):
    """
    Full VICReg model.

    Training:
      view1 → encoder → pool → projector → z1
      view2 → encoder → pool → projector → z2
      loss  = VICReg(z1, z2)

    Eval (projector discarded):
      x → encoder → pool → (B, embed_dim)  ← frozen for linear probe / kNN
    """
    def __init__(
        self,
        # Encoder
        in_channels:  int   = 11,
        embed_dim:    int   = 384,
        depth:        int   = 6,
        num_heads:    int   = 6,
        mlp_ratio:    float = 4.0,
        dropout:      float = 0.0,
        img_size:     int   = 224,
        patch_size:   int   = 32,
        tubelet:      int   = 2,
        num_frames:   int   = 16,
        # Projector
        proj_hidden:  int   = 2048,
        proj_out:     int   = 2048,
        # VICReg loss
        sim_weight:   float = 25.0,
        var_weight:   float = 25.0,
        cov_weight:   float = 1.0,
    ):
        super().__init__()

        self.encoder = ViTEncoder(
            in_channels=in_channels, embed_dim=embed_dim,
            depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
            dropout=dropout, img_size=img_size, patch_size=patch_size,
            tubelet=tubelet, num_frames=num_frames,
        )
        self.projector = ProjectionMLP(embed_dim, proj_hidden, proj_out)
        self.vicreg    = VICRegLoss(sim_weight, var_weight, cov_weight)

    def forward(self, view1: torch.Tensor, view2: torch.Tensor) -> tuple:
        """Training forward pass."""
        z1 = self.projector(self.encoder.forward_pooled(view1))  # (B, proj_out)
        z2 = self.projector(self.encoder.forward_pooled(view2))  # (B, proj_out)
        return self.vicreg(z1, z2)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract frozen encoder embedding for evaluation."""
        self.encoder.eval()
        return self.encoder.forward_pooled(x)   # (B, embed_dim)

    def count_parameters(self) -> dict:
        enc  = sum(p.numel() for p in self.encoder.parameters()   if p.requires_grad)
        proj = sum(p.numel() for p in self.projector.parameters() if p.requires_grad)
        return {"encoder": enc, "projector": proj, "total": enc + proj}


# ─────────────────────────────────────────────
# 8. Sanity Check
# ─────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    model = VICReg().to(device)

    params = model.count_parameters()
    print(f"[PARAMS] encoder:    {params['encoder']:,}")
    print(f"[PARAMS] projector:  {params['projector']:,}")
    print(f"[PARAMS] total:      {params['total']:,}")
    print(f"[PARAMS] < 100M:     {params['total'] < 100_000_000}\n")

    B    = 2
    v1   = torch.randn(B, 11, 16, 224, 224, device=device)
    v2   = torch.randn(B, 11, 16, 224, 224, device=device)

    print("[SHAPE] Encoder pooled output:")
    z = model.encoder.forward_pooled(v1)
    print(f"  {v1.shape} → {z.shape}  (expected ({B}, 384))\n")

    print("[SHAPE] Projector output:")
    zp = model.projector(z)
    print(f"  {z.shape} → {zp.shape}  (expected ({B}, 2048))\n")

    print("[FORWARD] Full forward pass...")
    loss, metrics = model(v1, v2)
    print(f"  loss: {loss.item():.4f}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\n[BACKWARD] Backward pass...")
    loss.backward()
    print(f"  Encoder has gradients: {any(p.grad is not None for p in model.encoder.parameters())}")

    model.eval()
    with torch.no_grad():
        emb = model.encode(v1)
    print(f"\n[EMBED] shape: {emb.shape}  (expected ({B}, 384))")
    print("\n✓ All checks passed!")
