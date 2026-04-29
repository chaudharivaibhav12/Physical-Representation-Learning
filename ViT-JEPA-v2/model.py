"""
ViT-JEPA v2 for Active Matter Physical Simulations
====================================================
Changes from v1:
  1. Token-level VICReg: variance/covariance computed on (B*N, D) not (B, D)
     → 1568 samples per batch instead of 4; statistically meaningful gradients
  2. fp32 upcast for all VICReg statistics (bf16 is too imprecise for cov/var)
  3. std_weight 40→20, variance term normalized by /2, unbiased=False + eps=1e-4
  4. Stop-gradient on target encoder path (torch.no_grad on target forward)
  5. Predictor returns token sequence before pooling; pooling moved to ViTJEPA
Architecture:
  - Encoder:   3D Vision Transformer with spatiotemporal patch embedding
  - Predictor: Shallow Transformer (2 layers) on full token sequence
  - Loss:      VICReg — token-level stats, pooled-vector invariance

Training Flow:
  1. context → Encoder → tokens (B, 392, 384)
                       → Predictor → pred_tokens (B, 392, 384)
                       → pool     → z_pred (B, 384)
  2. target  → SAME Encoder (stop-grad) → tgt_tokens (B, 392, 384)
                       → pool           → z_tgt (B, 384)
  3. VICReg:
       invariance  = MSE(z_pred, z_tgt)             on (B, 384)
       variance    = hinge(1 - std(tokens))          on (B*392, 384) in fp32
       covariance  = off_diag(cov(tokens))^2 / D    on (B*392, 384) in fp32

Patch config (patch_size=32):
  spatial patch:  32x32  →  7x7  = 49 spatial tokens
  temporal patch: 2      →  16/2 = 8  temporal tokens
  total tokens:   49 x 8 = 392 per clip
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# 1. Patch Embedding
# ─────────────────────────────────────────────

class PatchEmbed3D(nn.Module):
    """
    Splits a spatiotemporal clip into non-overlapping 3D patches
    and linearly projects each patch to embed_dim via Conv3D.

    Input:  (B, C, T, H, W)
    Output: (B, num_patches, embed_dim)

    With patch_size=32, tubelet=2, T=16, H=W=224:
      num_patches = 8 x 7 x 7 = 392
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
        assert img_size   % patch_size == 0, "img_size must be divisible by patch_size"
        assert num_frames % tubelet    == 0, "num_frames must be divisible by tubelet"

        self.num_t       = num_frames  // tubelet
        self.num_h       = img_size    // patch_size
        self.num_w       = img_size    // patch_size
        self.num_patches = self.num_t * self.num_h * self.num_w
        self.embed_dim   = embed_dim

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
# 2. Positional Embedding (3D Sinusoidal)
# ─────────────────────────────────────────────

def get_3d_sincos_pos_embed(
    embed_dim: int,
    num_t: int,
    num_h: int,
    num_w: int,
) -> torch.Tensor:
    """
    Fixed 3D sinusoidal positional embedding.
    Splits embed_dim into 3 equal parts for t, h, w axes.
    Returns: (1, num_t * num_h * num_w, embed_dim)
    """
    assert embed_dim % 3 == 0, "embed_dim must be divisible by 3"
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
    """Multi-head self-attention using F.scaled_dot_product_attention."""
    def __init__(self, dim: int, num_heads: int = 6, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.attn_drop = dropout

        self.qkv  = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop if self.training else 0.0,
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: x = x + Attn(LN(x)); x = x + MLP(LN(x))."""
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = Attention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = MLP(dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ─────────────────────────────────────────────
# 4. ViT Encoder
# ─────────────────────────────────────────────

class ViTEncoder(nn.Module):
    """
    3D Vision Transformer encoder.

    forward()        → (B, N, D)  full token sequence (training)
    forward_pooled() → (B, D)     globally pooled (evaluation / linear probe / kNN)
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

        self.patch_embed = PatchEmbed3D(
            in_channels=in_channels,
            embed_dim=embed_dim,
            img_size=img_size,
            patch_size=patch_size,
            tubelet=tubelet,
            num_frames=num_frames,
        )
        num_t = num_frames  // tubelet
        num_h = img_size    // patch_size
        num_w = img_size    // patch_size

        pos_embed = get_3d_sincos_pos_embed(embed_dim, num_t, num_h, num_w)
        self.register_buffer("pos_embed", pos_embed)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv3d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns full token sequence (B, N, D). Used during training."""
        x = self.patch_embed(x)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward_pooled(self, x: torch.Tensor) -> torch.Tensor:
        """Returns globally pooled embedding (B, D). Used at evaluation time."""
        tokens = self.forward(x)
        return tokens.mean(dim=1)


# ─────────────────────────────────────────────
# 5. Shallow Transformer Predictor
# ─────────────────────────────────────────────

class TransformerPredictor(nn.Module):
    """
    Shallow transformer predictor on the full token sequence.

    Returns (B, N, encoder_dim) — NOT pooled.
    Pooling is done in ViTJEPA.forward() so both the token sequence and the
    pooled vector are available for token-level VICReg statistics.

    Flow:
      tokens (B, N, encoder_dim)
        → input_proj  → (B, N, predictor_dim)
        → 2x TransformerBlock
        → LayerNorm
        → output_proj → (B, N, encoder_dim)   ← returned
    """
    def __init__(
        self,
        encoder_dim:   int   = 384,
        predictor_dim: int   = 192,
        depth:         int   = 2,
        num_heads:     int   = 4,
        mlp_ratio:     float = 4.0,
        dropout:       float = 0.0,
    ):
        super().__init__()
        self.input_proj  = nn.Linear(encoder_dim, predictor_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(predictor_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm        = nn.LayerNorm(predictor_dim)
        self.output_proj = nn.Linear(predictor_dim, encoder_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (B, N, encoder_dim)
        Returns:
            pred_tokens: (B, N, encoder_dim) — token sequence, not pooled
        """
        x = self.input_proj(tokens)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.output_proj(x)


# ─────────────────────────────────────────────
# 6. VICReg Loss (token-level, fp32 stats)
# ─────────────────────────────────────────────

class VICRegLoss(nn.Module):
    """
    VICReg v2 — token-level variance/covariance, fp32 statistics.

    Invariance:   MSE on pooled vectors (B, D)   — temporal prediction signal
    Variance:     computed on flat tokens (B*N, D) in fp32 — 1568 samples not 4
    Covariance:   computed on flat tokens (B*N, D) in fp32
    """
    def __init__(
        self,
        sim_weight: float = 2.0,
        std_weight: float = 20.0,
        cov_weight: float = 2.0,
    ):
        super().__init__()
        self.sim_weight = sim_weight
        self.std_weight = std_weight
        self.cov_weight = cov_weight

    def forward(
        self,
        z_pred:        torch.Tensor,   # (B, D)   pooled predictor output
        z_tgt:         torch.Tensor,   # (B, D)   pooled target output
        z_pred_tokens: torch.Tensor,   # (B*N, D) flat predictor tokens
        z_tgt_tokens:  torch.Tensor,   # (B*N, D) flat target tokens
    ) -> tuple[torch.Tensor, dict]:

        D = z_pred.shape[1]

        # Invariance on pooled vectors (B, D)
        loss_inv = F.mse_loss(z_pred, z_tgt)

        # Upcast to fp32 before statistical computations
        xf = z_pred_tokens.float()
        yf = z_tgt_tokens.float()

        def variance_loss(z: torch.Tensor) -> torch.Tensor:
            z   = z - z.mean(dim=0, keepdim=True)
            std = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-4)
            return F.relu(1.0 - std).mean()

        # Divide by 2 to normalize across pred+tgt (matches jepa-baseline/loss.py)
        loss_var = (variance_loss(xf) + variance_loss(yf)) / 2.0

        def covariance_loss(z: torch.Tensor) -> torch.Tensor:
            N    = z.shape[0]
            z    = z - z.mean(dim=0, keepdim=True)
            cov  = (z.T @ z) / max(1, N - 1)
            mask = ~torch.eye(D, dtype=torch.bool, device=z.device)
            return cov[mask].pow(2).sum() / D

        loss_cov = covariance_loss(xf) + covariance_loss(yf)

        loss = (
            self.sim_weight * loss_inv +
            self.std_weight * loss_var +
            self.cov_weight * loss_cov
        )

        metrics = {
            "loss_total":      loss.item(),
            "loss_invariance": loss_inv.item(),
            "loss_variance":   loss_var.item(),
            "loss_covariance": loss_cov.item(),
        }
        return loss, metrics


# ─────────────────────────────────────────────
# 7. Full ViT-JEPA Model
# ─────────────────────────────────────────────

class ViTJEPA(nn.Module):
    """
    Full ViT-JEPA v2.

    Training:
      context → Encoder → ctx_tokens (B, N, D)
                        → Predictor → pred_tokens (B, N, D) → pool → z_pred (B, D)
      target  → Encoder (stop-grad) → tgt_tokens (B, N, D) → pool → z_tgt (B, D)
      loss    = VICReg(z_pred, z_tgt, pred_tokens_flat, tgt_tokens_flat)

    Evaluation:
      x → Encoder.forward_pooled() → z (B, D) → linear probe / kNN
    """
    def __init__(
        self,
        in_channels:   int   = 11,
        embed_dim:     int   = 384,
        depth:         int   = 6,
        num_heads:     int   = 6,
        mlp_ratio:     float = 4.0,
        dropout:       float = 0.0,
        img_size:      int   = 224,
        patch_size:    int   = 32,
        tubelet:       int   = 2,
        num_frames:    int   = 16,
        predictor_dim: int   = 192,
        pred_depth:    int   = 2,
        pred_heads:    int   = 4,
        sim_weight:    float = 2.0,
        std_weight:    float = 20.0,
        cov_weight:    float = 2.0,
    ):
        super().__init__()

        self.encoder = ViTEncoder(
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            img_size=img_size,
            patch_size=patch_size,
            tubelet=tubelet,
            num_frames=num_frames,
        )

        self.predictor = TransformerPredictor(
            encoder_dim=embed_dim,
            predictor_dim=predictor_dim,
            depth=pred_depth,
            num_heads=pred_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        self.vicreg = VICRegLoss(
            sim_weight=sim_weight,
            std_weight=std_weight,
            cov_weight=cov_weight,
        )

    def forward(
        self,
        context: torch.Tensor,   # (B, C, T, H, W)
        target:  torch.Tensor,   # (B, C, T, H, W)
    ) -> tuple[torch.Tensor, dict]:
        # Context path — full gradients flow through encoder + predictor
        ctx_tokens  = self.encoder(context)          # (B, N, D)
        pred_tokens = self.predictor(ctx_tokens)     # (B, N, D)
        z_pred      = pred_tokens.mean(dim=1)        # (B, D)

        # Target path — stop-gradient: target is a stable reference
        with torch.no_grad():
            tgt_tokens = self.encoder(target)        # (B, N, D)
        z_tgt = tgt_tokens.mean(dim=1)               # (B, D)

        # Flatten tokens → (B*N, D) for token-level VICReg statistics
        B, N, D      = pred_tokens.shape
        pred_flat    = pred_tokens.reshape(B * N, D)
        tgt_flat     = tgt_tokens.reshape(B * N, D)

        loss, metrics = self.vicreg(z_pred, z_tgt, pred_flat, tgt_flat)
        return loss, metrics

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Frozen encoder embedding for evaluation."""
        self.encoder.eval()
        return self.encoder.forward_pooled(x)

    def count_parameters(self) -> dict:
        enc  = sum(p.numel() for p in self.encoder.parameters()   if p.requires_grad)
        pred = sum(p.numel() for p in self.predictor.parameters() if p.requires_grad)
        return {"encoder": enc, "predictor": pred, "total": enc + pred}


# ─────────────────────────────────────────────
# 8. Sanity Check
# ─────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    model = ViTJEPA().to(device)

    params = model.count_parameters()
    print(f"[PARAMS] encoder:    {params['encoder']:,}")
    print(f"[PARAMS] predictor:  {params['predictor']:,}")
    print(f"[PARAMS] total:      {params['total']:,}")
    print(f"[PARAMS] < 100M:     {params['total'] < 100_000_000}\n")

    B   = 2
    ctx = torch.randn(B, 11, 16, 224, 224, device=device)
    tgt = torch.randn(B, 11, 16, 224, 224, device=device)

    print("[SHAPE] Patch embedding:")
    tokens = model.encoder.patch_embed(ctx)
    print(f"  input  → {ctx.shape}")
    print(f"  output → {tokens.shape}  (expected ({B}, 392, 384))\n")

    print("[SHAPE] Encoder output:")
    enc_out = model.encoder(ctx)
    print(f"  {enc_out.shape}  (expected ({B}, 392, 384))\n")

    print("[SHAPE] Predictor output (token sequence, not pooled):")
    pred_tokens = model.predictor(enc_out)
    print(f"  {pred_tokens.shape}  (expected ({B}, 392, 384))\n")

    print("[FORWARD] Full forward pass (train mode)...")
    model.train()
    loss, metrics = model(ctx, tgt)
    print(f"  loss: {loss.item():.4f}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\n[BACKWARD] Backward pass...")
    loss.backward()
    enc_grad  = any(p.grad is not None for p in model.encoder.parameters())
    pred_grad = any(p.grad is not None for p in model.predictor.parameters())
    print(f"  Encoder has gradients:   {enc_grad}")
    print(f"  Predictor has gradients: {pred_grad}")

    model.eval()
    with torch.no_grad():
        z = model.encode(ctx)
    print(f"\n[EMBED] shape: {z.shape}  (expected ({B}, 384))")
    print(f"[EMBED] mean:  {z.mean():.4f}")
    print(f"[EMBED] std:   {z.std():.4f}")

    print("\n✓ All checks passed!")
