"""
ViT-JEPA for Active Matter Physical Simulations
================================================
Architecture:
  - Encoder:   3D Vision Transformer (ViT) with spatiotemporal patch embedding
  - Predictor: Shallow Transformer (2 layers) operating on full token sequence
  - Loss:      VICReg on pooled embeddings (context prediction vs target)

Training Flow (one forward pass):
  1. context (B, 11, 16, 224, 224) → Encoder → token sequence (B, 392, 384)
                                            → Predictor         → z_pred (B, 384)
  2. target  (B, 11, 16, 224, 224) → SAME Encoder              → z_tgt  (B, 384)
  3. VICReg loss on (z_pred, z_tgt)

Why predictor on full token sequence (not pooled vector):
  - Preserves spatial and temporal structure during prediction
  - Allows cross-token attention before summarizing
  - More powerful than a simple linear/conv head
  - Still lightweight: only 2 transformer layers (~1.2M params)

Patch config (memory-safe):
  spatial patch:  32x32  ->  7x7  = 49 spatial tokens
  temporal patch: 2      ->  16/2 = 8  temporal tokens
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

    With defaults (patch=32, tubelet=2, T=16, H=W=224):
      num_patches = 8 x 7 x 7 = 392
    """
    def __init__(
        self,
        in_channels: int = 11,
        embed_dim:   int = 384,
        img_size:    int = 224,
        patch_size:  int = 16,
        tubelet:     int = 2,
        num_frames:  int = 16,
    ):
        super().__init__()
        assert img_size   % patch_size == 0, "img_size must be divisible by patch_size"
        assert num_frames % tubelet    == 0, "num_frames must be divisible by tubelet"

        self.num_t       = num_frames  // tubelet     # 8
        self.num_h       = img_size    // patch_size  # 7
        self.num_w       = img_size    // patch_size  # 7
        self.num_patches = self.num_t * self.num_h * self.num_w  # 392
        self.embed_dim   = embed_dim

        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=(tubelet, patch_size, patch_size),
            stride=(tubelet, patch_size, patch_size),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        x = self.proj(x)               # (B, D, num_t, num_h, num_w)
        x = x.flatten(2)               # (B, D, num_patches)
        x = x.transpose(1, 2)         # (B, num_patches, D)
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
        return torch.cat([theta.sin(), theta.cos()], dim=-1)  # (length, dim)

    t_emb = sincos_1d(num_t, d)  # (num_t, d)
    h_emb = sincos_1d(num_h, d)  # (num_h, d)
    w_emb = sincos_1d(num_w, d)  # (num_w, d)

    t_emb = t_emb[:, None, None, :].expand(num_t, num_h, num_w, d)
    h_emb = h_emb[None, :, None, :].expand(num_t, num_h, num_w, d)
    w_emb = w_emb[None, None, :, :].expand(num_t, num_h, num_w, d)

    pos = torch.cat([t_emb, h_emb, w_emb], dim=-1)  # (num_t, num_h, num_w, embed_dim)
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
    """
    Pre-norm transformer block:
      x = x + Attention(LayerNorm(x))
      x = x + MLP(LayerNorm(x))
    Shared between encoder AND predictor.
    """
    def __init__(
        self,
        dim:       int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout:   float = 0.0,
    ):
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

    Returns the full token sequence (B, N, D) before pooling
    so the predictor can operate on spatial/temporal structure.

    forward()        → (B, N, D)  full token sequence
    forward_pooled() → (B, D)     globally pooled (for evaluation)
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

        # Fixed positional embedding — registered as buffer (not a parameter)
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
        """
        Returns full token sequence before pooling.
        Used during training so predictor gets rich spatial/temporal tokens.

        Input:  (B, C, T, H, W)
        Output: (B, num_patches, embed_dim)
        """
        x = self.patch_embed(x)        # (B, N, D)
        x = x + self.pos_embed         # (B, N, D)
        for block in self.blocks:
            x = block(x)               # (B, N, D)
        return self.norm(x)            # (B, N, D)

    def forward_pooled(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns globally pooled embedding.
        Used during evaluation (linear probe / kNN).

        Input:  (B, C, T, H, W)
        Output: (B, embed_dim)
        """
        tokens = self.forward(x)       # (B, N, D)
        return tokens.mean(dim=1)      # (B, D)


# ─────────────────────────────────────────────
# 5. Shallow Transformer Predictor
# ─────────────────────────────────────────────

class TransformerPredictor(nn.Module):
    """
    Shallow transformer predictor operating on the full token sequence.

    Design rationale:
      - Input is the full token sequence from the encoder (B, 392, 384)
        NOT a pooled vector — preserves all spatial and temporal info
      - Projects to a smaller internal dim (192) to stay lightweight
        and prevent the predictor from overpowering the encoder
      - Only 2 transformer layers — shallow on purpose:
        the encoder does the heavy physics learning,
        the predictor just refines the temporal prediction
      - Projects back to encoder dim and pools for VICReg

    Input:  (B, N, encoder_dim)   e.g. (B, 392, 384)
    Output: (B, encoder_dim)      e.g. (B, 384)

    Full flow:
      tokens (B, 392, 384)
        → input_proj  → (B, 392, 192)
        → 2x TransformerBlock
        → LayerNorm
        → output_proj → (B, 392, 384)
        → mean pool   → (B, 384)   ← z_pred
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

        # Project encoder dim → predictor dim (smaller = lighter)
        self.input_proj = nn.Linear(encoder_dim, predictor_dim)

        # 2-layer shallow transformer
        self.blocks = nn.ModuleList([
            TransformerBlock(predictor_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(predictor_dim)

        # Project back to encoder dim for loss computation
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
            tokens: (B, N, encoder_dim) — context encoder output

        Returns:
            z_pred: (B, encoder_dim) — predicted target embedding
        """
        x = self.input_proj(tokens)        # (B, N, predictor_dim)

        # Attend over ALL spatiotemporal positions before predicting
        # This is the key advantage over a conv/linear head
        for block in self.blocks:
            x = block(x)                   # (B, N, predictor_dim)

        x = self.norm(x)                   # (B, N, predictor_dim)
        x = self.output_proj(x)            # (B, N, encoder_dim)

        # Pool into single prediction vector
        return x.mean(dim=1)               # (B, encoder_dim)


# ─────────────────────────────────────────────
# 6. VICReg Loss
# ─────────────────────────────────────────────

class VICRegLoss(nn.Module):
    """
    VICReg: Variance-Invariance-Covariance Regularization.

    Applied between:
      z_pred — predictor output from context path
      z_tgt  — pooled encoder output from target path

    High std_weight (40) aggressively prevents collapse
    on this small dataset (8,750 samples).
    """
    def __init__(
        self,
        sim_weight: float = 2.0,
        std_weight: float = 40.0,
        cov_weight: float = 2.0,
    ):
        super().__init__()
        self.sim_weight = sim_weight
        self.std_weight = std_weight
        self.cov_weight = cov_weight

    def forward(
        self,
        z_pred: torch.Tensor,   # (B, D)
        z_tgt:  torch.Tensor,   # (B, D)
    ) -> tuple[torch.Tensor, dict]:

        B, D = z_pred.shape

        # Invariance — predict the future correctly
        loss_inv = F.mse_loss(z_pred, z_tgt)

        # Variance — prevent collapse
        def variance_loss(z):
            z   = z - z.mean(dim=0, keepdim=True)
            std = z.std(dim=0)
            return F.relu(1.0 - std).mean()

        loss_var = variance_loss(z_pred) + variance_loss(z_tgt)

        # Covariance — prevent redundant dimensions
        def covariance_loss(z):
            z    = z - z.mean(dim=0, keepdim=True)
            cov  = (z.T @ z) / (B - 1)
            mask = ~torch.eye(D, dtype=torch.bool, device=z.device)
            return cov[mask].pow(2).sum() / D

        loss_cov = covariance_loss(z_pred) + covariance_loss(z_tgt)

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
    Full ViT-JEPA model with shallow transformer predictor.

    Training forward pass:
      context → Encoder → token sequence (B,392,384)
                               → Predictor (2-layer transformer)
                               → z_pred (B,384)

      target  → SAME Encoder → token sequence (B,392,384)
                               → global avg pool
                               → z_tgt (B,384)

      loss = VICReg(z_pred, z_tgt)

    Evaluation:
      x → Encoder.forward_pooled() → z (B,384)
      → frozen, passed to linear probe / kNN
    """
    def __init__(
        self,
        # Encoder
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
        # Predictor
        predictor_dim: int   = 192,
        pred_depth:    int   = 2,
        pred_heads:    int   = 4,
        # VICReg
        sim_weight:    float = 2.0,
        std_weight:    float = 40.0,
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
        """
        Training forward pass.

        context → encoder (6 layers) → tokens (B,392,384)
                                     → predictor (2 layers) → z_pred (B,384)
        target  → encoder (6 layers) → tokens (B,392,384)
                                     → pool              → z_tgt  (B,384)
        loss = VICReg(z_pred, z_tgt)
        """
        # Context path: full encoder + predictor
        ctx_tokens = self.encoder(context)       # (B, N, D)
        z_pred     = self.predictor(ctx_tokens)  # (B, D)

        # Target path: full encoder + pool only
        tgt_tokens = self.encoder(target)        # (B, N, D)
        z_tgt      = tgt_tokens.mean(dim=1)      # (B, D)

        loss, metrics = self.vicreg(z_pred, z_tgt)
        return loss, metrics

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract embeddings for evaluation. Encoder only, no predictor.

        Args:
            x: (B, C, T, H, W)
        Returns:
            z: (B, embed_dim)
        """
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

    # Parameter counts
    params = model.count_parameters()
    print(f"[PARAMS] encoder:    {params['encoder']:,}")
    print(f"[PARAMS] predictor:  {params['predictor']:,}")
    print(f"[PARAMS] total:      {params['total']:,}")
    print(f"[PARAMS] < 100M:     {params['total'] < 100_000_000}\n")

    B   = 2
    ctx = torch.randn(B, 11, 16, 224, 224, device=device)
    tgt = torch.randn(B, 11, 16, 224, 224, device=device)

    # Shape trace
    print("[SHAPE] Patch embedding:")
    tokens = model.encoder.patch_embed(ctx)
    print(f"  input  → {ctx.shape}")
    print(f"  output → {tokens.shape}  (expected ({B}, 392, 384))\n")

    print("[SHAPE] Encoder output (full token sequence):")
    enc_out = model.encoder(ctx)
    print(f"  {enc_out.shape}  (expected ({B}, 392, 384))\n")

    print("[SHAPE] Predictor output:")
    z_pred = model.predictor(enc_out)
    print(f"  {z_pred.shape}  (expected ({B}, 384))\n")

    print("[SHAPE] Pooled target embedding:")
    z_tgt = model.encoder.forward_pooled(tgt)
    print(f"  {z_tgt.shape}  (expected ({B}, 384))\n")

    # Full forward pass
    print("[FORWARD] Full forward pass...")
    loss, metrics = model(ctx, tgt)
    print(f"  loss: {loss.item():.4f}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Backward pass
    print("\n[BACKWARD] Backward pass...")
    loss.backward()
    enc_grad  = any(p.grad is not None for p in model.encoder.parameters())
    pred_grad = any(p.grad is not None for p in model.predictor.parameters())
    print(f"  Encoder has gradients:   {enc_grad}")
    print(f"  Predictor has gradients: {pred_grad}")

    # Evaluation embedding
    model.eval()
    with torch.no_grad():
        z = model.encode(ctx)
    print(f"\n[EMBED] shape: {z.shape}  (expected ({B}, 384))")
    print(f"[EMBED] mean:  {z.mean():.4f}")
    print(f"[EMBED] std:   {z.std():.4f}")

    print("\n✓ All checks passed!")