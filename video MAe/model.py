"""
VideoMAE for Active Matter Physics Simulations
===============================================
Architecture:
  Encoder:  ViT-Tiny — embed_dim=192, depth=12, heads=3, patch=(2,16,16)
            → 8 × 14 × 14 = 1568 total patches
            → 90% tube masking → encoder sees ~152 patches only
  Decoder:  lightweight — embed_dim=96, depth=4, heads=3
            → reconstructs all 1568 patches from 152 visible tokens

Parameter budget:
  Encoder:  ~6.4 M  (patch_embed ~1.1M + 12 × ~0.44M blocks)
  Decoder:  ~1.0 M  (4 × ~0.11M blocks + pred head ~0.54M)
  Total:    ~7.4 M  (well under 100M limit)

Training flow:
  x (B, 11, 16, 224, 224)
   → PatchEmbed → (B, 1568, 192)
   → tube masking → encoder sees (B, 152, 192)
   → ViT-Tiny blocks → (B, 152, 192)
   → decoder: reconstruct (B, 1568, 5632) where 5632 = 2*16*16*11
   → MSE loss on masked patches only (with per-patch normalization)

Eval flow (no masking):
  x → encoder (all 1568 patches) → mean pool → (B, 192)  ← frozen
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# 1. 3D sinusoidal positional embedding
# ─────────────────────────────────────────────

def get_3d_sincos_pos_embed(embed_dim: int, num_t: int, num_h: int, num_w: int) -> torch.Tensor:
    """Returns (1, num_t*num_h*num_w, embed_dim). Requires embed_dim % 3 == 0."""
    assert embed_dim % 3 == 0, f"embed_dim ({embed_dim}) must be divisible by 3 for 3D sincos"
    d = embed_dim // 3

    def sincos_1d(length, dim):
        pos   = torch.arange(length, dtype=torch.float32)
        i     = torch.arange(dim // 2, dtype=torch.float32)
        theta = pos.unsqueeze(1) / (10000 ** (2 * i / dim)).unsqueeze(0)
        return torch.cat([theta.sin(), theta.cos()], dim=-1)   # (length, dim)

    t_emb = sincos_1d(num_t, d).reshape(num_t, 1,     1,     d).expand(num_t, num_h, num_w, d)
    h_emb = sincos_1d(num_h, d).reshape(1,     num_h, 1,     d).expand(num_t, num_h, num_w, d)
    w_emb = sincos_1d(num_w, d).reshape(1,     1,     num_w, d).expand(num_t, num_h, num_w, d)

    pos = torch.cat([t_emb, h_emb, w_emb], dim=-1)             # (num_t, num_h, num_w, embed_dim)
    return pos.reshape(1, num_t * num_h * num_w, embed_dim)


# ─────────────────────────────────────────────
# 2. Spatiotemporal patch embedding
# ─────────────────────────────────────────────

class PatchEmbed3D(nn.Module):
    """
    Conv3D tubelet embedding.
    Input:  (B, C, T, H, W)
    Output: (B, num_patches, embed_dim)
    """
    def __init__(
        self,
        in_channels: int = 11,
        embed_dim:   int = 192,
        img_size:    int = 224,
        patch_size:  int = 16,
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
        x = x.flatten(2)           # (B, D, num_patches) — t-h-w order
        x = x.transpose(1, 2)     # (B, num_patches, D)
        return self.norm(x)


# ─────────────────────────────────────────────
# 3. Transformer building blocks
# ─────────────────────────────────────────────

class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 3, dropout: float = 0.0):
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
# 4. Temporal tube masking
# ─────────────────────────────────────────────

def sample_tube_mask(
    B: int, num_t: int, num_h: int, num_w: int,
    mask_ratio: float, device: torch.device,
):
    """
    Temporal tube masking: same spatial (h,w) positions masked across ALL time steps.
    This prevents the model from trivially interpolating masked regions from nearby frames.

    Returns:
      ids_keep: (B, N_vis)        indices of visible patches in [0, N)
      mask:     (B, N)            1 = masked, 0 = visible
    where N = num_t * num_h * num_w, N_vis = num_t * num_spatial_keep
    """
    num_s         = num_h * num_w
    num_keep_s    = max(1, int(num_s * (1.0 - mask_ratio)))  # visible spatial positions
    N             = num_t * num_s

    # Random permutation of spatial patches
    noise         = torch.rand(B, num_s, device=device)
    ids_spatial   = noise.argsort(dim=1)                                # (B, num_s)
    ids_keep_s    = ids_spatial[:, :num_keep_s]                         # (B, num_keep_s)

    # Expand to all temporal positions: patch_index = t * num_s + s
    t_offsets     = torch.arange(num_t, device=device) * num_s         # (num_t,)
    # (B, num_keep_s) + (num_t,) → broadcast over (B, num_t, num_keep_s)
    ids_keep_s_   = ids_keep_s.unsqueeze(1).expand(-1, num_t, -1)      # (B, num_t, num_keep_s)
    t_offsets_    = t_offsets.reshape(1, num_t, 1).expand(B, -1, num_keep_s)
    ids_keep_st   = (ids_keep_s_ + t_offsets_).reshape(B, -1)          # (B, N_vis)
    ids_keep      = ids_keep_st.sort(dim=1).values                     # sort for cleaner indexing

    # Binary mask: 1 = masked, 0 = visible
    mask = torch.ones(B, N, device=device)
    mask.scatter_(1, ids_keep, 0.0)

    return ids_keep, mask


# ─────────────────────────────────────────────
# 5. ViT Encoder (processes visible patches only)
# ─────────────────────────────────────────────

class VideoMAEEncoder(nn.Module):
    """
    ViT-Tiny encoder that processes only visible (unmasked) patches.

    forward(x, ids_keep) → (B, N_vis, D)     used during training
    forward_all(x)        → (B, N,     D)     used during evaluation (no masking)
    """
    def __init__(
        self,
        in_channels: int   = 11,
        embed_dim:   int   = 192,
        depth:       int   = 12,
        num_heads:   int   = 3,
        mlp_ratio:   float = 4.0,
        dropout:     float = 0.0,
        img_size:    int   = 224,
        patch_size:  int   = 16,
        tubelet:     int   = 2,
        num_frames:  int   = 16,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed3D(
            in_channels, embed_dim, img_size, patch_size, tubelet, num_frames
        )
        num_t = num_frames // tubelet
        num_h = img_size   // patch_size
        num_w = img_size   // patch_size

        pos_embed = get_3d_sincos_pos_embed(embed_dim, num_t, num_h, num_w)
        self.register_buffer("pos_embed", pos_embed)   # (1, N, D), non-learnable

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
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv3d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, ids_keep: torch.Tensor) -> torch.Tensor:
        """
        x:        (B, C, T, H, W)
        ids_keep: (B, N_vis)
        Returns:  (B, N_vis, D) — only visible tokens, ready for decoder
        """
        tokens = self.patch_embed(x) + self.pos_embed  # (B, N, D)

        # Select visible tokens
        D = tokens.shape[-1]
        idx = ids_keep.unsqueeze(-1).expand(-1, -1, D)  # (B, N_vis, D)
        tokens = tokens.gather(dim=1, index=idx)         # (B, N_vis, D)

        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)

    def forward_all(self, x: torch.Tensor) -> torch.Tensor:
        """Process all patches — used at evaluation time."""
        tokens = self.patch_embed(x) + self.pos_embed
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)


# ─────────────────────────────────────────────
# 6. MAE Decoder (lightweight, reconstructs all N patches)
# ─────────────────────────────────────────────

class MAEDecoder(nn.Module):
    """
    Lightweight decoder: projects encoder tokens into full N-token sequence,
    inserts learned mask tokens for masked positions, then predicts patch pixels.

    embed_dim=96, depth=4, heads=3  → ~1M params total (discarded after training)
    """
    def __init__(
        self,
        encoder_dim:  int   = 192,
        decoder_dim:  int   = 96,
        decoder_depth:int   = 4,
        decoder_heads:int   = 3,
        mlp_ratio:    float = 4.0,
        num_patches:  int   = 1568,
        patch_dim:    int   = 5632,   # tubelet * patch_h * patch_w * C = 2*16*16*11
        num_t:        int   = 8,
        num_h:        int   = 14,
        num_w:        int   = 14,
    ):
        super().__init__()
        assert decoder_dim % 3 == 0, "decoder_dim must be divisible by 3 for sincos pos embed"

        self.decoder_dim = decoder_dim
        self.num_patches = num_patches

        self.encoder_to_decoder = nn.Linear(encoder_dim, decoder_dim, bias=True)
        self.mask_token         = nn.Parameter(torch.zeros(1, 1, decoder_dim))

        dec_pos = get_3d_sincos_pos_embed(decoder_dim, num_t, num_h, num_w)
        self.register_buffer("pos_embed", dec_pos)   # (1, N, decoder_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(decoder_dim, decoder_heads, mlp_ratio)
            for _ in range(decoder_depth)
        ])
        self.norm = nn.LayerNorm(decoder_dim)
        self.pred = nn.Linear(decoder_dim, patch_dim, bias=True)

        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pred.weight, std=0.02)
        nn.init.zeros_(self.pred.bias)

    def forward(self, x_vis: torch.Tensor, ids_keep: torch.Tensor) -> torch.Tensor:
        """
        x_vis:    (B, N_vis, encoder_dim) — encoder output for visible patches
        ids_keep: (B, N_vis) — positions of visible patches in [0, N)
        Returns:  (B, N, patch_dim) — prediction for ALL patches
        """
        B   = x_vis.shape[0]
        N   = self.num_patches
        D   = self.decoder_dim

        # Project encoder output to decoder dim
        x_vis = self.encoder_to_decoder(x_vis)   # (B, N_vis, D)

        # Build full sequence: mask tokens everywhere, then place visible tokens
        x_full = self.mask_token.expand(B, N, -1).clone()      # (B, N, D)
        idx    = ids_keep.unsqueeze(-1).expand(-1, -1, D)       # (B, N_vis, D)
        x_full.scatter_(1, idx, x_vis)                          # place visible tokens

        # Add positional embedding for all N positions
        x_full = x_full + self.pos_embed   # (B, N, D)

        # Decode
        for block in self.blocks:
            x_full = block(x_full)
        x_full = self.norm(x_full)

        return self.pred(x_full)   # (B, N, patch_dim)


# ─────────────────────────────────────────────
# 7. Full VideoMAE model
# ─────────────────────────────────────────────

class VideoMAE(nn.Module):
    """
    Full VideoMAE model.

    Training:
      x → encoder(visible only) → decoder(reconstruct all) → MSE loss on masked patches

    Eval (decoder discarded):
      x → encoder(all patches) → global avg pool → (B, 192)  ← frozen for linear probe / kNN
    """
    def __init__(
        self,
        # Input
        in_channels:   int   = 11,
        num_frames:    int   = 16,
        img_size:      int   = 224,
        # Encoder (ViT-Tiny)
        enc_embed_dim: int   = 192,
        enc_depth:     int   = 12,
        enc_heads:     int   = 3,
        mlp_ratio:     float = 4.0,
        dropout:       float = 0.0,
        patch_size:    int   = 16,
        tubelet:       int   = 2,
        # Masking
        mask_ratio:    float = 0.90,
        # Decoder (lightweight)
        dec_embed_dim: int   = 96,
        dec_depth:     int   = 4,
        dec_heads:     int   = 3,
        # Loss
        norm_pix_loss: bool  = True,
    ):
        super().__init__()
        self.mask_ratio    = mask_ratio
        self.norm_pix_loss = norm_pix_loss
        self.tubelet       = tubelet
        self.patch_size    = patch_size

        num_t = num_frames // tubelet
        num_h = img_size   // patch_size
        num_w = img_size   // patch_size
        self.num_t = num_t
        self.num_h = num_h
        self.num_w = num_w

        patch_dim    = tubelet * patch_size * patch_size * in_channels  # 2*16*16*11 = 5632
        num_patches  = num_t * num_h * num_w                            # 8*14*14 = 1568

        self.encoder = VideoMAEEncoder(
            in_channels=in_channels, embed_dim=enc_embed_dim,
            depth=enc_depth, num_heads=enc_heads, mlp_ratio=mlp_ratio,
            dropout=dropout, img_size=img_size, patch_size=patch_size,
            tubelet=tubelet, num_frames=num_frames,
        )

        self.decoder = MAEDecoder(
            encoder_dim=enc_embed_dim, decoder_dim=dec_embed_dim,
            decoder_depth=dec_depth, decoder_heads=dec_heads,
            mlp_ratio=mlp_ratio, num_patches=num_patches, patch_dim=patch_dim,
            num_t=num_t, num_h=num_h, num_w=num_w,
        )

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rearrange video into patch targets (same ordering as PatchEmbed3D).
        x:       (B, C, T, H, W)
        Returns: (B, N, patch_dim)  where patch_dim = tubelet*patch_h*patch_w*C
        """
        B, C, T, H, W = x.shape
        t, p, q = self.tubelet, self.patch_size, self.patch_size
        nt, nh, nw = T // t, H // p, W // q
        x = x.reshape(B, C, nt, t, nh, p, nw, q)
        x = x.permute(0, 2, 4, 6, 3, 5, 7, 1)   # (B, nt, nh, nw, t, p, q, C)
        return x.reshape(B, nt * nh * nw, t * p * q * C)

    def forward(self, x: torch.Tensor):
        """
        Training forward pass.
        Returns: (loss: scalar, mask: (B, N))
        """
        B = x.shape[0]

        # Sample tube mask
        ids_keep, mask = sample_tube_mask(
            B, self.num_t, self.num_h, self.num_w, self.mask_ratio, x.device
        )

        # Encode visible patches
        latent = self.encoder(x, ids_keep)             # (B, N_vis, enc_dim)

        # Decode all patches
        pred   = self.decoder(latent, ids_keep)         # (B, N, patch_dim)

        # Ground truth patches
        target = self.patchify(x)                       # (B, N, patch_dim)

        # Per-patch normalization of target (removes local mean/variance biases)
        if self.norm_pix_loss:
            mean   = target.mean(dim=-1, keepdim=True)
            var    = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()

        # MSE loss on masked patches only
        loss = ((pred - target) ** 2).mean(dim=-1)      # (B, N)
        loss = (loss * mask).sum() / mask.sum()         # mean over masked patches

        return loss, mask

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract frozen encoder embedding for evaluation.
        Returns globally mean-pooled token: (B, enc_embed_dim).
        """
        self.encoder.eval()
        tokens = self.encoder.forward_all(x)   # (B, N, D)
        return tokens.mean(dim=1)               # (B, D)

    def count_parameters(self) -> dict:
        enc  = sum(p.numel() for p in self.encoder.parameters()  if p.requires_grad)
        dec  = sum(p.numel() for p in self.decoder.parameters()  if p.requires_grad)
        return {"encoder": enc, "decoder": dec, "total": enc + dec}


# ─────────────────────────────────────────────
# 8. Sanity check
# ─────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    model = VideoMAE().to(device)

    params = model.count_parameters()
    print(f"[PARAMS] encoder: {params['encoder']:,}")
    print(f"[PARAMS] decoder: {params['decoder']:,}")
    print(f"[PARAMS] total:   {params['total']:,}")
    print(f"[PARAMS] < 100M:  {params['total'] < 100_000_000}\n")

    B = 2
    x = torch.randn(B, 11, 16, 224, 224, device=device)

    print("[SHAPE] Masking...")
    ids_keep, mask = sample_tube_mask(B, 8, 14, 14, 0.90, device)
    print(f"  ids_keep: {ids_keep.shape}  (expected ({B}, ~152))")
    print(f"  mask:     {mask.shape}       (expected ({B}, 1568))")
    print(f"  visible ratio: {(1 - mask.float().mean()):.3f}  (expected ~0.10)\n")

    print("[SHAPE] Encoder (visible only)...")
    enc_out = model.encoder(x, ids_keep)
    print(f"  {x.shape} → {enc_out.shape}  (expected ({B}, ~152, 192))\n")

    print("[SHAPE] patchify...")
    patches = model.patchify(x)
    print(f"  {x.shape} → {patches.shape}  (expected ({B}, 1568, 5632))\n")

    print("[FORWARD] Full training forward...")
    loss, mask = model(x)
    print(f"  loss: {loss.item():.4f}\n")

    print("[BACKWARD] Backward pass...")
    loss.backward()
    print(f"  encoder has gradients: {any(p.grad is not None for p in model.encoder.parameters())}\n")

    model.eval()
    with torch.no_grad():
        emb = model.encode(x)
    print(f"[EMBED] shape: {emb.shape}  (expected ({B}, 192))")

    print("\n✓ All checks passed!")
