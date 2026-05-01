"""
ViT-JEPA model for active_matter physical simulation data.

Components:
  PatchEmbed3D    — spatiotemporal tubelet tokenizer (t=2, h=16, w=16)
  TransformerBlock — pre-norm attention + MLP block
  ViTEncoder      — stack of TransformerBlocks + LayerNorm (no pos embed)
  Predictor       — narrow transformer: embed_dim → pred_dim → embed_dim
  ViTJEPA         — full model: patch_embed + pos_embed + online/target encoders
                    + mask_token + predictor

Token count: (16/2) × (224/16) × (224/16) = 8 × 14 × 14 = 1568

Training forward pass:
  frames (B, 11, 16, 224, 224)
    → patch_embed + pos_embed → all_tokens (B, 1568, D)
    → online_encoder on context_idx tokens (B, N_ctx, D)
    → predictor on [ctx_encoded | mask_tokens@target_pos] → pred (B, N_tgt, D)
    → MSE vs F.normalize(target_encoder(all_tokens)[:, target_idx])
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ─────────────────────────────────────────────────────────────────────────────
# Patch Embedding
# ─────────────────────────────────────────────────────────────────────────────

class PatchEmbed3D(nn.Module):
    """
    Spatiotemporal tubelet embedding via Conv3D.

    Input:  (B, C, T, H, W)
    Output: (B, num_patches, embed_dim)   num_patches = (T/t) * (H/h) * (W/w)
    """
    def __init__(
        self,
        in_channels: int = 11,
        embed_dim:   int = 256,
        img_size:    int = 224,
        patch_size:  int = 16,
        tubelet:     int = 2,
        num_frames:  int = 16,
    ):
        super().__init__()
        self.num_t = num_frames // tubelet    # 8
        self.num_h = img_size   // patch_size  # 14
        self.num_w = img_size   // patch_size  # 14
        self.num_patches = self.num_t * self.num_h * self.num_w  # 1568

        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=(tubelet, patch_size, patch_size),
            stride=(tubelet, patch_size, patch_size),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)           # (B, D, num_t, num_h, num_w)
        x = x.flatten(2)           # (B, D, num_patches)
        x = x.transpose(1, 2)      # (B, num_patches, D)
        return self.norm(x)


# ─────────────────────────────────────────────────────────────────────────────
# Transformer Building Blocks
# ─────────────────────────────────────────────────────────────────────────────

class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.attn_drop = dropout
        self.qkv  = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop if self.training else 0.0)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = MLP(dim, mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# ViT Encoder (shared architecture for online and target)
# ─────────────────────────────────────────────────────────────────────────────

class ViTEncoder(nn.Module):
    """
    Transformer blocks + final LayerNorm.
    Does NOT include patch embedding or positional embedding — those live in ViTJEPA.
    Accepts any token sequence; sequence length is flexible.
    """
    def __init__(self, embed_dim: int, depth: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
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

    def forward(self, x: torch.Tensor, use_checkpoint: bool = False) -> torch.Tensor:
        for block in self.blocks:
            if use_checkpoint:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.norm(x)


# ─────────────────────────────────────────────────────────────────────────────
# Predictor
# ─────────────────────────────────────────────────────────────────────────────

class Predictor(nn.Module):
    """
    Narrow transformer predictor.
    Projects embed_dim → pred_dim, runs depth transformer blocks, projects back.

    Input:  (B, N_ctx + N_tgt, embed_dim)  — context encodings + mask tokens
    Output: (B, N_ctx + N_tgt, embed_dim)  — predictions at every position
    Loss is computed only on the last N_tgt positions (target slots).
    """
    def __init__(
        self,
        embed_dim: int = 256,
        pred_dim:  int = 128,
        depth:     int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.input_proj  = nn.Linear(embed_dim, pred_dim)
        self.blocks      = nn.ModuleList([
            TransformerBlock(pred_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm        = nn.LayerNorm(pred_dim)
        self.output_proj = nn.Linear(pred_dim, embed_dim)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.output_proj(x)


# ─────────────────────────────────────────────────────────────────────────────
# Full ViT-JEPA Model
# ─────────────────────────────────────────────────────────────────────────────

class ViTJEPA(nn.Module):
    """
    ViT-JEPA: Joint-Embedding Predictive Architecture.

    Training:
      1. Tokenize full clip → add pos_embed.
      2. Online encoder processes context tokens only (backprop).
      3. EMA target encoder processes all tokens (no backprop).
      4. Predictor takes context encodings + mask_token@target_pos.
      5. MSE loss vs F.normalize(target_encoder output at target positions).

    Evaluation:
      encode(x) runs online encoder on all 1568 tokens and mean-pools → (B, D).
    """
    def __init__(
        self,
        in_channels: int   = 11,
        embed_dim:   int   = 256,
        depth:       int   = 6,
        num_heads:   int   = 8,
        mlp_ratio:   float = 4.0,
        img_size:    int   = 224,
        patch_size:  int   = 16,
        tubelet:     int   = 2,
        num_frames:  int   = 16,
        pred_dim:    int   = 128,
        pred_depth:  int   = 4,
        pred_heads:  int   = 4,
    ):
        super().__init__()

        self.patch_embed = PatchEmbed3D(
            in_channels=in_channels,
            embed_dim=embed_dim,
            img_size=img_size,
            patch_size=patch_size,
            tubelet=tubelet,
            num_frames=num_frames,
        )
        N = self.patch_embed.num_patches  # 1568
        self.pos_embed  = nn.Parameter(torch.zeros(1, N, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.online_encoder = ViTEncoder(embed_dim, depth, num_heads, mlp_ratio)
        self.target_encoder = ViTEncoder(embed_dim, depth, num_heads, mlp_ratio)
        self.predictor      = Predictor(embed_dim, pred_dim, pred_depth, pred_heads, mlp_ratio)

        # Target encoder starts as a copy of online; never updated by backprop
        self.target_encoder.load_state_dict(self.online_encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        nn.init.trunc_normal_(self.pos_embed,  std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(
        self,
        frames:      torch.Tensor,  # (B, C, T, H, W)
        context_idx: torch.Tensor,  # 1D LongTensor
        target_idx:  torch.Tensor,  # 1D LongTensor
    ) -> tuple[torch.Tensor, dict]:
        B      = frames.shape[0]
        N_ctx  = context_idx.shape[0]
        N_tgt  = target_idx.shape[0]

        # Tokenize + add positional embeddings
        tokens = self.patch_embed(frames) + self.pos_embed  # (B, N, D)

        # Online encoder: context tokens only
        ctx_tokens  = tokens[:, context_idx]  # (B, N_ctx, D)
        ctx_encoded = self.online_encoder(ctx_tokens, use_checkpoint=self.training)  # (B, N_ctx, D)

        # EMA target encoder: all tokens, no grad
        with torch.no_grad():
            tgt_all     = self.target_encoder(tokens)                # (B, N, D)
            tgt_encoded = tgt_all[:, target_idx]                     # (B, N_tgt, D)
            tgt_encoded = F.normalize(tgt_encoded, dim=-1)           # stable training signal

        # Predictor: context encodings + mask tokens at target positions
        mask_tokens = self.mask_token.expand(B, N_tgt, -1) + self.pos_embed[:, target_idx]
        pred_input  = torch.cat([ctx_encoded, mask_tokens], dim=1)   # (B, N_ctx+N_tgt, D)
        pred_output = self.predictor(pred_input)                      # (B, N_ctx+N_tgt, D)
        pred_tgt    = pred_output[:, N_ctx:]                         # (B, N_tgt, D)

        loss = F.mse_loss(pred_tgt, tgt_encoded)

        with torch.no_grad():
            emb_std = ctx_encoded.std(dim=0).mean().item()

        return loss, {"loss": loss.item(), "embedding_std": emb_std}

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Full-clip encoding for evaluation. Online encoder on all tokens → mean pool."""
        tokens = self.patch_embed(x) + self.pos_embed  # (B, N, D)
        tokens = self.online_encoder(tokens)            # (B, N, D)
        return tokens.mean(dim=1)                       # (B, D)

    def count_parameters(self) -> dict:
        embed = (
            sum(p.numel() for p in self.patch_embed.parameters() if p.requires_grad)
            + self.pos_embed.numel()
            + self.mask_token.numel()
        )
        enc  = sum(p.numel() for p in self.online_encoder.parameters() if p.requires_grad)
        pred = sum(p.numel() for p in self.predictor.parameters() if p.requires_grad)
        return {"patch_embed": embed, "encoder": enc, "predictor": pred, "total": embed + enc + pred}


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from masking import sample_block_mask

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = ViTJEPA().to(device)

    params = model.count_parameters()
    print(f"patch_embed: {params['patch_embed']:,}")
    print(f"encoder:     {params['encoder']:,}")
    print(f"predictor:   {params['predictor']:,}")
    print(f"total:       {params['total']:,}  (< 100M: {params['total'] < 100_000_000})")

    B = 2
    x = torch.randn(B, 11, 16, 224, 224, device=device)
    ctx_idx, tgt_idx = sample_block_mask(8, 14, 14)
    ctx_idx, tgt_idx = ctx_idx.to(device), tgt_idx.to(device)

    loss, metrics = model(x, ctx_idx, tgt_idx)
    print(f"\nloss: {loss.item():.4f}  emb_std: {metrics['embedding_std']:.4f}")
    loss.backward()
    print("backward: ok")

    model.eval()
    with torch.no_grad():
        z = model.encode(x)
    print(f"encode: {z.shape}  (expected ({B}, 256))")
    print("all checks passed")
