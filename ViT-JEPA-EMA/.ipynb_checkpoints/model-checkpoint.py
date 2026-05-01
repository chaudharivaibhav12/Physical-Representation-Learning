"""
model.py  —  I-JEPA for spatiotemporal physical simulations
============================================================
Architecture follows the original I-JEPA paper (Assran et al., 2023)
adapted for active matter data: (B, 11, T, 224, 224).

Key design choices (directly from the paper):
  - Context encoder (ViT-S) sees only visible patches
  - Target encoder = EMA of context encoder (no gradient)
  - Narrow predictor ViT conditioned on positional mask tokens
  - Loss: average L2 between predicted and target patch embeddings
  - NO VICReg, NO explicit collapse prevention — EMA handles it
  - Evaluation uses average-pooled TARGET encoder output
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# Patch Embedding  (11-channel spatiotemporal input)
# ─────────────────────────────────────────────────────────────
class PatchEmbed(nn.Module):
    """
    Embed each frame independently using a 2D conv.

    Input : (B, C_in, T, H, W)   C_in=11, T=8, H=W=224
    Output: (B, T*N_patches, embed_dim)   N_patches = (H/P)*(W/P)

    Each frame is treated as a separate 11-channel "image".
    Temporal position is encoded separately via learned embeddings.
    """
    def __init__(self, in_channels=11, embed_dim=384,
                 patch_size=16, img_size=224, num_frames=8):
        super().__init__()
        self.patch_size  = patch_size
        self.num_frames  = num_frames
        self.h_patches   = img_size // patch_size          # 14
        self.w_patches   = img_size // patch_size          # 14
        self.num_patches = self.h_patches * self.w_patches  # 196

        # 2D conv applied independently per frame
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

        # Learned spatial positional embeddings (shared across frames)
        self.spatial_pos = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim))

        # Learned temporal positional embeddings (one per frame)
        self.temporal_pos = nn.Parameter(
            torch.zeros(1, num_frames, embed_dim))

        nn.init.trunc_normal_(self.spatial_pos, std=0.02)
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)

    def forward(self, x):
        """
        x : (B, C, T, H, W)
        returns tokens : (B, T*N, D)
        """
        B, C, T, H, W = x.shape
        assert T == self.num_frames, f"Expected {self.num_frames} frames, got {T}"

        # Process each frame: reshape to (B*T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        x = self.proj(x)                         # (B*T, D, h, w)
        x = x.flatten(2).transpose(1, 2)         # (B*T, N, D)
        x = x.reshape(B, T, self.num_patches, -1)  # (B, T, N, D)

        # Add spatial positional embeddings (broadcast over T)
        x = x + self.spatial_pos.unsqueeze(1)    # (B, T, N, D)

        # Add temporal positional embeddings (broadcast over N)
        x = x + self.temporal_pos.unsqueeze(2)   # (B, T, N, D)

        # Flatten T and N into sequence dimension
        x = x.reshape(B, T * self.num_patches, -1)  # (B, T*N, D)
        return x


# ─────────────────────────────────────────────────────────────
# Standard Transformer Block
# ─────────────────────────────────────────────────────────────
class Attention(nn.Module):
    def __init__(self, dim, num_heads, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.scale     = (dim // num_heads) ** -0.5
        self.qkv       = nn.Linear(dim, dim * 3, bias=True)
        self.proj      = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                   C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0.):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1  = nn.Linear(dim, hidden)
        self.act  = nn.GELU()
        self.fc2  = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = Attention(dim, num_heads, attn_drop, proj_drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = MLP(dim, mlp_ratio, proj_drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ─────────────────────────────────────────────────────────────
# Context Encoder  (ViT-Small)
# ─────────────────────────────────────────────────────────────
class ContextEncoder(nn.Module):
    """
    Standard ViT that processes ONLY the visible (unmasked) patches.
    No CLS token — use average pool of output for downstream eval.

    ViT-Small: embed_dim=384, depth=12, heads=6  → ~22M params
    """
    def __init__(self, embed_dim=384, depth=12, num_heads=6,
                 mlp_ratio=4., in_channels=11, patch_size=16,
                 img_size=224, num_frames=8):
        super().__init__()
        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            img_size=img_size,
            num_frames=num_frames,
        )
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        self.norm      = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x, mask_keep=None):
        """
        x         : (B, C, T, H, W)
        mask_keep : (B, N_total) bool — True = keep this token
                    If None, keep all tokens (target encoder path).
        returns   : (B, N_kept, D)
        """
        tokens = self.patch_embed(x)   # (B, N_total, D)

        if mask_keep is not None:
            # Select only visible tokens per sample
            # mask_keep: (B, N) bool
            B, N, D = tokens.shape
            tokens = tokens[mask_keep].reshape(B, -1, D)

        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        return tokens

    def forward_pooled(self, x):
        """Average-pool over all tokens → (B, D). Used for linear probe."""
        tokens = self.forward(x, mask_keep=None)  # (B, N, D)
        return tokens.mean(dim=1)                  # (B, D)


# ─────────────────────────────────────────────────────────────
# Predictor  (narrow ViT conditioned on positional mask tokens)
# ─────────────────────────────────────────────────────────────
class Predictor(nn.Module):
    """
    Narrow ViT that takes context encoder outputs + positional mask
    tokens and predicts target patch representations.

    Key detail from I-JEPA paper:
      - Predictor embed_dim is SMALLER than encoder (bottleneck)
      - Input projected down to predictor_dim, output projected back up
      - Mask tokens are learnable vectors + positional embeddings
    """
    def __init__(self, encoder_dim=384, predictor_dim=192,
                 depth=6, num_heads=6, mlp_ratio=4.,
                 num_patches_total=1568):
        super().__init__()
        self.predictor_dim = predictor_dim

        # Project context tokens into predictor dimension
        self.input_proj  = nn.Linear(encoder_dim, predictor_dim)

        # Learnable mask token (shared across all masked positions)
        self.mask_token  = nn.Parameter(torch.zeros(1, 1, predictor_dim))

        # Positional embeddings for ALL positions (context + target)
        self.pos_embed   = nn.Parameter(
            torch.zeros(1, num_patches_total, predictor_dim))

        self.blocks = nn.ModuleList([
            Block(predictor_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        self.norm       = nn.LayerNorm(predictor_dim)

        # Project back to encoder dimension for loss computation
        self.output_proj = nn.Linear(predictor_dim, encoder_dim)

        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed,  std=0.02)

    def forward(self, context_tokens, context_ids, target_ids):
            """
            context_tokens : (B, N_ctx, encoder_dim)  — context encoder output
            context_ids    : (B, N_ctx) long           — patch indices of context tokens
            target_ids     : (B, N_tgt) long           — patch indices to predict
    
            Returns predictions : (B, N_tgt, encoder_dim)
            """
            B, N_ctx, _ = context_tokens.shape
            N_tgt = target_ids.shape[1]
    
            # Project context to predictor dim
            ctx = self.input_proj(context_tokens)                  # (B, N_ctx, D_pred)
            
            # 🔥 CRITICAL FIX: Use torch.gather so every video in the batch gets its own unique mask coordinates
            pos_expanded = self.pos_embed.expand(B, -1, -1)
            ctx_pos = torch.gather(pos_expanded, 1, context_ids.unsqueeze(-1).expand(-1, -1, self.predictor_dim))
            ctx = ctx + ctx_pos
    
            # Build mask tokens for target positions
            mask = self.mask_token.expand(B, N_tgt, -1).clone()   # (B, N_tgt, D_pred)
            
            # 🔥 CRITICAL FIX: Gather target positional embeddings per-sample
            tgt_pos = torch.gather(pos_expanded, 1, target_ids.unsqueeze(-1).expand(-1, -1, self.predictor_dim))
            mask = mask + tgt_pos
    
            # Concatenate context + mask tokens
            tokens = torch.cat([ctx, mask], dim=1)                 # (B, N_ctx+N_tgt, D_pred)
    
            for blk in self.blocks:
                tokens = blk(tokens)
            tokens = self.norm(tokens)
    
            # Extract only the target predictions (last N_tgt tokens)
            pred = tokens[:, N_ctx:, :]                            # (B, N_tgt, D_pred)
            pred = self.output_proj(pred)                          # (B, N_tgt, encoder_dim)
            return pred

# ─────────────────────────────────────────────────────────────
# I-JEPA  (full model)
# ─────────────────────────────────────────────────────────────
class IJEPA(nn.Module):
    """
    Image-based Joint-Embedding Predictive Architecture
    adapted for spatiotemporal physical simulations.

    Training:
      1. Target encoder (EMA) encodes full input → all patch representations
      2. Context encoder encodes only visible patches
      3. Predictor maps context + positional mask tokens → target patches
      4. Loss = average L2 between predictions and target encoder outputs

    Evaluation:
      - Use target_encoder.forward_pooled(x) → (B, D) for linear probe / kNN
      - Target encoder has richer representations than context encoder
        because it sees the full input (confirmed in I-JEPA paper)
    """
    def __init__(
        self,
        in_channels=11,
        img_size=224,
        patch_size=16,
        num_frames=8,
        # Context / target encoder (ViT-Small)
        encoder_dim=384,
        encoder_depth=12,
        encoder_heads=6,
        # Predictor (narrow ViT)
        predictor_dim=192,
        predictor_depth=6,
        predictor_heads=6,
        # EMA
        ema_momentum=0.996,
    ):
        super().__init__()

        h_patches = img_size // patch_size
        w_patches = img_size // patch_size
        num_patches_spatial = h_patches * w_patches      # 196
        num_patches_total   = num_frames * num_patches_spatial  # 1568

        self.num_frames           = num_frames
        self.num_patches_spatial  = num_patches_spatial
        self.num_patches_total    = num_patches_total
        self.encoder_dim          = encoder_dim
        self.ema_momentum         = ema_momentum

        # Context encoder (trained via backprop)
        self.context_encoder = ContextEncoder(
            embed_dim=encoder_dim,
            depth=encoder_depth,
            num_heads=encoder_heads,
            in_channels=in_channels,
            patch_size=patch_size,
            img_size=img_size,
            num_frames=num_frames,
        )

        # Target encoder (EMA — no gradient)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # Predictor
        self.predictor = Predictor(
            encoder_dim=encoder_dim,
            predictor_dim=predictor_dim,
            depth=predictor_depth,
            num_heads=predictor_heads,
            num_patches_total=num_patches_total,
        )

    @torch.no_grad()
    def update_ema(self, momentum=None):
        """Update target encoder as EMA of context encoder."""
        m = momentum if momentum is not None else self.ema_momentum
        for ctx_p, tgt_p in zip(self.context_encoder.parameters(),
                                 self.target_encoder.parameters()):
            tgt_p.data = tgt_p.data * m + ctx_p.data * (1.0 - m)

    def forward(self, x, masks):
        """
        x     : (B, C, T, H, W)
        masks : dict with keys:
                  'context_ids'  (B, N_ctx)  — indices of context patches
                  'target_ids'   (B, N_tgt)  — indices of target patches
                  'context_mask' (B, N_total) bool — True=keep for context encoder

        Returns loss (scalar), metrics (dict)
        """
        context_ids  = masks['context_ids']    # (B, N_ctx)
        target_ids   = masks['target_ids']     # (B, N_tgt)
        context_mask = masks['context_mask']   # (B, N_total) bool

        # ── Target encoder: full input → all patch representations ───────
        with torch.no_grad():
            all_target_tokens = self.target_encoder(x, mask_keep=None)  # (B, N_total, D)
            # Extract only the target patch representations
            B, N_total, D = all_target_tokens.shape
            # Gather target tokens using target_ids
            target_tokens = torch.gather(
                all_target_tokens, 1,
                target_ids.unsqueeze(-1).expand(-1, -1, D)
            )  # (B, N_tgt, D)

        # ── Context encoder: visible patches only ────────────────────────
        context_tokens = self.context_encoder(x, mask_keep=context_mask)  # (B, N_ctx, D)

        # ── Predictor: context → target predictions ───────────────────────
        predictions = self.predictor(context_tokens, context_ids, target_ids)  # (B, N_tgt, D)

        # ── Loss: average L2 distance ─────────────────────────────────────
        loss = F.mse_loss(predictions, target_tokens)

        # ── Update EMA ────────────────────────────────────────────────────
        self.update_ema()

        metrics = {
            'loss': loss.item(),
            'pred_norm':   predictions.detach().norm(dim=-1).mean().item(),
            'target_norm': target_tokens.detach().norm(dim=-1).mean().item(),
        }

        return loss, metrics

    def encode(self, x):
        """
        Inference: average-pool of TARGET encoder output.
        Use this for linear probing and kNN evaluation.
        """
        with torch.no_grad():
            return self.target_encoder.forward_pooled(x)  # (B, D)
