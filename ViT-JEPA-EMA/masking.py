"""
masking.py  —  Multi-block masking strategy for I-JEPA
=======================================================
Directly implements the masking strategy from the I-JEPA paper
(Assran et al., 2023), adapted for spatiotemporal data.

Key design choices from Table 6 of the paper:
  - 4 target blocks, scale (0.15, 0.2), aspect ratio (0.75, 1.5)
  - 1 context block, scale (0.85, 1.0)
  - Remove target regions from context (non-trivial prediction task)

We apply spatial masking independently per frame, keeping the
same spatial mask pattern across all frames (temporal tube masking).
This is consistent with VideoMAE and the baseline paper.
"""

import math
import random
import torch


def sample_block_mask(
    h_patches: int,
    w_patches: int,
    scale_range: tuple,
    aspect_ratio_range: tuple,
    num_frames: int,
) -> list:
    """
    Sample a spatial block mask and extend it across all frames.
    Returns list of patch indices (in the flattened T*H*W space).

    The block is defined in 2D spatial space and replicated
    across all time steps (temporal tube masking).
    """
    N_spatial = h_patches * w_patches

    for _ in range(100):  # retry if block is invalid
        # Sample block size
        scale  = random.uniform(*scale_range)
        ratio  = random.uniform(*aspect_ratio_range)
        area   = int(N_spatial * scale)
        h_blk  = max(1, int(math.sqrt(area * ratio)))
        w_blk  = max(1, int(math.sqrt(area / ratio)))
        h_blk  = min(h_blk, h_patches)
        w_blk  = min(w_blk, w_patches)

        # Sample top-left corner
        top  = random.randint(0, h_patches - h_blk)
        left = random.randint(0, w_patches - w_blk)

        # Build spatial patch indices for this block
        spatial_ids = []
        for r in range(top, top + h_blk):
            for c in range(left, left + w_blk):
                spatial_ids.append(r * w_patches + c)

        if len(spatial_ids) == 0:
            continue

        # Extend to all frames: token index = frame * N_spatial + spatial_id
        all_ids = []
        for t in range(num_frames):
            for sid in spatial_ids:
                all_ids.append(t * N_spatial + sid)

        return all_ids

    # Fallback: return a minimal single patch
    return list(range(num_frames))


class MultiBlockMaskSampler:
    """
    Samples I-JEPA multi-block masks for a batch.

    Returns per-sample:
      - context_ids  : indices of context patches (sorted)
      - target_ids   : indices of target patches (union of M blocks)
      - context_mask : bool tensor of shape (N_total,)

    Paper defaults (Table 6, best configuration):
      num_target_blocks = 4
      target_scale      = (0.15, 0.2)
      target_ratio      = (0.75, 1.5)
      context_scale     = (0.85, 1.0)
    """
    def __init__(
        self,
        h_patches: int = 14,
        w_patches: int = 14,
        num_frames: int = 8,
        num_target_blocks: int = 4,
        target_scale: tuple = (0.15, 0.2),
        target_ratio: tuple = (0.75, 1.5),
        context_scale: tuple = (0.85, 1.0),
    ):
        self.h_patches         = h_patches
        self.w_patches         = w_patches
        self.num_frames        = num_frames
        self.num_target_blocks = num_target_blocks
        self.target_scale      = target_scale
        self.target_ratio      = target_ratio
        self.context_scale     = context_scale
        self.N_total           = h_patches * w_patches * num_frames

    def sample_one(self, device):
        """Sample masks for a single sample."""
        N_total = self.N_total

        # ── Target blocks ──────────────────────────────────────────────
        target_set = set()
        for _ in range(self.num_target_blocks):
            ids = sample_block_mask(
                self.h_patches, self.w_patches,
                self.target_scale, self.target_ratio,
                self.num_frames,
            )
            target_set.update(ids)
        target_ids = sorted(target_set)

        # ── Context block ──────────────────────────────────────────────
        context_all_ids = sample_block_mask(
            self.h_patches, self.w_patches,
            self.context_scale, (1.0, 1.0),   # unit aspect ratio for context
            self.num_frames,
        )
        # Remove any overlap with target patches
        target_set_fast = set(target_ids)
        context_ids = [i for i in context_all_ids if i not in target_set_fast]

        # Ensure at least a few context patches remain
        if len(context_ids) < 4:
            # Fallback: use all non-target patches
            context_ids = [i for i in range(N_total) if i not in target_set_fast]

        target_ids  = sorted(target_ids)
        context_ids = sorted(context_ids)

        return context_ids, target_ids

    def __call__(self, batch_size: int, device: torch.device):
        """
        Sample masks for a full batch.
        To ensure tensor dimensions match for batching, we sample ONE mask
        and apply it across the entire batch.
        """
        # 1. Sample exactly once so N_ctx and N_tgt are constant
        ctx_ids, tgt_ids = self.sample_one(device)

        # 2. Expand across the batch dimension
        ctx_tensor = torch.tensor(ctx_ids, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
        tgt_tensor = torch.tensor(tgt_ids, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # 3. Create the boolean mask for the context encoder
        ctx_mask = torch.zeros(batch_size, self.N_total, dtype=torch.bool, device=device)
        ctx_mask[:, ctx_ids] = True

        return {
            'context_ids':  ctx_tensor,   # (B, N_ctx)
            'target_ids':   tgt_tensor,   # (B, N_tgt)
            'context_mask': ctx_mask,     # (B, N_total) bool
        }
