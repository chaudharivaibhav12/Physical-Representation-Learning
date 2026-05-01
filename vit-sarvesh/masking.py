"""
Block masking for ViT-JEPA.

Token grid: (num_t=8, num_h=14, num_w=14) = 1568 tokens total.
Target tokens form 1-4 contiguous 3D blocks (~25% of tokens).
Context tokens are everything else (~75%).
"""

import random
import torch


def sample_block_mask(
    num_t: int,
    num_h: int,
    num_w: int,
    target_ratio: float = 0.25,
    num_blocks: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample context and target token indices using spatiotemporal block masking.

    Args:
        num_t, num_h, num_w: token grid dimensions (e.g. 8, 14, 14)
        target_ratio: fraction of tokens to mask as target (~0.25)
        num_blocks: number of contiguous target blocks

    Returns:
        context_idx: 1D LongTensor of context token indices
        target_idx:  1D LongTensor of target token indices
    """
    total = num_t * num_h * num_w
    target_mask = torch.zeros(total, dtype=torch.bool)

    per_block = target_ratio / num_blocks  # fraction per block

    for _ in range(num_blocks):
        scale = random.uniform(0.5, 1.5)
        n     = max(1, int(total * per_block * scale))
        side  = max(1, round(n ** (1.0 / 3)))

        dt = random.randint(max(1, side - 1), min(num_t, side + 2))
        dh = random.randint(max(1, side - 1), min(num_h, side + 2))
        dw = max(1, n // (dt * dh))
        dw = min(dw, num_w)

        t0 = random.randint(0, max(0, num_t - dt))
        h0 = random.randint(0, max(0, num_h - dh))
        w0 = random.randint(0, max(0, num_w - dw))

        t_idx = torch.arange(t0, min(t0 + dt, num_t))
        h_idx = torch.arange(h0, min(h0 + dh, num_h))
        w_idx = torch.arange(w0, min(w0 + dw, num_w))

        flat = (
            t_idx[:, None, None] * (num_h * num_w) +
            h_idx[None, :, None] * num_w +
            w_idx[None, None, :]
        ).reshape(-1)
        target_mask[flat] = True

    all_idx     = torch.arange(total)
    target_idx  = all_idx[target_mask]
    context_idx = all_idx[~target_mask]
    return context_idx, target_idx


if __name__ == "__main__":
    ctx, tgt = sample_block_mask(8, 14, 14)
    total = 8 * 14 * 14
    print(f"Total: {total}  Context: {len(ctx)} ({len(ctx)/total:.1%})  Target: {len(tgt)} ({len(tgt)/total:.1%})")
