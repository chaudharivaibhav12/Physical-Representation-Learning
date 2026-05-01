"""
train.py  —  I-JEPA pretraining on active_matter
=================================================
Fixed version addressing:
  1. OOM: reduced batch_size, num_frames, patch_size
  2. Slow attention: fewer tokens via larger patches + fewer frames
  3. Frequent checkpointing every CKPT_EVERY steps (not just per epoch)
  4. Atomic checkpoint writes (no corruption on preemption)
  5. Single-GPU (no broken torchrun DDP)
  6. Step timing logged so you can see actual speed
"""

import os
import math
import time
import argparse
import tempfile
import torch
import wandb
from torch.utils.data import DataLoader

from model import IJEPA
from masking import MultiBlockMaskSampler

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import ActiveMatterDataset


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
CFG = {
    # ── paths ──────────────────────────────────
    "data_dir":  "/scratch/ok2287/data/active_matter/data",
    "ckpt_dir":  "/scratch/ok2287/checkpoints/ijepa",

    # ── data ───────────────────────────────────
    # REDUCED: 8→4 frames cuts token count by 4x (1568→784 tokens)
    "num_frames": 4,

    # ── training ───────────────────────────────
    "epochs":        20,
    # REDUCED: 32→8 to avoid OOM with 11-channel 224x224 ViT input
    "batch_size":    16,
    "lr":            1e-3,
    "min_lr":        1e-6,
    "warmup_epochs": 2,
    "weight_decay_start": 0.04,
    "weight_decay_end":   0.40,
    "grad_clip":     1.0,
    # Save checkpoint every N steps (not just per epoch)
    "ckpt_every":    50,

    # ── EMA momentum schedule (0.996 → 1.0) ───
    "ema_start": 0.996,
    "ema_end":   1.000,

    # ── model ──────────────────────────────────
    "in_channels":      11,
    "img_size":         224,
    # INCREASED: 16→32 cuts spatial patches from 196 to 49 per frame
    # Total tokens: 4 frames × 49 patches = 196 (vs 1568 before)
    # Attention cost: 196² vs 1568² = 64x cheaper!
    "patch_size":       32,
    "encoder_dim":      384,
    "encoder_depth":    12,
    "encoder_heads":    6,
    "predictor_dim":    192,
    "predictor_depth":  6,
    "predictor_heads":  6,

    # ── masking (I-JEPA paper Table 6) ────────
    "num_target_blocks": 4,
    "target_scale":      (0.15, 0.2),
    "target_ratio":      (0.75, 1.5),
    "context_scale":     (0.85, 1.0),

    # ── misc ───────────────────────────────────
    "num_workers":   4,
    "log_every":     10,
    "wandb_project": "ijepa-active-matter",
}


# ─────────────────────────────────────────────
# Frame subsampling
# ─────────────────────────────────────────────
def subsample_frames(x: torch.Tensor, num_frames: int) -> torch.Tensor:
    """Uniformly subsample to num_frames. x: (B, C, T, H, W)"""
    T = x.shape[2]
    if T == num_frames:
        return x
    indices = torch.linspace(0, T - 1, num_frames).long()
    return x[:, :, indices, :, :]


# ─────────────────────────────────────────────
# LR / EMA / WD schedules
# ─────────────────────────────────────────────
def get_lr(step, total_steps, warmup_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))

def get_ema(step, total_steps, ema_start, ema_end):
    return ema_start + (ema_end - ema_start) * step / max(1, total_steps)

def get_wd(step, total_steps, wd_start, wd_end):
    return wd_start + (wd_end - wd_start) * step / max(1, total_steps)


# ─────────────────────────────────────────────
# Atomic checkpoint (no corruption on preemption)
# ─────────────────────────────────────────────
def save_checkpoint(model, opt, epoch, step, cfg):
    os.makedirs(cfg["ckpt_dir"], exist_ok=True)
    final_path = os.path.join(cfg["ckpt_dir"], "latest.pt")
    payload = {
        "epoch": epoch,
        "step":  step,
        "model": model.state_dict(),
        "opt":   opt.state_dict(),
        "cfg":   cfg,
    }
    # Write to temp file first, then atomically rename
    tmp_fd, tmp_path = tempfile.mkstemp(dir=cfg["ckpt_dir"], suffix=".tmp")
    try:
        os.close(tmp_fd)
        torch.save(payload, tmp_path)
        os.replace(tmp_path, final_path)   # atomic on POSIX
        print(f"[ckpt] Saved → {final_path}  (epoch {epoch}, step {step})")
    except Exception as e:
        print(f"[ckpt] Save failed: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def load_checkpoint(model, opt, cfg):
    path = os.path.join(cfg["ckpt_dir"], "latest.pt")
    if not os.path.exists(path):
        print("[ckpt] No checkpoint — starting fresh.")
        return 0, 0
    try:
        ckpt = torch.load(path, map_location="cpu")
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if missing or unexpected:
            print(f"[ckpt] Architecture mismatch ({len(missing)} missing, "
                  f"{len(unexpected)} unexpected) — starting fresh.")
            return 0, 0
        opt.load_state_dict(ckpt["opt"])
        print(f"[ckpt] Resumed from epoch {ckpt['epoch']}, step {ckpt['step']}")
        return ckpt["epoch"], ckpt["step"]
    except Exception as e:
        print(f"[ckpt] Corrupt checkpoint ({e}) — starting fresh.")
        return 0, 0


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # Override ckpt path if --resume provided via sbatch
    if args.resume:
        CFG["ckpt_dir"] = os.path.dirname(args.resume)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] Using {device}")
    if device.type == "cuda":
        print(f"[device] {torch.cuda.get_device_name(0)}, "
              f"VRAM={torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    # ── dataset ──────────────────────────────────────────────────────────
    ds = ActiveMatterDataset(CFG["data_dir"], split="train")
    dl = DataLoader(
        ds,
        batch_size=CFG["batch_size"],
        shuffle=True,
        num_workers=CFG["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    print(f"[data] {len(ds)} samples, {len(dl)} batches/epoch "
          f"(batch_size={CFG['batch_size']})")

    # ── model ────────────────────────────────────────────────────────────
    model = IJEPA(
        in_channels=CFG["in_channels"],
        img_size=CFG["img_size"],
        patch_size=CFG["patch_size"],
        num_frames=CFG["num_frames"],
        encoder_dim=CFG["encoder_dim"],
        encoder_depth=CFG["encoder_depth"],
        encoder_heads=CFG["encoder_heads"],
        predictor_dim=CFG["predictor_dim"],
        predictor_depth=CFG["predictor_depth"],
        predictor_heads=CFG["predictor_heads"],
        ema_momentum=CFG["ema_start"],
    ).to(device)

    ctx_params  = sum(p.numel() for p in model.context_encoder.parameters())
    pred_params = sum(p.numel() for p in model.predictor.parameters())
    total_train = ctx_params + pred_params
    print(f"[model] Context encoder: {ctx_params:,}")
    print(f"[model] Predictor      : {pred_params:,}")
    print(f"[model] Total trainable: {total_train:,}")
    assert total_train < 100_000_000, "Exceeds 100M param limit!"

    # Token count sanity check
    h = w = CFG["img_size"] // CFG["patch_size"]
    N_total = CFG["num_frames"] * h * w
    print(f"[model] Tokens per sample: {N_total} "
          f"({CFG['num_frames']} frames × {h}×{w} patches)")
    print(f"[model] Attention matrix : {N_total}×{N_total} = {N_total**2:,} elements")

    # ── mask sampler ─────────────────────────────────────────────────────
    h_patches = CFG["img_size"] // CFG["patch_size"]
    mask_sampler = MultiBlockMaskSampler(
        h_patches=h_patches,
        w_patches=h_patches,
        num_frames=CFG["num_frames"],
        num_target_blocks=CFG["num_target_blocks"],
        target_scale=CFG["target_scale"],
        target_ratio=CFG["target_ratio"],
        context_scale=CFG["context_scale"],
    )

    # ── optimiser ────────────────────────────────────────────────────────
    opt = torch.optim.AdamW(
        list(model.context_encoder.parameters()) +
        list(model.predictor.parameters()),
        lr=CFG["lr"],
        weight_decay=CFG["weight_decay_start"],
        betas=(0.9, 0.95),
    )

    total_steps  = CFG["epochs"] * len(dl)
    warmup_steps = CFG["warmup_epochs"] * len(dl)

    # ── resume ───────────────────────────────────────────────────────────
    start_epoch, step = load_checkpoint(model, opt, CFG)

    # ── wandb ────────────────────────────────────────────────────────────
    wandb.init(project=CFG["wandb_project"], config=CFG, resume="allow")

    # ── VRAM check before training ────────────────────────────────────────
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        x_test = torch.randn(CFG["batch_size"], CFG["in_channels"],
                             CFG["num_frames"], CFG["img_size"], CFG["img_size"],
                             device=device)
        x_test = subsample_frames(x_test, CFG["num_frames"])
        masks_test = mask_sampler(CFG["batch_size"], device)
        with torch.no_grad():
            _ = model(x_test, masks_test)
        peak_mb = torch.cuda.max_memory_allocated() / 1e6
        total_mb = torch.cuda.get_device_properties(0).total_memory / 1e6
        print(f"[vram] Peak after test forward: {peak_mb:.0f}MB / {total_mb:.0f}MB "
              f"({100*peak_mb/total_mb:.1f}%)")
        del x_test, masks_test
        torch.cuda.empty_cache()

    # ── training loop ────────────────────────────────────────────────────
    for epoch in range(start_epoch, CFG["epochs"]):
        model.train()
        epoch_start = time.time()

        for batch_idx, batch in enumerate(dl):
            step_start = time.time()

            # Schedules
            lr    = get_lr(step, total_steps, warmup_steps, CFG["lr"], CFG["min_lr"])
            wd    = get_wd(step, total_steps, CFG["weight_decay_start"], CFG["weight_decay_end"])
            ema_m = get_ema(step, total_steps, CFG["ema_start"], CFG["ema_end"])

            for pg in opt.param_groups:
                pg["lr"]           = lr
                pg["weight_decay"] = wd
            model.ema_momentum = ema_m

            # Data
            x = batch["context"].to(device, non_blocking=True)
            x = subsample_frames(x, CFG["num_frames"])

            # Masks
            masks = mask_sampler(CFG["batch_size"], device)

            # Forward + backward
            loss, metrics = model(x, masks)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.context_encoder.parameters()) +
                list(model.predictor.parameters()),
                CFG["grad_clip"]
            )
            opt.step()

            step_time = time.time() - step_start
            metrics["lr"]        = lr
            metrics["ema_mom"]   = ema_m
            metrics["step_time"] = step_time

            if step % CFG["log_every"] == 0:
                eta_epoch = step_time * (len(dl) - batch_idx)
                print(
                    f"E{epoch:02d} S{step:05d} | "
                    f"loss={metrics['loss']:.4f}  "
                    f"pred_norm={metrics['pred_norm']:.3f}  "
                    f"tgt_norm={metrics['target_norm']:.3f}  "
                    f"lr={lr:.2e}  "
                    f"step={step_time:.2f}s  "
                    f"eta_epoch={eta_epoch/60:.1f}min"
                )
                wandb.log(metrics, step=step)

            # Frequent checkpointing — every CKPT_EVERY steps
            if step % CFG["ckpt_every"] == 0 and step > 0:
                save_checkpoint(model, opt, epoch, step, CFG)

            step += 1

        epoch_time = time.time() - epoch_start
        print(f"[epoch {epoch}] completed in {epoch_time/60:.1f} min")
        # Also save at end of each epoch
        save_checkpoint(model, opt, epoch, step, CFG)

    print("Training complete.")
    wandb.finish()


if __name__ == "__main__":
    main()