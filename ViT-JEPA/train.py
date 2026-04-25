"""
Training script for ViT-JEPA with WandB logging.

All fixes applied:
  1. GradScaler updated to torch.amp (no deprecation warning)
  2. num_workers and pin_memory auto-detected (CPU dry run safe)
  3. SIGUSR1 signal handler saves checkpoint before preemption
  4. best_val_loss updated BEFORE saving latest.pt (fixes inf bug)
  5. batch_size=8, target_batch=64 (35 steps/epoch instead of 8)
  6. WandB run ID persisted to file — resumes same run on restart
     (clean continuous charts, no more crashed runs)

Usage:
  Fresh training:
    python train.py

  Resume from checkpoint:
    python train.py --resume /scratch/ok2287/checkpoints/vit_jepa/latest.pt

  Dry run (1 epoch, no wandb):
    python train.py --dry-run
"""

import os
import math
import signal
import argparse
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader

from model   import ViTJEPA
from dataset import ActiveMatterDataset


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

CFG = {
    # Data
    "data_dir":       "/scratch/ok2287/data/active_matter/data",
    "num_frames":     16,
    "crop_size":      224,
    "stride":         1,
    "noise_std":      1.0,

    # Model
    "embed_dim":      384,
    "depth":          8,
    "num_heads":      6,
    "patch_size":     32,
    "tubelet":        2,

    # Predictor
    "predictor_dim":  192,
    "pred_depth":     2,
    "pred_heads":     4,

    # VICReg
    "sim_weight":     2.0,
    "std_weight":     40.0,
    "cov_weight":     2.0,

    # Training
    # batch_size=8 + target_batch=64 → accum=8 → ~35 steps/epoch
    # (previously batch=4, target=256 → accum=64 → only 8 steps/epoch)
    "epochs":         100,
    "batch_size":     8,
    "target_batch":   64,
    "lr":             1e-3,
    "weight_decay":   0.05,
    "grad_clip":      1.0,
    "warmup_epochs":  5,

    # Checkpointing
    "out_dir":        "/scratch/ok2287/checkpoints/vit_jepa",
    "save_every":     5,

    # Logging
    "log_every":      10,
    "wandb_project":  "active-matter-jepa",
    "run_name":       "vit-jepa-d384-p16",
}


# ─────────────────────────────────────────────
# LR Schedule: Cosine with Linear Warmup
# ─────────────────────────────────────────────

def get_lr(step, total_steps, warmup_steps, base_lr, min_lr=1e-6):
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ─────────────────────────────────────────────
# Checkpoint Utilities
# ─────────────────────────────────────────────

def save_checkpoint(path, epoch, model, optimizer, scaler, best_val_loss):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch":         epoch,
        "encoder":       model.encoder.state_dict(),
        "model":         model.state_dict(),
        "optimizer":     optimizer.state_dict(),
        "scaler":        scaler.state_dict(),
        "best_val_loss": best_val_loss,
        "config":        CFG,
    }, path)
    print(f"  Saved: {path}")


def load_checkpoint(path, model, optimizer, scaler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    start_epoch   = ckpt["epoch"]
    best_val_loss = ckpt.get("best_val_loss", float("inf"))
    print(f"  Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")
    return start_epoch, best_val_loss


# ─────────────────────────────────────────────
# WandB: persist run ID across preemptions
# ─────────────────────────────────────────────

def init_wandb(cfg, dry_run):
    """
    Initialize WandB, resuming the same run if it exists.
    Saves run ID to a file so preempted jobs resume the same run
    instead of creating new crashed runs.
    """
    if dry_run:
        return

    os.makedirs(cfg["out_dir"], exist_ok=True)
    wandb_id_file = os.path.join(cfg["out_dir"], "wandb_run_id.txt")

    if os.path.exists(wandb_id_file):
        with open(wandb_id_file, "r") as f:
            run_id = f.read().strip()
        print(f"Resuming wandb run: {run_id}")
        wandb.init(
            project = cfg["wandb_project"],
            name    = cfg["run_name"],
            id      = run_id,
            resume  = "must",
            config  = cfg,
        )
    else:
        run = wandb.init(
            project = cfg["wandb_project"],
            name    = cfg["run_name"],
            config  = cfg,
        )
        with open(wandb_id_file, "w") as f:
            f.write(run.id)
        print(f"Started new wandb run: {run.id}")


# ─────────────────────────────────────────────
# Main Training Function
# ─────────────────────────────────────────────

def main(args):
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Auto-detect workers and pin_memory ───────────────────────────
    # num_workers=0 on CPU avoids worker OOM kills during dry run
    num_workers = 4 if torch.cuda.is_available() else 0
    pin_memory  = torch.cuda.is_available()
    print(f"num_workers={num_workers}, pin_memory={pin_memory}")

    # ── WandB ────────────────────────────────────────────────────────
    init_wandb(CFG, args.dry_run)

    # ── Datasets ─────────────────────────────────────────────────────
    train_ds = ActiveMatterDataset(
        CFG["data_dir"], split="train",
        stride=CFG["stride"], noise_std=CFG["noise_std"],
    )
    val_ds = ActiveMatterDataset(
        CFG["data_dir"], split="valid",
        stride=CFG["stride"], noise_std=0.0,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size  = CFG["batch_size"],
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = pin_memory,
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = CFG["batch_size"],
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = pin_memory,
    )

    # ── Model ─────────────────────────────────────────────────────────
    model = ViTJEPA(
        in_channels   = 11,
        embed_dim     = CFG["embed_dim"],
        depth         = CFG["depth"],
        num_heads     = CFG["num_heads"],
        img_size      = CFG["crop_size"],
        patch_size    = CFG["patch_size"],
        tubelet       = CFG["tubelet"],
        num_frames    = CFG["num_frames"],
        predictor_dim = CFG["predictor_dim"],
        pred_depth    = CFG["pred_depth"],
        pred_heads    = CFG["pred_heads"],
        sim_weight    = CFG["sim_weight"],
        std_weight    = CFG["std_weight"],
        cov_weight    = CFG["cov_weight"],
    ).to(device)

    params = model.count_parameters()
    print(f"Encoder: {params['encoder']:,}  "
          f"Predictor: {params['predictor']:,}  "
          f"Total: {params['total']:,}  "
          f"(< 100M: {params['total'] < 100_000_000})")

    # ── Optimizer ────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = CFG["lr"],
        betas        = (0.9, 0.95),
        weight_decay = CFG["weight_decay"],
    )

    # ── GradScaler (fixed deprecation) ───────────────────────────────
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    # ── Gradient accumulation ────────────────────────────────────────
    accum_steps  = max(1, CFG["target_batch"] // CFG["batch_size"])
    steps_per_ep = len(train_loader) // accum_steps
    total_steps  = CFG["epochs"] * steps_per_ep
    warmup_steps = CFG["warmup_epochs"] * steps_per_ep
    print(f"Accum steps: {accum_steps} | Steps/epoch: {steps_per_ep} | Total: {total_steps}")

    # ── State variables ───────────────────────────────────────────────
    # Defined before signal handler so handler can access them via closure
    start_epoch   = 0
    best_val_loss = float("inf")
    global_step   = 0
    epoch         = 0

    os.makedirs(CFG["out_dir"], exist_ok=True)

    # ── Resume from checkpoint ───────────────────────────────────────
    if args.resume and os.path.exists(args.resume):
        print(f"Loading checkpoint: {args.resume}")
        start_epoch, best_val_loss = load_checkpoint(
            args.resume, model, optimizer, scaler, device
        )
        global_step = start_epoch * steps_per_ep

    # ── Preemption Signal Handler ────────────────────────────────────
    # Slurm sends SIGUSR1 90s before killing (requires --signal=SIGUSR1@90)
    # Saves checkpoint so --requeue can resume cleanly
    def handle_preemption(signum, frame):
        print(f"\n⚠ SIGUSR1 received at epoch {epoch} — saving before preemption...")
        save_checkpoint(
            f"{CFG['out_dir']}/latest.pt",
            epoch, model, optimizer, scaler, best_val_loss,
        )
        print("✓ Checkpoint saved. Job will requeue and resume automatically.")
        if not args.dry_run:
            wandb.finish()
        exit(0)

    signal.signal(signal.SIGUSR1, handle_preemption)
    # ─────────────────────────────────────────────────────────────────

    # ── Training Loop ────────────────────────────────────────────────
    for epoch in range(start_epoch, CFG["epochs"]):
        model.train()
        optimizer.zero_grad()
        epoch_loss = 0.0
        n_batches  = 0

        for step, batch in enumerate(train_loader):
            ctx = batch["context"].to(device, non_blocking=True)
            tgt = batch["target"].to(device,  non_blocking=True)

            # Forward pass with mixed precision
            with torch.autocast(
                device_type = "cuda" if torch.cuda.is_available() else "cpu",
                dtype       = torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            ):
                loss, metrics = model(ctx, tgt)
                loss = loss / accum_steps

            # Backward
            scaler.scale(loss).backward()

            # Optimizer step every accum_steps
            if (step + 1) % accum_steps == 0:
                lr = get_lr(global_step, total_steps, warmup_steps, CFG["lr"])
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                global_step += 1
                epoch_loss  += metrics["loss_total"]
                n_batches   += 1

                if global_step % CFG["log_every"] == 0:
                    print(
                        f"Ep {epoch+1:3d} | Step {global_step:5d} | LR {lr:.2e} | "
                        f"Loss {metrics['loss_total']:.4f} | "
                        f"Inv {metrics['loss_invariance']:.4f} | "
                        f"Var {metrics['loss_variance']:.4f} | "
                        f"Cov {metrics['loss_covariance']:.4f}"
                    )
                    if not args.dry_run:
                        wandb.log({**metrics, "lr": lr, "epoch": epoch + 1},
                                  step=global_step)

        # ── Validation ───────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        n_val    = 0
        with torch.no_grad():
            for batch in val_loader:
                ctx = batch["context"].to(device)
                tgt = batch["target"].to(device)
                with torch.autocast(
                    device_type = "cuda" if torch.cuda.is_available() else "cpu",
                    dtype       = torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                ):
                    _, m = model(ctx, tgt)
                val_loss += m["loss_total"]
                n_val    += 1
        val_loss /= max(n_val, 1)

        # ── Collapse Check ───────────────────────────────────────────
        with torch.no_grad():
            sample_batch = next(iter(val_loader))
            z = model.encode(sample_batch["context"].to(device))
            emb_std = z.std(dim=0).mean().item()

        # ── Epoch Summary ────────────────────────────────────────────
        avg_train        = epoch_loss / max(n_batches, 1)
        collapse_warning = "⚠ COLLAPSE RISK" if emb_std < 0.1 else "✓ healthy"
        print(f"\n── Epoch {epoch+1} ──────────────────────────────")
        print(f"  Train Loss:    {avg_train:.4f}")
        print(f"  Val Loss:      {val_loss:.4f}")
        print(f"  Embedding Std: {emb_std:.4f}  {collapse_warning}\n")

        if not args.dry_run:
            wandb.log({
                "val_loss":      val_loss,
                "train_loss":    avg_train,
                "embedding_std": emb_std,
                "epoch":         epoch + 1,
            }, step=global_step)

        # ── Checkpointing ────────────────────────────────────────────
        # IMPORTANT: update best_val_loss FIRST, then save latest.pt
        # This fixes the bug where latest.pt was saved with best_val_loss=inf

        # 1. Check and update best model first
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                f"{CFG['out_dir']}/best.pt",
                epoch + 1, model, optimizer, scaler, best_val_loss,
            )
            print("  ✓ New best model!\n")

        # 2. Save latest AFTER best_val_loss is updated
        save_checkpoint(
            f"{CFG['out_dir']}/latest.pt",
            epoch + 1, model, optimizer, scaler, best_val_loss,
        )

        # 3. Periodic epoch snapshots
        if (epoch + 1) % CFG["save_every"] == 0:
            save_checkpoint(
                f"{CFG['out_dir']}/epoch_{epoch+1}.pt",
                epoch + 1, model, optimizer, scaler, best_val_loss,
            )

        if args.dry_run:
            print("Dry run complete.")
            break

    # ── Finish ───────────────────────────────────────────────────────
    if not args.dry_run:
        wandb.finish()
    print("Training complete!")


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ViT-JEPA on active matter dataset")
    parser.add_argument("--resume",  type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run 1 epoch without wandb for testing")
    main(parser.parse_args())