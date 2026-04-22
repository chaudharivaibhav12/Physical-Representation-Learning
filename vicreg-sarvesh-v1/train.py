"""
VICReg Training Script — Active Matter Physics Simulations
==========================================================
Usage:
  Single GPU:
    python train.py

  Resume from checkpoint:
    python train.py --resume /scratch/sb10583/checkpoints/vicreg/latest.pt

  Dry run (1 epoch, no wandb):
    python train.py --dry-run
"""

import os
import math
import time
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler

import wandb

from model   import VICReg
from dataset import ActiveMatterDataset


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

CONFIG = {
    # Data
    "data_dir":       "/scratch/sb10583/data/data",
    "crop_size":      224,
    "noise_std":      1.0,
    "stride":         4,

    # Model
    "in_channels":    11,
    "embed_dim":      384,
    "depth":          6,
    "num_heads":      6,
    "mlp_ratio":      4.0,
    "dropout":        0.0,
    "patch_size":     32,
    "tubelet":        2,
    "num_frames":     16,
    "proj_hidden":    2048,
    "proj_out":       2048,

    # VICReg loss
    "sim_weight":     25.0,
    "var_weight":     25.0,
    "cov_weight":     1.0,

    # Training
    "epochs":         20,
    "batch_size":     4,              # per GPU
    "target_batch":   32,             # effective batch via grad accum
    "lr":             1e-3,
    "weight_decay":   0.05,
    "grad_clip":      1.0,
    "warmup_epochs":  5,
    "amp_dtype":      "bf16",

    # Checkpointing
    "out_dir":        "/scratch/sb10583/checkpoints/vicreg",
    "save_every":     5,

    # Logging
    "wandb_project":  "DL",
    "wandb_entity":   "sb10583",
    "run_name":       "sarvesh v1",
    "log_every":      10,
}


# ─────────────────────────────────────────────
# LR Schedule
# ─────────────────────────────────────────────

def get_lr(step, total_steps, warmup_steps, base_lr, min_lr=1e-6):
    if step < warmup_steps:
        return base_lr * max(step, 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ─────────────────────────────────────────────
# Distributed Setup
# ─────────────────────────────────────────────

def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank       = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank)
        return rank, world_size, True
    return 0, 1, False


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


# ─────────────────────────────────────────────
# Checkpoint Utilities
# ─────────────────────────────────────────────

def save_checkpoint(path, epoch, model, optimizer, scaler, best_val_loss, cfg):
    encoder = model.module.encoder if hasattr(model, "module") else model.encoder
    torch.save({
        "epoch":         epoch,
        "encoder":       encoder.state_dict(),
        "model":         model.state_dict(),
        "optimizer":     optimizer.state_dict(),
        "scaler":        scaler.state_dict(),
        "best_val_loss": best_val_loss,
        "config":        cfg,
    }, path)
    print(f"  Saved: {path}")


def load_checkpoint(path, model, optimizer, scaler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    return ckpt["epoch"], ckpt.get("best_val_loss", float("inf"))


# ─────────────────────────────────────────────
# Collapse Monitor
# ─────────────────────────────────────────────

@torch.no_grad()
def check_collapse(model, val_loader, device, n_batches=10):
    """Average std across embedding dims — near 0 means collapse."""
    model.eval()
    enc = model.module.encoder if hasattr(model, "module") else model.encoder
    stds = []
    for i, batch in enumerate(val_loader):
        if i >= n_batches:
            break
        x   = batch["view1"].to(device)
        z   = enc.forward_pooled(x)
        stds.append(z.std(dim=0).mean().item())
    return sum(stds) / len(stds)


# ─────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────

def train(args, cfg):
    rank, world_size, distributed = setup_distributed()
    is_main = (rank == 0)
    device  = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # ── Datasets ──────────────────────────────────────────────────────
    train_dataset = ActiveMatterDataset(
        data_dir   = cfg["data_dir"],
        split      = "train",
        crop_size  = cfg["crop_size"],
        noise_std  = cfg["noise_std"],
        stride     = cfg["stride"],
    )
    val_dataset = ActiveMatterDataset(
        data_dir   = cfg["data_dir"],
        split      = "valid",
        crop_size  = cfg["crop_size"],
        noise_std  = 0.0,
        stride     = cfg["stride"],
    )

    train_sampler = DistributedSampler(train_dataset) if distributed else None
    train_loader  = DataLoader(
        train_dataset,
        batch_size  = cfg["batch_size"],
        sampler     = train_sampler,
        shuffle     = (train_sampler is None),
        num_workers = 4,
        pin_memory  = True,
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = cfg["batch_size"],
        shuffle     = False,
        num_workers = 2,
        pin_memory  = True,
    )

    # ── Model ─────────────────────────────────────────────────────────
    model = VICReg(
        in_channels = cfg["in_channels"],
        embed_dim   = cfg["embed_dim"],
        depth       = cfg["depth"],
        num_heads   = cfg["num_heads"],
        mlp_ratio   = cfg["mlp_ratio"],
        dropout     = cfg["dropout"],
        img_size    = cfg["crop_size"],
        patch_size  = cfg["patch_size"],
        tubelet     = cfg["tubelet"],
        num_frames  = cfg["num_frames"],
        proj_hidden = cfg["proj_hidden"],
        proj_out    = cfg["proj_out"],
        sim_weight  = cfg["sim_weight"],
        var_weight  = cfg["var_weight"],
        cov_weight  = cfg["cov_weight"],
    ).to(device)

    if distributed:
        model = DDP(model, device_ids=[rank])

    if is_main:
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n[MODEL] Total parameters: {total:,}  (< 100M: {total < 100_000_000})\n")

    # ── Optimizer ─────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg["lr"],
        betas        = (0.9, 0.95),
        weight_decay = cfg["weight_decay"],
    )

    amp_dtype   = torch.bfloat16 if cfg["amp_dtype"] == "bf16" else torch.float16
    scaler      = GradScaler(enabled=(cfg["amp_dtype"] == "fp16"))
    accum_steps = max(1, cfg["target_batch"] // (cfg["batch_size"] * world_size))

    if is_main:
        print(f"[TRAIN] Grad accum steps:     {accum_steps}")
        print(f"[TRAIN] Effective batch size: {cfg['batch_size'] * world_size * accum_steps}\n")

    # ── LR Schedule ───────────────────────────────────────────────────
    steps_per_epoch = len(train_loader) // accum_steps
    total_steps     = cfg["epochs"] * steps_per_epoch
    warmup_steps    = cfg["warmup_epochs"] * steps_per_epoch

    # ── Resume ────────────────────────────────────────────────────────
    start_epoch   = 0
    best_val_loss = float("inf")
    global_step   = 0

    if args.resume and os.path.exists(args.resume):
        if is_main:
            print(f"[RESUME] Loading: {args.resume}")
        start_epoch, best_val_loss = load_checkpoint(args.resume, model, optimizer, scaler, device)
        global_step = start_epoch * steps_per_epoch
        if is_main:
            print(f"[RESUME] Resuming from epoch {start_epoch}\n")

    # ── W&B ───────────────────────────────────────────────────────────
    os.makedirs(cfg["out_dir"], exist_ok=True)
    if is_main and not args.dry_run:
        wandb.init(
            project = cfg["wandb_project"],
            entity  = cfg["wandb_entity"],
            name    = cfg["run_name"],
            config  = cfg,
        )

    # ── Training ──────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        if train_sampler:
            train_sampler.set_epoch(epoch)

        epoch_loss = 0.0
        n_batches  = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            v1 = batch["view1"].to(device, non_blocking=True)
            v2 = batch["view2"].to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                loss, metrics = model(v1, v2)
                loss = loss / accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0:
                lr = get_lr(global_step, total_steps, warmup_steps, cfg["lr"])
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1

                epoch_loss += metrics["loss_total"]
                n_batches  += 1

                if is_main and global_step % cfg["log_every"] == 0:
                    print(
                        f"Epoch {epoch+1:3d} | Step {global_step:5d} | LR {lr:.2e} | "
                        f"Loss {metrics['loss_total']:.4f} | "
                        f"Inv {metrics['loss_inv_raw']:.4f} | "
                        f"Var {metrics['loss_var_raw']:.4f} | "
                        f"Cov {metrics['loss_cov_raw']:.4f}"
                    )
                    if not args.dry_run:
                        wandb.log({**metrics, "lr": lr, "epoch": epoch + 1}, step=global_step)

        # ── Validation ────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        n_val    = 0
        with torch.no_grad():
            for batch in val_loader:
                v1 = batch["view1"].to(device)
                v2 = batch["view2"].to(device)
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    _, metrics = model(v1, v2)
                val_loss += metrics["loss_total"]
                n_val    += 1
        val_loss /= max(n_val, 1)

        emb_std = check_collapse(model, val_loader, device)

        if is_main:
            avg_train = epoch_loss / max(n_batches, 1)
            collapse  = "⚠ COLLAPSE RISK" if emb_std < 0.1 else "✓ healthy"
            print(
                f"\n── Epoch {epoch+1} ──────────────────────────────────\n"
                f"  Train Loss:    {avg_train:.4f}\n"
                f"  Val Loss:      {val_loss:.4f}\n"
                f"  Embedding Std: {emb_std:.4f}  {collapse}\n"
            )
            if not args.dry_run:
                wandb.log({
                    "val_loss":      val_loss,
                    "embedding_std": emb_std,
                    "epoch":         epoch + 1,
                }, step=global_step)

            # Checkpoints
            save_checkpoint(
                os.path.join(cfg["out_dir"], "latest.pt"),
                epoch + 1, model, optimizer, scaler, best_val_loss, cfg,
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    os.path.join(cfg["out_dir"], "best.pt"),
                    epoch + 1, model, optimizer, scaler, best_val_loss, cfg,
                )
                print("  ✓ New best model saved!\n")
            if (epoch + 1) % cfg["save_every"] == 0:
                save_checkpoint(
                    os.path.join(cfg["out_dir"], f"epoch_{epoch+1}.pt"),
                    epoch + 1, model, optimizer, scaler, best_val_loss, cfg,
                )

        if args.dry_run:
            print("[DRY RUN] Stopping after 1 epoch.")
            break

    if is_main and not args.dry_run:
        wandb.finish()

    cleanup_distributed()
    print("Training complete!")


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume",  type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    train(args, CONFIG)
