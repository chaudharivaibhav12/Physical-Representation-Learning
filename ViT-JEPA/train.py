"""
Training Script for ViT-JEPA v2 on Active Matter Dataset
=========================================================
Changes from v1:
  - stride: 4 → 1 (4x more temporal diversity, matches jepa-baseline)
  - std_weight: 40 → 20 (stable with token-level VICReg)
  - collapse monitoring now uses forward_pooled() to match eval representation

Usage:
  Single GPU:
    python train.py

  Multi GPU:
    torchrun --nproc_per_node=4 train.py

  Resume from checkpoint:
    python train.py --resume /path/to/checkpoint.pt

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

from model   import ViTJEPA
from dataset import ActiveMatterDataset


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

CONFIG = {
    # Data
    "data_dir":          "/scratch/ok2287/data/active_matter/data",
    "num_frames":        16,
    "crop_size":         224,
    "stride":            1,        # was 4; stride=1 gives ~4x more training windows
    "noise_std":         1.0,

    # Model (v2)
    "in_channels":       11,
    "embed_dim":         384,
    "depth":             6,
    "num_heads":         6,
    "mlp_ratio":         4.0,
    "dropout":           0.0,
    "patch_size":        32,       # 392 tokens (7x7x8)
    "tubelet":           2,

    # VICReg (v2)
    "sim_weight":        2.0,
    "std_weight":        20.0,     # was 40; stable with token-level VICReg
    "cov_weight":        2.0,

    # Training
    "epochs":            100,
    "batch_size":        4,
    "target_batch":      256,
    "lr":                1e-3,
    "weight_decay":      0.05,
    "grad_clip":         1.0,
    "warmup_epochs":     5,
    "amp_dtype":         "bf16",

    # Checkpointing
    "out_dir":           "/scratch/ok2287/checkpoints/vit_jepa_v2",
    "save_every":        5,

    # Logging
    "wandb_project":     "active-matter-jepa",
    "run_name":          "vit-jepa-v2-d384-p32",
    "log_every":         10,
}


# ─────────────────────────────────────────────
# Cosine LR Schedule with Linear Warmup
# ─────────────────────────────────────────────

def get_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float, min_lr: float = 1e-6) -> float:
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
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
        "epoch":          epoch,
        "encoder":        encoder.state_dict(),
        "model":          model.state_dict(),
        "optimizer":      optimizer.state_dict(),
        "scaler":         scaler.state_dict(),
        "best_val_loss":  best_val_loss,
        "config":         cfg,
    }, path)
    print(f"  Saved checkpoint: {path}")


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
    """
    Checks for representation collapse using forward_pooled() — the same
    representation used at evaluation time (linear probe / kNN).
    Reports average std across embedding dimensions; < 0.1 signals collapse risk.
    """
    model.eval()
    enc = model.module.encoder if hasattr(model, "module") else model.encoder

    stds = []
    for i, batch in enumerate(val_loader):
        if i >= n_batches:
            break
        ctx = batch["context"].to(device)
        z   = enc.forward_pooled(ctx)          # (B, D) — matches eval representation
        std = z.std(dim=0).mean().item()
        stds.append(std)

    return sum(stds) / len(stds)


# ─────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────

def train(args, cfg):
    rank, world_size, distributed = setup_distributed()
    is_main = (rank == 0)
    device  = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # ── Datasets ────────────────────────────────────────────────────
    train_dataset = ActiveMatterDataset(
        data_dir   = cfg["data_dir"],
        split      = "train",
        num_frames = cfg["num_frames"],
        crop_size  = cfg["crop_size"],
        stride     = cfg["stride"],
        noise_std  = cfg["noise_std"],
    )
    val_dataset = ActiveMatterDataset(
        data_dir   = cfg["data_dir"],
        split      = "valid",
        num_frames = cfg["num_frames"],
        crop_size  = cfg["crop_size"],
        stride     = cfg["stride"],
        noise_std  = 0.0,
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

    # ── Model ────────────────────────────────────────────────────────
    model = ViTJEPA(
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
        sim_weight  = cfg["sim_weight"],
        std_weight  = cfg["std_weight"],
        cov_weight  = cfg["cov_weight"],
    ).to(device)

    if distributed:
        model = DDP(model, device_ids=[rank])

    if is_main:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n[MODEL] Total parameters: {total_params:,}")
        print(f"[MODEL] Under 100M: {total_params < 100_000_000}\n")

    # ── Optimizer ────────────────────────────────────────────────────
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
        print(f"[TRAIN] Gradient accumulation steps: {accum_steps}")
        print(f"[TRAIN] Effective batch size: {cfg['batch_size'] * world_size * accum_steps}\n")

    # ── LR Schedule ─────────────────────────────────────────────────
    steps_per_epoch = len(train_loader) // accum_steps
    total_steps     = cfg["epochs"] * steps_per_epoch
    warmup_steps    = cfg["warmup_epochs"] * steps_per_epoch

    # ── Resume ───────────────────────────────────────────────────────
    start_epoch   = 0
    best_val_loss = float("inf")
    global_step   = 0

    if args.resume and os.path.exists(args.resume):
        if is_main:
            print(f"[RESUME] Loading checkpoint: {args.resume}")
        start_epoch, best_val_loss = load_checkpoint(args.resume, model, optimizer, scaler, device)
        global_step = start_epoch * steps_per_epoch
        if is_main:
            print(f"[RESUME] Resuming from epoch {start_epoch}\n")

    # ── WandB ────────────────────────────────────────────────────────
    os.makedirs(cfg["out_dir"], exist_ok=True)
    if is_main and not args.dry_run:
        wandb.init(
            project = cfg["wandb_project"],
            name    = cfg["run_name"],
            config  = cfg,
        )

    # ── Training ─────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        if train_sampler:
            train_sampler.set_epoch(epoch)

        epoch_loss = 0.0
        n_batches  = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            ctx = batch["context"].to(device, non_blocking=True)
            tgt = batch["target"].to(device,  non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                loss, metrics = model(ctx, tgt)
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
                        f"Epoch {epoch+1:3d} | Step {global_step:5d} | "
                        f"LR {lr:.2e} | "
                        f"Loss {metrics['loss_total']:.4f} | "
                        f"Inv {metrics['loss_invariance']:.4f} | "
                        f"Var {metrics['loss_variance']:.4f} | "
                        f"Cov {metrics['loss_covariance']:.4f}"
                    )
                    if not args.dry_run:
                        wandb.log({**metrics, "lr": lr, "epoch": epoch + 1}, step=global_step)

        # ── Validation ───────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        n_val    = 0
        with torch.no_grad():
            for batch in val_loader:
                ctx = batch["context"].to(device)
                tgt = batch["target"].to(device)
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    loss, metrics = model(ctx, tgt)
                val_loss += metrics["loss_total"]
                n_val    += 1

        val_loss /= max(n_val, 1)

        # Collapse check on pooled embeddings (matches eval representation)
        emb_std = check_collapse(model, val_loader, device)

        if is_main:
            avg_train = epoch_loss / max(n_batches, 1)
            collapse_status = "⚠ COLLAPSE RISK" if emb_std < 0.1 else "✓ healthy"
            print(
                f"\n── Epoch {epoch+1} Summary ──────────────────────────\n"
                f"  Train Loss:    {avg_train:.4f}\n"
                f"  Val Loss:      {val_loss:.4f}\n"
                f"  Embedding Std: {emb_std:.4f}  {collapse_status}\n"
            )
            if not args.dry_run:
                wandb.log({
                    "val_loss":      val_loss,
                    "embedding_std": emb_std,
                    "epoch":         epoch + 1,
                }, step=global_step)

            latest_path = os.path.join(cfg["out_dir"], "latest.pt")
            save_checkpoint(latest_path, epoch + 1, model, optimizer, scaler, best_val_loss, cfg)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(cfg["out_dir"], "best.pt")
                save_checkpoint(best_path, epoch + 1, model, optimizer, scaler, best_val_loss, cfg)
                print(f"  ✓ New best model saved!\n")

            if (epoch + 1) % cfg["save_every"] == 0:
                ep_path = os.path.join(cfg["out_dir"], f"epoch_{epoch+1}.pt")
                save_checkpoint(ep_path, epoch + 1, model, optimizer, scaler, best_val_loss, cfg)

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
