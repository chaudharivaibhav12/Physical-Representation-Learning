"""
VICReg Training — sarvesh v4
=============================
Single-GPU, no DDP. WandB run ID persisted to file for preemption recovery.

Usage:
  python train.py
  python train.py --resume /scratch/sb10583/checkpoints/vicreg-v4/latest.pt
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

from model   import VICReg
from dataset import ActiveMatterDataset


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

CFG = {
    # Data
    "data_dir":       "/scratch/sb10583/data/data",
    "crop_size":      224,
    "noise_std":      1.0,

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
    "var_weight":     50.0,
    "cov_weight":     1.0,

    # Training
    "epochs":         100,
    "batch_size":     8,
    "target_batch":   64,
    "lr":             1e-3,
    "weight_decay":   0.05,
    "grad_clip":      1.0,
    "warmup_epochs":  5,

    # Checkpointing
    "out_dir":        "/scratch/sb10583/checkpoints/vicreg-v4",
    "save_every":     5,
    "save_every_steps": 50,

    # Logging
    "wandb_project":  "vicreg-active-matter-v4",
    "run_name":       "sarvesh-v4",
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
# Checkpoint Utilities
# ─────────────────────────────────────────────

def save_checkpoint(path, epoch, global_step, model, optimizer, scaler, best_val_loss):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch":         epoch,
        "global_step":   global_step,
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
    state_dict = ckpt["model"]
    if all(k.startswith("module.") for k in state_dict):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler_state = ckpt.get("scaler", {})
    if scaler_state:
        scaler.load_state_dict(scaler_state)
    start_epoch   = ckpt["epoch"]
    global_step   = ckpt.get("global_step", 0)
    best_val_loss = ckpt.get("best_val_loss", float("inf"))
    print(f"  Resumed from epoch {start_epoch}, step {global_step}, best_val_loss={best_val_loss:.4f}")
    return start_epoch, global_step, best_val_loss




# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main(args):
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    num_workers = 4 if torch.cuda.is_available() else 0
    pin_memory  = torch.cuda.is_available()

    if not args.dry_run:
        wandb.init(
            project = CFG["wandb_project"],
            name    = CFG["run_name"],
            config  = CFG,
        )

    # ── Datasets ─────────────────────────────────────────────────────
    train_ds = ActiveMatterDataset(
        CFG["data_dir"], split="train",
        crop_size=CFG["crop_size"], noise_std=CFG["noise_std"],
    )
    val_ds = ActiveMatterDataset(
        CFG["data_dir"], split="valid",
        crop_size=CFG["crop_size"], noise_std=0.0,
    )
    train_loader = DataLoader(
        train_ds, batch_size=CFG["batch_size"], shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=CFG["batch_size"], shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    # ── Model ─────────────────────────────────────────────────────────
    model = VICReg(
        in_channels = CFG["in_channels"],
        embed_dim   = CFG["embed_dim"],
        depth       = CFG["depth"],
        num_heads   = CFG["num_heads"],
        mlp_ratio   = CFG["mlp_ratio"],
        dropout     = CFG["dropout"],
        img_size    = CFG["crop_size"],
        patch_size  = CFG["patch_size"],
        tubelet     = CFG["tubelet"],
        num_frames  = CFG["num_frames"],
        proj_hidden = CFG["proj_hidden"],
        proj_out    = CFG["proj_out"],
        sim_weight  = CFG["sim_weight"],
        var_weight  = CFG["var_weight"],
        cov_weight  = CFG["cov_weight"],
    ).to(device)

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total:,}  (< 100M: {total < 100_000_000})")

    # ── Optimizer ────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG["lr"],
        betas=(0.9, 0.95), weight_decay=CFG["weight_decay"],
    )
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    accum_steps  = max(1, CFG["target_batch"] // CFG["batch_size"])
    steps_per_ep = len(train_loader) // accum_steps
    total_steps  = CFG["epochs"] * steps_per_ep
    warmup_steps = CFG["warmup_epochs"] * steps_per_ep
    print(f"Accum: {accum_steps} | Steps/epoch: {steps_per_ep} | Total: {total_steps}")

    start_epoch   = 0
    best_val_loss = float("inf")
    global_step   = 0
    epoch         = 0

    os.makedirs(CFG["out_dir"], exist_ok=True)

    if args.resume and os.path.exists(args.resume):
        print(f"Loading checkpoint: {args.resume}")
        start_epoch, global_step, best_val_loss = load_checkpoint(
            args.resume, model, optimizer, scaler, device,
        )

    # ── Preemption handler ───────────────────────────────────────────
    def handle_preemption(signum, frame):
        print(f"\n⚠ SIGUSR1 received at epoch {epoch} — saving before preemption...")
        save_checkpoint(
            f"{CFG['out_dir']}/latest.pt",
            epoch, global_step, model, optimizer, scaler, best_val_loss,
        )
        print("✓ Checkpoint saved. Job will requeue and resume.")
        if not args.dry_run:
            wandb.finish()
        exit(0)

    signal.signal(signal.SIGUSR1, handle_preemption)

    # ── Training loop ────────────────────────────────────────────────
    for epoch in range(start_epoch, CFG["epochs"]):
        model.train()
        optimizer.zero_grad()
        epoch_loss = 0.0
        n_batches  = 0

        for step, batch in enumerate(train_loader):
            v1 = batch["view1"].to(device, non_blocking=True)
            v2 = batch["view2"].to(device, non_blocking=True)

            with torch.autocast(
                device_type = "cuda" if torch.cuda.is_available() else "cpu",
                dtype       = torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            ):
                loss, metrics = model(v1, v2)
                loss = loss / accum_steps

            scaler.scale(loss).backward()

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

                epoch_loss += metrics["loss_total"]
                n_batches  += 1

                if global_step % CFG["log_every"] == 0:
                    print(
                        f"Ep {epoch+1:3d} | Step {global_step:5d} | LR {lr:.2e} | "
                        f"Loss {metrics['loss_total']:.4f} | "
                        f"Inv {metrics['loss_inv_raw']:.4f} | "
                        f"Var {metrics['loss_var_raw']:.4f} | "
                        f"Cov {metrics['loss_cov_raw']:.4f}"
                    )
                    if not args.dry_run:
                        wandb.log({**metrics, "lr": lr, "epoch": epoch + 1}, step=global_step)

                if global_step % CFG["save_every_steps"] == 0:
                    save_checkpoint(
                        f"{CFG['out_dir']}/latest.pt",
                        epoch, global_step, model, optimizer, scaler, best_val_loss,
                    )

        # ── Validation ───────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        n_val    = 0
        with torch.no_grad():
            for batch in val_loader:
                v1 = batch["view1"].to(device)
                v2 = batch["view2"].to(device)
                with torch.autocast(
                    device_type = "cuda" if torch.cuda.is_available() else "cpu",
                    dtype       = torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                ):
                    _, m = model(v1, v2)
                val_loss += m["loss_total"]
                n_val    += 1
        val_loss /= max(n_val, 1)

        # ── Collapse check (10 batches for reliable estimate) ────────
        model.eval()
        with torch.no_grad():
            stds = []
            for i, b in enumerate(val_loader):
                if i >= 10:
                    break
                z = model.encoder.forward_pooled(b["view1"].to(device))
                stds.append(z.std(dim=0).mean().item())
            emb_std = sum(stds) / len(stds)

        avg_train = epoch_loss / max(n_batches, 1)
        print(f"\n── Epoch {epoch+1} ──────────────────────────────")
        print(f"  Train Loss:    {avg_train:.4f}")
        print(f"  Val Loss:      {val_loss:.4f}")
        print(f"  Embedding Std: {emb_std:.4f}  {'⚠ COLLAPSE RISK' if emb_std < 0.1 else '✓ healthy'}\n")

        if not args.dry_run:
            wandb.log({
                "val_loss": val_loss, "train_loss": avg_train,
                "embedding_std": emb_std, "epoch": epoch + 1,
            }, step=global_step)

        # ── Checkpointing ────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(f"{CFG['out_dir']}/best.pt",
                            epoch + 1, global_step, model, optimizer, scaler, best_val_loss)
            print("  ✓ New best model!\n")

        save_checkpoint(f"{CFG['out_dir']}/latest.pt",
                        epoch + 1, global_step, model, optimizer, scaler, best_val_loss)

        if (epoch + 1) % CFG["save_every"] == 0:
            save_checkpoint(f"{CFG['out_dir']}/epoch_{epoch+1}.pt",
                            epoch + 1, global_step, model, optimizer, scaler, best_val_loss)

        if args.dry_run:
            print("Dry run complete.")
            break

    if not args.dry_run:
        wandb.finish()
    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume",  type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    main(parser.parse_args())
