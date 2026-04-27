"""
VideoMAE Training Script
========================
Self-supervised masked autoencoder training on the active_matter dataset.

Key features:
  - 90% temporal tube masking → encoder sees only ~10% of patches
  - MSE reconstruction loss with per-patch normalization
  - AdamW optimizer + cosine LR schedule with warmup
  - bf16 mixed precision on A100
  - Step-level checkpointing every N steps (spot-instance safe)
  - SIGUSR1 preemption handling + SLURM --requeue compatible
  - Weights & Biases logging with run ID persisted across preemptions
  - Gradient accumulation for effective larger batch sizes

Usage:
  python train.py
  python train.py --resume /scratch/sb10583/checkpoints/videomae-v1/latest.pt
  python train.py --dry-run
"""

import os
import math
import signal
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import GradScaler

import wandb

from model   import VideoMAE
from dataset import VideoMAEDataset


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

CONFIG = {
    # Data
    "data_dir":    "/scratch/sb10583/data/data",
    "crop_size":   224,
    "num_frames":  16,
    "stride":      1,         # stride=1 → ~11,550 train samples

    # Model — ViT-Tiny encoder
    "in_channels":     11,
    "enc_embed_dim":   192,
    "enc_depth":       12,
    "enc_heads":       3,
    "mlp_ratio":       4.0,
    "dropout":         0.0,
    "patch_size":      16,
    "tubelet":         2,
    "mask_ratio":      0.90,

    # Decoder — lightweight, discarded after training
    "dec_embed_dim":   96,
    "dec_depth":       4,
    "dec_heads":       3,

    # Loss
    "norm_pix_loss":   True,

    # Training
    "epochs":          200,
    "batch_size":      8,       # per GPU (A100 40GB)
    "target_batch":    64,      # effective batch size via gradient accumulation
    "lr":              1.5e-4,  # base LR (scaled by effective batch / 256 at runtime)
    "weight_decay":    0.05,
    "grad_clip":       1.0,
    "warmup_epochs":   20,
    "amp_dtype":       "bf16",

    # Checkpointing
    "out_dir":         "/scratch/sb10583/checkpoints/videomae-v1",
    "save_every_steps": 50,   # step-level saves for spot-instance preemption
    "save_every_epochs": 5,   # extra epoch-level save for safety

    # Logging
    "wandb_project":   "DL",
    "wandb_entity":    "sb10583-",
    "run_name":        "videomae-v1",
    "log_every":       10,
}


# ─────────────────────────────────────────────
# LR Schedule — cosine with linear warmup
# ─────────────────────────────────────────────

def get_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float, min_lr: float = 1e-6) -> float:
    if step < warmup_steps:
        return base_lr * max(step, 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ─────────────────────────────────────────────
# Checkpoint Utilities
# ─────────────────────────────────────────────

def save_checkpoint(path, epoch, global_step, model, optimizer, scaler, best_loss, cfg):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    encoder = model.encoder
    torch.save({
        "epoch":       epoch,
        "global_step": global_step,
        "encoder":     encoder.state_dict(),
        "model":       model.state_dict(),
        "optimizer":   optimizer.state_dict(),
        "scaler":      scaler.state_dict(),
        "best_loss":   best_loss,
        "config":      cfg,
    }, path)
    print(f"  Saved: {path}")


def load_checkpoint(path, model, optimizer, scaler, device):
    ckpt = torch.load(path, map_location=device)
    state = ckpt["model"]
    if all(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    model.load_state_dict(state)
    optimizer.load_state_dict(ckpt["optimizer"])
    if ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    start_epoch   = ckpt["epoch"]
    global_step   = ckpt.get("global_step", 0)
    best_loss     = ckpt.get("best_loss", float("inf"))
    print(f"  Resumed from epoch {start_epoch}, step {global_step}, best_loss={best_loss:.4f}")
    return start_epoch, global_step, best_loss


# ─────────────────────────────────────────────
# W&B — persist run ID across preemptions
# ─────────────────────────────────────────────

def init_wandb(cfg, dry_run):
    if dry_run:
        return
    os.makedirs(cfg["out_dir"], exist_ok=True)
    id_file = os.path.join(cfg["out_dir"], "wandb_run_id.txt")
    if os.path.exists(id_file):
        run_id = open(id_file).read().strip()
        print(f"Resuming W&B run: {run_id}")
        wandb.init(
            project=cfg["wandb_project"], entity=cfg["wandb_entity"],
            name=cfg["run_name"], id=run_id, resume="allow", config=cfg,
        )
    else:
        run = wandb.init(
            project=cfg["wandb_project"], entity=cfg["wandb_entity"],
            name=cfg["run_name"], config=cfg,
        )
        open(id_file, "w").write(run.id)
        print(f"Started W&B run: {run.id}")


# ─────────────────────────────────────────────
# Per-epoch DataLoader (deterministic, resumable)
# ─────────────────────────────────────────────

def make_epoch_loader(dataset, epoch, batch_size, num_workers, skip_batches=0):
    g = torch.Generator()
    g.manual_seed(epoch)
    indices = torch.randperm(len(dataset), generator=g).tolist()
    n = (len(indices) // batch_size) * batch_size
    indices = indices[:n]
    if skip_batches > 0:
        indices = indices[skip_batches * batch_size:]
    subset = Subset(dataset, indices)
    return DataLoader(
        subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train(args, cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Datasets ──────────────────────────────────────────────────────
    train_dataset = VideoMAEDataset(
        data_dir=cfg["data_dir"], split="train",
        num_frames=cfg["num_frames"], crop_size=cfg["crop_size"], stride=cfg["stride"],
    )
    val_dataset = VideoMAEDataset(
        data_dir=cfg["data_dir"], split="valid",
        num_frames=cfg["num_frames"], crop_size=cfg["crop_size"], stride=cfg["stride"],
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg["batch_size"],
        shuffle=False, num_workers=2, pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────
    model = VideoMAE(
        in_channels=cfg["in_channels"], num_frames=cfg["num_frames"], img_size=cfg["crop_size"],
        enc_embed_dim=cfg["enc_embed_dim"], enc_depth=cfg["enc_depth"], enc_heads=cfg["enc_heads"],
        mlp_ratio=cfg["mlp_ratio"], dropout=cfg["dropout"],
        patch_size=cfg["patch_size"], tubelet=cfg["tubelet"], mask_ratio=cfg["mask_ratio"],
        dec_embed_dim=cfg["dec_embed_dim"], dec_depth=cfg["dec_depth"], dec_heads=cfg["dec_heads"],
        norm_pix_loss=cfg["norm_pix_loss"],
    ).to(device)

    params = model.count_parameters()
    print(f"Parameters — encoder: {params['encoder']:,}  decoder: {params['decoder']:,}  total: {params['total']:,}")
    assert params["total"] < 100_000_000, "Model exceeds 100M parameter limit"

    # ── Optimizer ─────────────────────────────────────────────────────
    # Separate weight decay: apply to weight matrices, not biases/LayerNorm
    decay_params     = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim >= 2]
    no_decay_params  = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim < 2]
    optimizer = torch.optim.AdamW([
        {"params": decay_params,    "weight_decay": cfg["weight_decay"]},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=cfg["lr"], betas=(0.9, 0.95))

    amp_dtype   = torch.bfloat16 if cfg["amp_dtype"] == "bf16" else torch.float16
    scaler      = GradScaler(enabled=(cfg["amp_dtype"] == "fp16"))
    accum_steps = max(1, cfg["target_batch"] // cfg["batch_size"])

    steps_per_epoch = (len(train_dataset) // cfg["batch_size"]) // accum_steps
    total_steps     = cfg["epochs"] * steps_per_epoch
    warmup_steps    = cfg["warmup_epochs"] * steps_per_epoch

    print(f"Effective batch: {cfg['batch_size'] * accum_steps} | Steps/epoch: {steps_per_epoch} | Total: {total_steps}")

    start_epoch   = 0
    global_step   = 0
    best_loss     = float("inf")
    epoch         = 0

    os.makedirs(cfg["out_dir"], exist_ok=True)

    if args.resume and os.path.exists(args.resume):
        print(f"Loading checkpoint: {args.resume}")
        start_epoch, global_step, best_loss = load_checkpoint(
            args.resume, model, optimizer, scaler, device,
        )

    init_wandb(cfg, args.dry_run)

    # ── Preemption handler ─────────────────────────────────────────────
    def handle_preemption(signum, frame):
        print(f"\nSIGUSR1 at epoch {epoch} step {global_step} — saving checkpoint before preemption...")
        save_checkpoint(
            f"{cfg['out_dir']}/latest.pt",
            epoch, global_step, model, optimizer, scaler, best_loss, cfg,
        )
        if not args.dry_run:
            wandb.finish()
        exit(0)

    signal.signal(signal.SIGUSR1, handle_preemption)
    signal.signal(signal.SIGTERM, handle_preemption)

    # ── Main loop ─────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0
        optimizer.zero_grad()

        # Fast-forward batches already done in this epoch (after preemption resume)
        batches_done = (global_step % max(steps_per_epoch, 1)) * accum_steps if epoch == start_epoch else 0
        if batches_done > 0:
            print(f"  Fast-forwarding: skipping {batches_done} batches in epoch {epoch + 1}")

        loader = make_epoch_loader(train_dataset, epoch, cfg["batch_size"], 4, skip_batches=batches_done)

        for step, batch in enumerate(loader):
            x = batch["frames"].to(device, non_blocking=True)  # (B, 11, 16, 224, 224)

            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                loss, _ = model(x)
                loss     = loss / accum_steps

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

                epoch_loss += loss.item() * accum_steps
                n_batches  += 1

                if global_step % cfg["log_every"] == 0:
                    print(
                        f"Epoch {epoch + 1:3d} | Step {global_step:6d} | "
                        f"LR {lr:.2e} | Loss {epoch_loss / n_batches:.4f}"
                    )
                    if not args.dry_run:
                        wandb.log({
                            "train_loss": epoch_loss / n_batches,
                            "lr": lr,
                            "epoch": epoch + 1,
                        }, step=global_step)

                if global_step % cfg["save_every_steps"] == 0:
                    save_checkpoint(
                        f"{cfg['out_dir']}/latest.pt",
                        epoch, global_step, model, optimizer, scaler, best_loss, cfg,
                    )

        # ── Validation ────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        n_val    = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["frames"].to(device)
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    loss, _ = model(x)
                val_loss += loss.item()
                n_val    += 1
        val_loss /= max(n_val, 1)

        avg_train = epoch_loss / max(n_batches, 1)
        print(
            f"\n── Epoch {epoch + 1} ──────────────────────────────\n"
            f"  Train Loss: {avg_train:.4f}\n"
            f"  Val Loss:   {val_loss:.4f}\n"
        )
        if not args.dry_run:
            wandb.log({
                "val_loss":   val_loss,
                "train_loss": avg_train,
                "epoch":      epoch + 1,
            }, step=global_step)

        save_checkpoint(
            f"{cfg['out_dir']}/latest.pt",
            epoch + 1, global_step, model, optimizer, scaler, best_loss, cfg,
        )
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(
                f"{cfg['out_dir']}/best.pt",
                epoch + 1, global_step, model, optimizer, scaler, best_loss, cfg,
            )
            print("  New best model saved!\n")

        if (epoch + 1) % cfg["save_every_epochs"] == 0:
            save_checkpoint(
                f"{cfg['out_dir']}/epoch_{epoch + 1}.pt",
                epoch + 1, global_step, model, optimizer, scaler, best_loss, cfg,
            )

        if args.dry_run:
            print("Dry run complete.")
            break

    if not args.dry_run:
        wandb.finish()
    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume",  type=str,  default=None, help="path to checkpoint")
    parser.add_argument("--dry-run", action="store_true",     help="run one epoch and exit")
    args = parser.parse_args()
    train(args, CONFIG)
