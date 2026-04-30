"""
ViT-JEPA training — vit-sarvesh
================================
Self-supervised masked patch prediction in latent space on active_matter data.

Usage:
  python train.py
  python train.py --resume /scratch/sb10583/checkpoints/vit-jepa-sarvesh/latest.pt
  python train.py --dry-run
"""

import os
import math
import signal
import argparse
import yaml
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader, Subset

from dataset import ActiveMatterDataset
from masking  import sample_block_mask
from model    import ViTJEPA


# ─────────────────────────────────────────────────────────────────────────────
# LR schedule
# ─────────────────────────────────────────────────────────────────────────────

def get_lr(step, total_steps, warmup_steps, base_lr, min_lr=1e-6):
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ─────────────────────────────────────────────────────────────────────────────
# Per-epoch DataLoader — deterministic shuffle seeded by epoch
# ─────────────────────────────────────────────────────────────────────────────

def make_epoch_loader(dataset, epoch, batch_size, num_workers, pin_memory, skip_batches=0):
    g = torch.Generator()
    g.manual_seed(epoch)
    indices = torch.randperm(len(dataset), generator=g).tolist()
    # Trim to complete batches (same effect as drop_last=True)
    n = (len(indices) // batch_size) * batch_size
    indices = indices[:n]
    # Skip already-processed batches — no data loading wasted
    if skip_batches > 0:
        indices = indices[skip_batches * batch_size:]
        print(f"  fast-forwarding: skipping {skip_batches} batches, {len(indices) // batch_size} remaining")
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=pin_memory, drop_last=False)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpointing
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(path, epoch, global_step, model, optimizer, scaler, best_val_loss):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch":         epoch,
        "global_step":   global_step,
        "model":         model.state_dict(),
        "optimizer":     optimizer.state_dict(),
        "scaler":        scaler.state_dict(),
        "best_val_loss": best_val_loss,
    }, path)
    print(f"  saved: {path}")


def load_checkpoint(path, model, optimizer, scaler, device):
    ckpt          = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    start_epoch   = ckpt["epoch"]
    global_step   = ckpt.get("global_step", 0)
    best_val_loss = ckpt.get("best_val_loss", float("inf"))
    print(f"  resumed from epoch {start_epoch}, step {global_step}, best_val_loss={best_val_loss:.4f}")
    return start_epoch, global_step, best_val_loss


# ─────────────────────────────────────────────────────────────────────────────
# WandB — persist run ID across preemptions
# ─────────────────────────────────────────────────────────────────────────────

def init_wandb(cfg, dry_run):
    if dry_run:
        return
    out_dir = cfg["checkpointing"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    id_file = os.path.join(out_dir, "wandb_run_id.txt")
    project = cfg["logging"]["wandb_project"]
    name    = cfg["logging"]["run_name"]

    if os.path.exists(id_file):
        run_id = open(id_file).read().strip()
        print(f"resuming wandb run: {run_id}")
        wandb.init(project=project, name=name, id=run_id, resume="allow", config=cfg)
    else:
        run = wandb.init(project=project, name=name, config=cfg)
        open(id_file, "w").write(run.id)
        print(f"started wandb run: {run.id}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(42)
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda    = torch.cuda.is_available()
    num_workers = 4 if use_cuda else 0
    print(f"device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    d = cfg["data"]
    train_ds = ActiveMatterDataset(
        d["data_dir"], split="train",
        stride=d["stride"], noise_std=d["noise_std"],
    )
    val_ds = ActiveMatterDataset(
        d["data_dir"], split="valid",
        stride=d["stride"], noise_std=0.0,
    )
    t = cfg["training"]
    val_loader = DataLoader(
        val_ds, batch_size=t["batch_size"], shuffle=False,
        num_workers=num_workers, pin_memory=use_cuda,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    m = cfg["model"]
    model = ViTJEPA(
        in_channels=m["in_channels"],
        embed_dim=m["embed_dim"],
        depth=m["depth"],
        num_heads=m["num_heads"],
        mlp_ratio=m["mlp_ratio"],
        patch_size=m["patch_size"],
        tubelet=m["tubelet"],
        num_frames=m["num_frames"],
        pred_dim=m["pred_dim"],
        pred_depth=m["pred_depth"],
        pred_heads=m["pred_heads"],
    ).to(device)

    params = model.count_parameters()
    print(
        f"patch_embed: {params['patch_embed']:,}  encoder: {params['encoder']:,}  "
        f"predictor: {params['predictor']:,}  total: {params['total']:,}"
    )

    # Token grid dimensions derived from model
    num_t = m["num_frames"] // m["tubelet"]               # 8
    num_h = d.get("crop_size", 224) // m["patch_size"]    # 14
    num_w = num_h                                          # 14

    # ── Steps per epoch (from dataset size, not loader) ───────────────────────
    samples_per_epoch = (len(train_ds) // t["batch_size"]) * t["batch_size"]
    accum_steps  = max(1, t["target_batch"] // t["batch_size"])
    steps_per_ep = (samples_per_epoch // t["batch_size"]) // accum_steps
    total_steps  = t["epochs"] * steps_per_ep
    warmup_steps = t["warmup_epochs"] * steps_per_ep
    print(f"accum: {accum_steps}  steps/epoch: {steps_per_ep}  total: {total_steps}")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=t["lr"], betas=(0.9, 0.95), weight_decay=t["weight_decay"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    start_epoch   = 0
    best_val_loss = float("inf")
    global_step   = 0
    epoch         = 0

    os.makedirs(cfg["checkpointing"]["out_dir"], exist_ok=True)

    if args.resume and os.path.exists(args.resume):
        print(f"loading checkpoint: {args.resume}")
        start_epoch, global_step, best_val_loss = load_checkpoint(
            args.resume, model, optimizer, scaler, device,
        )

    init_wandb(cfg, args.dry_run)

    # ── Preemption handler ────────────────────────────────────────────────────
    def handle_preemption(signum, frame):
        print(f"\nSIGUSR1 at epoch {epoch}, step {global_step} — saving before preemption...")
        save_checkpoint(
            f"{cfg['checkpointing']['out_dir']}/latest.pt",
            epoch, global_step, model, optimizer, scaler, best_val_loss,
        )
        if not args.dry_run:
            wandb.finish()
        exit(0)

    signal.signal(signal.SIGUSR1, handle_preemption)

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, t["epochs"]):
        model.train()
        optimizer.zero_grad()
        epoch_loss = 0.0
        n_batches  = 0

        # Compute how many batches were already done in this epoch.
        # batches_done is always a multiple of accum_steps since we only
        # checkpoint at optimizer-step boundaries.
        batches_done = (global_step % steps_per_ep) * accum_steps if epoch == start_epoch else 0
        epoch_loader = make_epoch_loader(
            train_ds, epoch, t["batch_size"], num_workers, use_cuda,
            skip_batches=batches_done,
        )

        for step, batch in enumerate(epoch_loader):
            frames = batch["frames"].to(device, non_blocking=True)

            ctx_idx, tgt_idx = sample_block_mask(
                num_t, num_h, num_w,
                target_ratio=t["mask_ratio"],
                num_blocks=t["num_mask_blocks"],
            )
            ctx_idx = ctx_idx.to(device)
            tgt_idx = tgt_idx.to(device)

            with torch.autocast(
                device_type="cuda" if use_cuda else "cpu",
                dtype=torch.bfloat16 if use_cuda else torch.float32,
            ):
                loss, metrics = model(frames, ctx_idx, tgt_idx)
                loss = loss / accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0:
                lr = get_lr(global_step, total_steps, warmup_steps, t["lr"])
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    t["grad_clip"],
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # EMA update: tau cosine-annealed 0.996 → 0.9999
                progress = global_step / max(total_steps, 1)
                tau = 1.0 - (1.0 - cfg["ema"]["momentum_start"]) * (math.cos(math.pi * progress) + 1) / 2
                with torch.no_grad():
                    for p_o, p_t in zip(
                        model.online_encoder.parameters(),
                        model.target_encoder.parameters(),
                    ):
                        p_t.data.mul_(tau).add_((1.0 - tau) * p_o.data)

                global_step += 1
                epoch_loss  += metrics["loss"]
                n_batches   += 1

                if global_step % cfg["checkpointing"]["save_every_steps"] == 0:
                    save_checkpoint(
                        f"{cfg['checkpointing']['out_dir']}/latest.pt",
                        epoch, global_step, model, optimizer, scaler, best_val_loss,
                    )

                if global_step % cfg["logging"]["log_every"] == 0:
                    print(
                        f"ep {epoch+1:3d} | step {global_step:5d} | lr {lr:.2e} | "
                        f"tau {tau:.4f} | loss {metrics['loss']:.4f} | "
                        f"emb_std {metrics['embedding_std']:.4f}"
                    )
                    if not args.dry_run:
                        wandb.log({**metrics, "lr": lr, "ema_tau": tau, "epoch": epoch + 1}, step=global_step)

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        n_val    = 0
        with torch.no_grad():
            for batch in val_loader:
                frames = batch["frames"].to(device)
                ctx_idx, tgt_idx = sample_block_mask(num_t, num_h, num_w,
                                                     target_ratio=t["mask_ratio"],
                                                     num_blocks=t["num_mask_blocks"])
                ctx_idx, tgt_idx = ctx_idx.to(device), tgt_idx.to(device)
                with torch.autocast(
                    device_type="cuda" if use_cuda else "cpu",
                    dtype=torch.bfloat16 if use_cuda else torch.float32,
                ):
                    _, m_val = model(frames, ctx_idx, tgt_idx)
                val_loss += m_val["loss"]
                n_val    += 1
        val_loss /= max(n_val, 1)

        # ── Collapse check ─────────────────────────────────────────────────────
        with torch.no_grad():
            probe_batch = next(iter(val_loader))
            probe_frames = probe_batch["frames"].to(device)
            emb_std = model.encode(probe_frames).std(dim=0).mean().item()

        avg_train = epoch_loss / max(n_batches, 1)
        collapse  = emb_std < 0.1
        print(
            f"\n── epoch {epoch+1} ──\n"
            f"  train loss: {avg_train:.4f}  val loss: {val_loss:.4f}\n"
            f"  emb std: {emb_std:.4f}  {'COLLAPSE RISK' if collapse else 'healthy'}\n"
        )

        if not args.dry_run:
            wandb.log({
                "val_loss": val_loss, "train_loss": avg_train,
                "embedding_std": emb_std, "epoch": epoch + 1,
            }, step=global_step)

        # ── Checkpointing ──────────────────────────────────────────────────────
        ckpt_dir = cfg["checkpointing"]["out_dir"]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(f"{ckpt_dir}/best.pt", epoch + 1, global_step, model, optimizer, scaler, best_val_loss)
            print("  new best model\n")

        save_checkpoint(f"{ckpt_dir}/latest.pt", epoch + 1, global_step, model, optimizer, scaler, best_val_loss)

        if (epoch + 1) % cfg["checkpointing"]["save_every"] == 0:
            save_checkpoint(f"{ckpt_dir}/epoch_{epoch+1}.pt", epoch + 1, global_step, model, optimizer, scaler, best_val_loss)

        if args.dry_run:
            print("dry run complete.")
            break

    if not args.dry_run:
        wandb.finish()
    print("training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  type=str, default="config.yaml")
    parser.add_argument("--resume",  type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    main(parser.parse_args())
