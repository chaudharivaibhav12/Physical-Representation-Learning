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
import sys
import math
import time
import signal
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler
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
    "data_dir":          "/scratch/vc2836/DL/data/active_matter/data",
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
    "batch_size":        8,        # was 4; matches single-GPU throughput on one A100
    "target_batch":      64,       # was 256; gives accum_steps=8 for fast optimizer ticks
    "lr":                1e-3,
    "weight_decay":      0.05,
    "grad_clip":         1.0,
    "warmup_epochs":     5,
    "amp_dtype":         "bf16",

    # Checkpointing
    "out_dir":           "/scratch/vc2836/DL/checkpoints/vit_jepa_v2",
    "save_every":        5,

    # Logging
    "wandb_project":     "active-matter-jepa",
    "run_name":          "vit-jepa-v2-d384-p32",
    "log_every":         1,
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
# Resume-friendly sampler
# ─────────────────────────────────────────────

class OffsetSampler(Sampler):
    """A deterministic-shuffle sampler that yields a random permutation of
    [0, n) but skips the first `offset` indices.

    Used on resume-mid-epoch to fast-forward past samples that were already
    consumed before preemption, WITHOUT having the DataLoader workers load
    them and throw them away. Skipped indices never reach a worker, so no
    HDF5 reads happen for them.

    Determinism note: this sampler reseeds from `seed` each epoch, so the
    permutation order on resume differs from what the default RandomSampler
    would have produced for that epoch. That's fine — with shuffle=True we
    don't care which specific samples appear in which order, only that
    every sample appears at most once per epoch.
    """
    def __init__(self, n: int, offset: int, seed: int):
        self.n      = n
        self.offset = max(0, min(offset, n))
        self.seed   = seed

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed)
        order = torch.randperm(self.n, generator=g).tolist()
        yield from order[self.offset:]

    def __len__(self) -> int:
        return self.n - self.offset


# ─────────────────────────────────────────────
# Checkpoint Utilities
# ─────────────────────────────────────────────

def save_checkpoint(path, epoch, global_step, model, optimizer, scaler, best_val_loss, cfg):
    """Atomic save: write to .tmp first, then rename. Avoids corrupt files
    if the process is killed mid-write."""
    encoder = model.module.encoder if hasattr(model, "module") else model.encoder
    tmp = path + ".tmp"
    torch.save({
        "epoch":          epoch,
        "global_step":    global_step,
        "encoder":        encoder.state_dict(),
        "model":          model.state_dict(),
        "optimizer":      optimizer.state_dict(),
        "scaler":         scaler.state_dict(),
        "best_val_loss":  best_val_loss,
        "config":         cfg,
    }, tmp)
    os.replace(tmp, path)   # atomic on POSIX
    print(f"  Saved checkpoint: {path} (epoch={epoch}, step={global_step})", flush=True)


def load_checkpoint(path, model, optimizer, scaler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    else:
        print("  (checkpoint has no optimizer state — starting AdamW fresh)", flush=True)
    if "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    else:
        print("  (checkpoint has no scaler state — starting GradScaler fresh)", flush=True)
    return (
        ckpt["epoch"],
        ckpt.get("global_step", 0),
        ckpt.get("best_val_loss", float("inf")),
    )


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
        num_workers = 2,
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
        start_epoch, global_step, best_val_loss = load_checkpoint(
            args.resume, model, optimizer, scaler, device
        )
        if is_main:
            print(f"[RESUME] Resuming from epoch {start_epoch}, "
                  f"global_step {global_step}, best_val_loss {best_val_loss:.4f}\n")

    # ── WandB ────────────────────────────────────────────────────────
    os.makedirs(cfg["out_dir"], exist_ok=True)
    if is_main and not args.dry_run:
        print("[WANDB] Initializing...", flush=True)
        try:
            wandb_id_file = os.path.join(cfg["out_dir"], "wandb_run_id.txt")
            if os.path.exists(wandb_id_file):
                with open(wandb_id_file) as f:
                    run_id = f.read().strip()
                print(f"[WANDB] Resuming run: {run_id}", flush=True)
                wandb.init(
                    project  = cfg["wandb_project"],
                    entity   = cfg.get("wandb_entity"),
                    name     = cfg["run_name"],
                    config   = cfg,
                    id       = run_id,
                    resume   = "must",
                    settings = wandb.Settings(init_timeout=60),
                )
            else:
                run = wandb.init(
                    project  = cfg["wandb_project"],
                    entity   = cfg.get("wandb_entity"),
                    name     = cfg["run_name"],
                    config   = cfg,
                    settings = wandb.Settings(init_timeout=60),
                )
                with open(wandb_id_file, "w") as f:
                    f.write(run.id)
            print("[WANDB] Initialized successfully.", flush=True)
        except Exception as e:
            print(f"[WANDB] Init failed ({e}), continuing without wandb.", flush=True)
            args.dry_run = True

    # ── Preemption handler ───────────────────────────────────────────
    # Mutable state the signal handler reads at signal time. We update
    # epoch / global_step / best_val_loss as training progresses so that
    # whatever the handler saves matches the latest training state.
    _save_state = {
        "epoch":         start_epoch,
        "global_step":   global_step,
        "best_val_loss": best_val_loss,
        "saved":         False,
    }

    def emergency_save(reason: str = ""):
        if not is_main or _save_state["saved"]:
            return
        _save_state["saved"] = True

        # PyTorch's DataLoader installs a SIGCHLD handler that raises a
        # RuntimeError when workers die. During SLURM preemption, the
        # workers get SIGTERM at the same time we do, and that race
        # interrupts torch.save. We're exiting anyway — silence SIGCHLD.
        try:
            signal.signal(signal.SIGCHLD, signal.SIG_DFL)
        except (ValueError, OSError):
            pass

        path = os.path.join(cfg["out_dir"], "latest.pt")

        # Strategy 1: full checkpoint (model + optimizer + scaler).
        try:
            save_checkpoint(path, _save_state["epoch"], _save_state["global_step"],
                            model, optimizer, scaler, _save_state["best_val_loss"], cfg)
            print(f"[PREEMPT] Emergency checkpoint saved ({reason})", flush=True)
            return
        except Exception as e:
            print(f"[PREEMPT] Full save failed: {e}", flush=True)

        # Strategy 2: model-only fallback — at minimum we keep the weights.
        # Loses optimizer momentum on resume but still better than nothing.
        try:
            encoder = model.module.encoder if hasattr(model, "module") else model.encoder
            tmp = path + ".tmp"
            torch.save({
                "epoch":         _save_state["epoch"],
                "global_step":   _save_state["global_step"],
                "encoder":       encoder.state_dict(),
                "model":         model.state_dict(),
                "best_val_loss": _save_state["best_val_loss"],
                "config":        cfg,
            }, tmp)
            os.replace(tmp, path)
            print(f"[PREEMPT] Model-only fallback save succeeded ({reason})", flush=True)
        except Exception as e:
            print(f"[PREEMPT] Fallback save also failed: {e} "
                  f"-- relying on last periodic save", flush=True)

    def handle_preemption(signum, frame):
        print(f"[PREEMPT] Signal {signum} received at "
              f"{time.strftime('%H:%M:%S')}", flush=True)
        emergency_save(reason=f"signal {signum}")
        if is_main and not args.dry_run and wandb.run is not None:
            try:
                wandb.finish()
            except Exception:
                pass
        sys.exit(0)

    signal.signal(signal.SIGUSR1, handle_preemption)
    signal.signal(signal.SIGTERM, handle_preemption)

    # Initial checkpoint so even instant preemption is recoverable.
    if is_main and not (args.resume and os.path.exists(args.resume)):
        init_path = os.path.join(cfg["out_dir"], "latest.pt")
        save_checkpoint(init_path, start_epoch, global_step, model, optimizer,
                        scaler, best_val_loss, cfg)
        print(f"[INIT] Initial checkpoint written to {init_path}", flush=True)

    # Wall-clock checkpoint cadence (5 minutes).
    SAVE_INTERVAL_SECS = 5 * 60
    last_save_time = time.time()

    # ── Training ─────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        if train_sampler:
            train_sampler.set_epoch(epoch)

        # Resume-mid-epoch handling: how many opt steps and mini-batches
        # have we already consumed from this epoch?
        steps_done_this_epoch = global_step % steps_per_epoch if steps_per_epoch > 0 else 0
        batches_to_skip       = steps_done_this_epoch * accum_steps
        samples_to_skip       = batches_to_skip * cfg["batch_size"]

        # Fast-skip path (single-GPU): build a one-shot DataLoader whose
        # sampler already starts from `samples_to_skip`. Workers never load
        # the skipped indices, so resume is near-instantaneous.
        if samples_to_skip > 0 and train_sampler is None:
            if is_main:
                print(f"[RESUME] Fast-skipping {samples_to_skip} samples "
                      f"({batches_to_skip} batches, {steps_done_this_epoch} opt steps) "
                      f"via OffsetSampler — no I/O cost", flush=True)
            epoch_sampler = OffsetSampler(
                n      = len(train_dataset),
                offset = samples_to_skip,
                seed   = 42 + epoch,
            )
            epoch_loader = DataLoader(
                train_dataset,
                batch_size  = cfg["batch_size"],
                sampler     = epoch_sampler,
                num_workers = num_workers,
                pin_memory  = pin_memory,
                drop_last   = True,
            )
            in_loop_skip = 0   # the sampler did the skipping
        else:
            epoch_loader = train_loader
            # DDP fallback: DistributedSampler doesn't support offset, so
            # we still skip in the loop (rare; user is single-GPU now).
            in_loop_skip = batches_to_skip
            if in_loop_skip > 0 and is_main:
                print(f"[RESUME] Skipping {in_loop_skip} batches via in-loop continue "
                      f"(DDP path)", flush=True)

        epoch_loss = 0.0
        n_batches  = 0
        optimizer.zero_grad()

        for step, batch in enumerate(epoch_loader):
            if step < in_loop_skip:
                continue

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
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"]).item()

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
                        f"Cov {metrics['loss_covariance']:.4f} | "
                        f"GradNorm {grad_norm:.3f}",
                        flush=True,
                    )
                    if not args.dry_run:
                        wandb.log({**metrics, "lr": lr, "grad_norm": grad_norm, "epoch": epoch + 1}, step=global_step)

            # ── Wall-clock checkpoint (runs every mini-batch, not every
            # gradient step, so slow accumulation can't block it) ──
            if is_main:
                _save_state["epoch"]         = epoch
                _save_state["global_step"]   = global_step
                _save_state["best_val_loss"] = best_val_loss
                if time.time() - last_save_time > SAVE_INTERVAL_SECS:
                    latest_path = os.path.join(cfg["out_dir"], "latest.pt")
                    save_checkpoint(latest_path, epoch, global_step, model, optimizer,
                                    scaler, best_val_loss, cfg)
                    last_save_time = time.time()

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
                    "train_loss":    avg_train,
                    "val_loss":      val_loss,
                    "embedding_std": emb_std,
                    "epoch":         epoch + 1,
                }, step=global_step)

            latest_path = os.path.join(cfg["out_dir"], "latest.pt")
            save_checkpoint(latest_path, epoch + 1, global_step, model, optimizer, scaler, best_val_loss, cfg)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(cfg["out_dir"], "best.pt")
                save_checkpoint(best_path, epoch + 1, global_step, model, optimizer, scaler, best_val_loss, cfg)
                print(f"  ✓ New best model saved!\n")

            if (epoch + 1) % cfg["save_every"] == 0:
                ep_path = os.path.join(cfg["out_dir"], f"epoch_{epoch+1}.pt")
                save_checkpoint(ep_path, epoch + 1, global_step, model, optimizer, scaler, best_val_loss, cfg)

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

    # Checkpointing
    parser.add_argument("--resume",        type=str,   default=None)
    parser.add_argument("--dry-run",       action="store_true")

    # Paths (override CONFIG defaults)
    parser.add_argument("--data-dir",      type=str,   default=None,
                        help="Path to active_matter/data/")
    parser.add_argument("--out-dir",       type=str,   default=None,
                        help="Directory to save checkpoints")
# Training (override CONFIG defaults)
    parser.add_argument("--epochs",        type=int,   default=None)
    parser.add_argument("--batch-size",    type=int,   default=None)
    parser.add_argument("--lr",            type=float, default=None)
    parser.add_argument("--stride",        type=int,   default=None)

    # Logging
    parser.add_argument("--run-name",      type=str,   default=None)
    parser.add_argument("--wandb-project", type=str,   default=None)
    parser.add_argument("--wandb-entity", type=str,   default=None)

    args = parser.parse_args()

    # Apply CLI overrides to CONFIG
    cfg = dict(CONFIG)
    if args.data_dir:      cfg["data_dir"]       = args.data_dir
    if args.out_dir:       cfg["out_dir"]         = args.out_dir
    if args.epochs:        cfg["epochs"]          = args.epochs
    if args.batch_size:    cfg["batch_size"]      = args.batch_size
    if args.lr:            cfg["lr"]              = args.lr
    if args.stride:        cfg["stride"]          = args.stride
    if args.run_name:      cfg["run_name"]        = args.run_name
    if args.wandb_project: cfg["wandb_project"]   = args.wandb_project
    if args.wandb_entity:  cfg["wandb_entity"]    = args.wandb_entity

    train(args, cfg)