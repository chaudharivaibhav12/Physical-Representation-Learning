"""
JEPA pretraining script — Option A (conv-JEPA baseline).
FULLY ENHANCED VERSION WITH ALL FEATURES

Features:
✅ Professional W&B logging (24+ metrics)
✅ Custom run names (via run_name parameter)
✅ Experiment groups (via experiment_group parameter)
✅ Auto-generated names (from hyperparameters)
✅ Multiple W&B projects (via wandb_project parameter)
✅ Loss statistics (std, min, max)
✅ Gradient norm monitoring
✅ Per-batch timing
✅ Professional tags and notes
✅ Weight/gradient histograms

Training recipe (matches the reference paper for active_matter):
    - AdamW optimizer, lr=1e-3, weight_decay=0.05, betas=(0.9, 0.95)
    - Cosine LR schedule with 2-epoch warmup, min_lr=1e-6
    - VICReg loss (sim=2, std=40, cov=2) on dense (C, H, W) embeddings
    - num_frames=16 context + 16 target, non-overlapping
    - Batch size 8 per device, target global batch size 256 via grad accum

Usage Examples:
    # Example 1: Custom names
    python train.py --config config.yaml \
      experiment_group="lr-study" \
      run_name="lr-1e3"

    # Example 2: Auto-generated names
    python train.py --config config.yaml \
      experiment_group="quick-tests" \
      train.lr=1e-3

    # Example 3: Different projects
    python train.py --config config.yaml \
      wandb_project="convjepa-lr-studies" \
      experiment_group="main" \
      run_name="lr-1e3"

    # Example 4: Production run
    python train.py --config config.yaml \
      wandb_project="convjepa-production" \
      experiment_group="production" \
      run_name="final-model-v1.0" \
      notes="Submitted to paper"

    # Example 5: Multiple seeds
    for SEED in 42 43 44; do
      python train.py --config config.yaml \
        experiment_group="final-experiments" \
        run_name="final-seed$SEED" \
        seed=$SEED
    done
"""

from __future__ import annotations

import argparse
import gc
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False

import sys
_DATASET_DIR = os.environ.get("JEPA_DATASET_DIR")
if _DATASET_DIR:
    sys.path.insert(0, _DATASET_DIR)

from model import build_jepa, count_params
from loss import vicreg_loss
from scheduler import CosineWarmupLR


# ============================================================================
# Distributed setup
# ============================================================================

def ddp_setup() -> tuple[int, int, int]:
    """Initialize distributed process group if run under torchrun.

    Returns:
        (rank, world_size, local_rank). Falls back to (0, 1, 0) for single GPU.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return rank, world_size, local_rank
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    return 0, 1, 0


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def rprint(msg: str):
    """Print only from rank 0."""
    if is_main_process():
        print(msg, flush=True)


def reduce_scalar(x: torch.Tensor) -> torch.Tensor:
    """All-reduce a scalar tensor across DDP workers, returning the mean."""
    if dist.is_initialized():
        x = x.detach().clone()
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        x = x / dist.get_world_size()
    return x


# ============================================================================
# Gradient monitoring
# ============================================================================

def compute_gradient_norm(optimizer: torch.optim.Optimizer) -> float:
    """
    Compute total gradient norm across all parameters.
    
    Useful for detecting exploding/vanishing gradients.
    Returns a float (scalar).
    """
    total_norm = 0.0
    for param_group in optimizer.param_groups:
        for p in param_group['params']:
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
    
    total_norm = total_norm ** 0.5
    return total_norm


# ============================================================================
# W&B Configuration (Approach C: Hybrid)
# ============================================================================

def generate_run_name(cfg, custom_name: str = None) -> str:
    """
    Generate a run name from config or use custom name.
    
    If custom_name is provided, uses that.
    Otherwise, auto-generates from hyperparameters.
    
    Examples:
        Custom: "my-important-run"
        Auto-generated: "lr1e-03-bs8-seed42"
    """
    if custom_name:
        return custom_name
    
    # Auto-generate from hyperparameters
    lr = cfg.train.lr
    bs = cfg.train.batch_size
    seed = cfg.get('seed', 42)
    
    lr_str = f"{lr:.0e}".replace("+", "")
    run_name = f"lr{lr_str}-bs{bs}-seed{seed}"
    return run_name


def get_wandb_config(cfg) -> dict:
    """
    Extract and generate W&B configuration from OmegaConf (Approach C: Hybrid).
    
    Provides sensible defaults while allowing full customization.
    
    Returns:
        Dict with keys: project, name, group, tags, notes
    """
    # 1. W&B Project (can override with wandb_project param)
    wandb_project = cfg.get("wandb_project", "physics-jepa-experiments")
    
    # 2. Run name (custom or auto-generated)
    run_name = generate_run_name(cfg, cfg.get("run_name", None))
    
    # 3. Experiment group (for organizing runs in same project)
    experiment_group = cfg.get("experiment_group", "default")
    
    # 4. Build tags for filtering
    # Use .get() with default because some configs may not define dataset.name explicitly.
    dataset_name = cfg.dataset.get("name", cfg.dataset.get("class_name", "unknown"))
    tags = [
        "convjepa",
        dataset_name,
        f"lr_{cfg.train.lr:.0e}",
        f"bs_{cfg.train.batch_size}",
        f"epochs_{cfg.train.num_epochs}",
    ]
    
    # Optional: Add experiment type if specified
    if cfg.get("experiment_type"):
        tags.append(cfg.experiment_type)
    
    # 5. Notes (visible in W&B, useful for documentation)
    notes = cfg.get("notes", f"Experiment: {experiment_group}")
    
    return {
        "project": wandb_project,
        "name": run_name,
        "group": experiment_group,
        "tags": tags,
        "notes": notes,
    }


# ============================================================================
# Data
# ============================================================================

def build_dataloaders(cfg, rank: int, world_size: int) -> tuple[DataLoader, DataLoader]:
    """
    Construct train/val dataloaders using the user's ActiveMatterDataset.

    We import the dataset class lazily so this script does not hard-couple to
    a specific module layout. The config's `dataset.module` and
    `dataset.class_name` tell us where to import from.
    """
    import importlib

    module = importlib.import_module(cfg.dataset.module)
    DatasetCls = getattr(module, cfg.dataset.class_name)

    # The dataset class is expected to take (split, **kwargs).
    ds_kwargs = OmegaConf.to_container(cfg.dataset.get("kwargs", {}), resolve=True)

    train_ds = DatasetCls(split="train", **ds_kwargs)
    val_ds = DatasetCls(split="val", **ds_kwargs)

    rprint(f"[data] train size: {len(train_ds)} | val size: {len(val_ds)}")

    if world_size > 1:
        train_sampler = DistributedSampler(train_ds, rank=rank, num_replicas=world_size,
                                           shuffle=True, drop_last=True)
        val_sampler = DistributedSampler(val_ds, rank=rank, num_replicas=world_size,
                                         shuffle=False, drop_last=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=cfg.train.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=cfg.train.get("num_workers", 4),
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
    )
    return train_loader, val_loader


# ============================================================================
# Trainer
# ============================================================================

class JEPATrainer:

    def __init__(self, cfg, rank: int, world_size: int, local_rank: int):
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        self.out_path = Path(cfg.out_path) / cfg.run_name
        if is_main_process():
            self.out_path.mkdir(parents=True, exist_ok=True)

        # Data
        self.train_loader, self.val_loader = build_dataloaders(cfg, rank, world_size)

        # Model
        encoder, predictor = build_jepa(cfg)
        encoder = encoder.to(self.device)
        predictor = predictor.to(self.device)
        n_enc = count_params(encoder)
        n_pred = count_params(predictor)
        rprint(f"[model] encoder params: {n_enc:,} | predictor params: {n_pred:,} "
               f"| total: {n_enc + n_pred:,}")

        if world_size > 1:
            encoder = DDP(encoder, device_ids=[local_rank], find_unused_parameters=False)
            predictor = DDP(predictor, device_ids=[local_rank], find_unused_parameters=False)
        self.encoder = encoder
        self.predictor = predictor

        # Optimizer
        params = list(encoder.parameters()) + list(predictor.parameters())
        self.optimizer = torch.optim.AdamW(
            params,
            lr=cfg.train.lr,
            betas=tuple(cfg.train.get("betas", (0.9, 0.95))),
            weight_decay=cfg.train.weight_decay,
        )

        # Gradient accumulation to reach target global batch.
        effective_batch = cfg.train.batch_size * world_size
        target_global = cfg.train.get("target_global_batch_size", effective_batch)
        self.grad_accum = max(1, target_global // effective_batch)
        rprint(f"[train] effective batch: {effective_batch} | "
               f"target global: {target_global} | grad_accum: {self.grad_accum}")

        # LR schedule — one schedule step per optimizer step (not per micro-step).
        steps_per_epoch = len(self.train_loader) // self.grad_accum
        total_steps = cfg.train.num_epochs * steps_per_epoch
        warmup_steps = cfg.train.get("lr_scheduler_warmup_epochs", 0) * steps_per_epoch
        self.scheduler = CosineWarmupLR(
            self.optimizer,
            base_lr=cfg.train.lr,
            min_lr=cfg.train.get("min_lr", 1e-6),
            total_steps=total_steps,
            warmup_steps=warmup_steps,
        )
        self.total_steps = total_steps
        self.steps_per_epoch = steps_per_epoch

        # AMP
        self.use_amp = cfg.train.get("use_amp", True) and torch.cuda.is_available()
        self.amp_dtype = torch.bfloat16 if cfg.train.get("amp_dtype", "bf16") == "bf16" else torch.float16
        self.scaler = (torch.amp.GradScaler("cuda")
                       if self.use_amp and self.amp_dtype == torch.float16 else None)

        # State
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")

        # Logging
        self.log_every = cfg.train.get("report_every", 50)
        if is_main_process() and _HAS_WANDB and cfg.get("use_wandb", False) and not cfg.get("dry_run", False):
            # Get W&B configuration (Approach C: Hybrid)
            wb_config = get_wandb_config(cfg)
            
            # Pull team entity from config (required for NYU accounts where
            # personal entities are disabled).
            wb_entity = cfg.get("wandb_entity", None)
            
            # Print what we're logging
            rprint(f"[wandb] entity='{wb_entity}' project='{wb_config['project']}' "
                   f"group='{wb_config['group']}' run='{wb_config['name']}'")
            
            # Initialize W&B with custom configuration
            wandb.init(
                entity=wb_entity,
                project=wb_config["project"],
                name=wb_config["name"],
                group=wb_config["group"],
                config=OmegaConf.to_container(cfg, resolve=True),
                tags=wb_config["tags"],
                job_type="train",
                notes=wb_config["notes"],
            )
            self.wandb_run = wandb.run
            
            # Watch models for weight/gradient histograms
            wandb.watch(
                self.encoder,
                log="all",
                log_freq=max(1, self.log_every // 5),
                idx=0,
            )
            wandb.watch(
                self.predictor,
                log="all",
                log_freq=max(1, self.log_every // 5),
                idx=1,
            )
        else:
            self.wandb_run = None

    # --- Forward / loss --------------------------------------------------------

    def _forward_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        ctx = batch["context"].to(self.device, non_blocking=True)
        tgt = batch["target"].to(self.device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
            ctx_embed = self.encoder(ctx)
            tgt_embed = self.encoder(tgt)
            pred = self.predictor(ctx_embed)
            loss_dict = vicreg_loss(
                pred, tgt_embed,
                sim_coeff=self.cfg.train.sim_coeff,
                std_coeff=self.cfg.train.std_coeff,
                cov_coeff=self.cfg.train.cov_coeff,
                n_chunks=self.cfg.train.get("vicreg_chunks", 5),
            )
        return loss_dict

    # --- Training loop ---------------------------------------------------------

    def train(self):
        start_epoch = self.epoch
        for epoch in range(start_epoch, self.cfg.train.num_epochs):
            self.epoch = epoch
            if hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(epoch)

            self._train_one_epoch(epoch)
            val_metrics = self._validate(epoch)

            if is_main_process():
                avg_val = val_metrics.get("val/loss", float("inf"))
                is_best = avg_val < self.best_val_loss
                if is_best:
                    self.best_val_loss = avg_val

                save_every = self.cfg.train.get("save_every", 1)
                if (epoch + 1) % save_every == 0 or is_best:
                    self._save_checkpoint(epoch, is_best=is_best)

            if dist.is_initialized():
                dist.barrier()

        rprint(f"[train] done. best val loss: {self.best_val_loss:.4f}")
        rprint(f"[train] checkpoints in: {self.out_path}")

    def _train_one_epoch(self, epoch: int):
        self.encoder.train()
        self.predictor.train()
        self.optimizer.zero_grad(set_to_none=True)

        t0 = time.time()
        running: Dict[str, list] = defaultdict(list)
        micro_in_accum = 0
        batch_times = []  # Track batch processing time

        for step, batch in enumerate(self.train_loader):
            batch_start = time.time()
            
            loss_dict = self._forward_loss(batch)
            loss = loss_dict["loss"] / self.grad_accum

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Accumulate logging stats
            for k, v in loss_dict.items():
                running[k].append(v.detach())

            micro_in_accum += 1
            batch_times.append(time.time() - batch_start)
            is_update_step = (micro_in_accum == self.grad_accum) or (step + 1 == len(self.train_loader))

            if is_update_step:
                if self.cfg.train.get("grad_clip", 0) > 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for g in self.optimizer.param_groups for p in g["params"]],
                        self.cfg.train.grad_clip,
                    )

                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                self.global_step += 1
                micro_in_accum = 0

                # Logging
                if self.global_step % self.log_every == 0:
                    avg_batch_time_ms = (sum(batch_times) / len(batch_times) * 1000
                                        if batch_times else 0)
                    self._log_train(running, epoch, t0, batch_time_ms=avg_batch_time_ms)
                    running.clear()
                    batch_times = []
                    t0 = time.time()

        # Flush any remaining
        if running:
            avg_batch_time_ms = (sum(batch_times) / len(batch_times) * 1000
                                if batch_times else 0)
            self._log_train(running, epoch, t0, batch_time_ms=avg_batch_time_ms)

    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict[str, float]:
        self.encoder.eval()
        self.predictor.eval()

        agg: Dict[str, list] = defaultdict(list)
        val_steps = self.cfg.train.get("val_steps", None)

        for i, batch in enumerate(self.val_loader):
            if val_steps is not None and i >= val_steps:
                break
            loss_dict = self._forward_loss(batch)
            for k, v in loss_dict.items():
                agg[k].append(v.detach())

        # Reduce across ranks
        reduced: Dict[str, float] = {}
        for k, vals in agg.items():
            mean = torch.stack(vals).mean()
            mean = reduce_scalar(mean)
            reduced[f"val/{k}"] = mean.item()

        torch.cuda.empty_cache()
        gc.collect()

        if is_main_process():
            msg = " | ".join(f"{k}={v:.4f}" for k, v in reduced.items())
            rprint(f"[val  ] epoch {epoch} | {msg}")
            if self.wandb_run is not None:
                self.wandb_run.log({**reduced, "epoch": epoch}, step=self.global_step)

        return reduced

    def _log_train(self, running: Dict[str, list], epoch: int, t0: float, 
                   batch_time_ms: float = 0.0):
        """Log training metrics to W&B and console with enhanced statistics."""
        metrics = {}
        
        # Enhanced loss statistics: mean, std, min, max
        for k, vals in running.items():
            vals_stacked = torch.stack(vals)
            
            # Mean
            m = vals_stacked.mean()
            m = reduce_scalar(m)
            metrics[f"train/{k}"] = m.item()
            
            # Standard deviation
            m_std = vals_stacked.std()
            m_std = reduce_scalar(m_std)
            metrics[f"train/{k}_std"] = m_std.item()
            
            # Min/Max
            m_min = vals_stacked.min()
            m_min = reduce_scalar(m_min)
            metrics[f"train/{k}_min"] = m_min.item()
            
            m_max = vals_stacked.max()
            m_max = reduce_scalar(m_max)
            metrics[f"train/{k}_max"] = m_max.item()

        metrics["train/lr"] = self.scheduler.get_last_lr()
        metrics["epoch"] = epoch
        steps_per_sec = self.log_every / max(1e-6, time.time() - t0)
        metrics["train/steps_per_sec"] = steps_per_sec
        
        # Gradient norm
        grad_norm = compute_gradient_norm(self.optimizer)
        metrics["train/grad_norm"] = grad_norm
        
        # Batch timing
        if batch_time_ms > 0:
            metrics["train/batch_time_ms"] = batch_time_ms

        if is_main_process():
            msg = f"[train] epoch {epoch} step {self.global_step}/{self.total_steps} | "
            msg += f"loss={metrics['train/loss']:.4f} | "
            msg += f"repr={metrics['train/repr_loss']:.4f} "
            msg += f"std={metrics['train/std_loss']:.4f} "
            msg += f"cov={metrics['train/cov_loss']:.4f} | "
            msg += f"lr={metrics['train/lr']:.2e} | "
            msg += f"grad={grad_norm:.2e} | "
            msg += f"{steps_per_sec:.2f} step/s"
            
            if batch_time_ms > 0:
                msg += f" | batch={batch_time_ms:.1f}ms"
            
            rprint(msg)
            if self.wandb_run is not None:
                self.wandb_run.log(metrics, step=self.global_step)

    # --- Checkpointing ---------------------------------------------------------

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        def _unwrap(m):
            return m.module if isinstance(m, DDP) else m

        ckpt = {
            "epoch": epoch,
            "global_step": self.global_step,
            "encoder": _unwrap(self.encoder).state_dict(),
            "predictor": _unwrap(self.predictor).state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": OmegaConf.to_container(self.cfg, resolve=True),
        }

        path = self.out_path / f"epoch_{epoch}.pt"
        torch.save(ckpt, path)
        rprint(f"[ckpt ] saved {path}")
        
        # Log best checkpoint as W&B artifact
        if is_best and self.wandb_run is not None:
            artifact = wandb.Artifact(
                name="jepa-best-checkpoint",
                type="model",
                description=f"Best checkpoint at epoch {epoch} (val_loss={self.best_val_loss:.4f})",
            )
            artifact.add_file(str(path), name="model.pt")
            self.wandb_run.log_artifact(artifact)
            rprint(f"[ckpt ] logged best checkpoint to W&B")

        latest = self.out_path / "latest.pt"
        torch.save(ckpt, latest)
        if is_best:
            best = self.out_path / "best.pt"
            torch.save(ckpt, best)
            rprint(f"[ckpt ] new best val loss: {self.best_val_loss:.4f} -> {best}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        def _unwrap(m):
            return m.module if isinstance(m, DDP) else m

        _unwrap(self.encoder).load_state_dict(ckpt["encoder"])
        _unwrap(self.predictor).load_state_dict(ckpt["predictor"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.epoch = ckpt["epoch"] + 1
        self.global_step = ckpt["global_step"]
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        rprint(f"[ckpt ] resumed from {path} at epoch {self.epoch}, step {self.global_step}")


# ============================================================================
# Entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Path to OmegaConf YAML config.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Disable wandb and run a short sanity check.")
    parser.add_argument("overrides", nargs="*",
                        help="OmegaConf dotlist overrides, e.g. train.lr=5e-4")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg.merge_with_dotlist(args.overrides)
    if args.dry_run:
        cfg.dry_run = True

    rank, world_size, local_rank = ddp_setup()
    rprint(OmegaConf.to_yaml(cfg, resolve=True))

    # Seed
    seed = cfg.get("seed", 42) + rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    trainer = JEPATrainer(cfg, rank, world_size, local_rank)

    if args.resume is not None:
        trainer.load_checkpoint(args.resume)

    trainer.train()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()