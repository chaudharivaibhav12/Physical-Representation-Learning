"""
JEPA pretraining script — Conv-JEPA with EMA target encoder.

Training recipe:
    - AdamW optimizer, lr=1e-3, weight_decay=0.05, betas=(0.9, 0.95)
    - Cosine LR schedule with 1-epoch warmup, min_lr=1e-6
    - EMA target encoder: τ cosine-scheduled from 0.996 → 0.9999
    - Loss: MSE(predictor(ctx_embed), stop_grad(target_embed))
    - No augmentation (augment=false in dataset config)
    - Batch size 8 per device, target global batch size 256 via grad accum

EMA schedule:
    τ(t) = τ_end − (τ_end − τ_start) * (cos(π·t/T) + 1) / 2
    At t=0: τ = τ_start (0.996) — target updates 0.4% toward online per step
    At t=T: τ = τ_end  (0.9999) — nearly frozen; stable, high-quality targets

Usage (single GPU):
    python train.py --config config.yaml

Usage (multi-GPU with torchrun):
    torchrun --nproc_per_node=4 train.py --config config.yaml

Config overrides via CLI (OmegaConf dotlist syntax):
    python train.py --config config.yaml train.lr=5e-4 train.num_epochs=50
"""

from __future__ import annotations

import argparse
import copy
import gc
import math
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
from loss import ema_loss
from scheduler import CosineWarmupLR


# ============================================================================
# Distributed setup
# ============================================================================

def ddp_setup() -> tuple[int, int, int]:
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
    if is_main_process():
        print(msg, flush=True)


def reduce_scalar(x: torch.Tensor) -> torch.Tensor:
    if dist.is_initialized():
        x = x.detach().clone()
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        x = x / dist.get_world_size()
    return x


# ============================================================================
# Gradient monitoring
# ============================================================================

def compute_gradient_norm(optimizer: torch.optim.Optimizer) -> float:
    total_norm = 0.0
    for param_group in optimizer.param_groups:
        for p in param_group['params']:
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


# ============================================================================
# W&B configuration
# ============================================================================

def generate_run_name(cfg, custom_name: str = None) -> str:
    if custom_name:
        return custom_name
    lr = cfg.train.lr
    bs = cfg.train.batch_size
    seed = cfg.get('seed', 42)
    lr_str = f"{lr:.0e}".replace("+", "")
    return f"lr{lr_str}-bs{bs}-seed{seed}"


def get_wandb_config(cfg) -> dict:
    wandb_project = cfg.get("wandb_project", "physics-jepa-experiments")
    run_name = generate_run_name(cfg, cfg.get("run_name", None))
    experiment_group = cfg.get("experiment_group", "default")
    dataset_name = cfg.dataset.get("name", cfg.dataset.get("class_name", "unknown"))
    tags = [
        "convjepa-ema",
        dataset_name,
        f"lr_{cfg.train.lr:.0e}",
        f"bs_{cfg.train.batch_size}",
        f"epochs_{cfg.train.num_epochs}",
        f"ema_start_{cfg.train.get('ema_momentum_start', 0.996)}",
        f"ema_end_{cfg.train.get('ema_momentum_end', 0.9999)}",
    ]
    if cfg.get("experiment_type"):
        tags.append(cfg.experiment_type)
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
    import importlib
    module = importlib.import_module(cfg.dataset.module)
    DatasetCls = getattr(module, cfg.dataset.class_name)
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

        # Online encoder + predictor
        encoder, predictor = build_jepa(cfg)
        encoder = encoder.to(self.device)
        predictor = predictor.to(self.device)
        n_enc = count_params(encoder)
        n_pred = count_params(predictor)
        rprint(f"[model] encoder params: {n_enc:,} | predictor params: {n_pred:,} "
               f"| total: {n_enc + n_pred:,}")

        # Target encoder: EMA copy of online encoder, never trained by gradient.
        # Created before DDP wrapping so it stays a plain nn.Module.
        target_encoder = copy.deepcopy(encoder)
        target_encoder.to(self.device)
        for p in target_encoder.parameters():
            p.requires_grad_(False)
        self.target_encoder = target_encoder

        if world_size > 1:
            encoder = DDP(encoder, device_ids=[local_rank], find_unused_parameters=False)
            predictor = DDP(predictor, device_ids=[local_rank], find_unused_parameters=False)
        self.encoder = encoder
        self.predictor = predictor

        # Optimizer: only online encoder + predictor (NOT target_encoder).
        params = list(encoder.parameters()) + list(predictor.parameters())
        self.optimizer = torch.optim.AdamW(
            params,
            lr=cfg.train.lr,
            betas=tuple(cfg.train.get("betas", (0.9, 0.95))),
            weight_decay=cfg.train.weight_decay,
        )

        # Gradient accumulation
        effective_batch = cfg.train.batch_size * world_size
        target_global = cfg.train.get("target_global_batch_size", effective_batch)
        self.grad_accum = max(1, target_global // effective_batch)
        rprint(f"[train] effective batch: {effective_batch} | "
               f"target global: {target_global} | grad_accum: {self.grad_accum}")

        # LR schedule
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

        # EMA momentum params (read from config; stored for schedule computation)
        self.ema_momentum_start = cfg.train.get("ema_momentum_start", 0.996)
        self.ema_momentum_end = cfg.train.get("ema_momentum_end", 0.9999)
        rprint(f"[ema  ] momentum cosine schedule: {self.ema_momentum_start} → {self.ema_momentum_end}")

        # AMP
        self.use_amp = cfg.train.get("use_amp", True) and torch.cuda.is_available()
        self.amp_dtype = (torch.bfloat16 if cfg.train.get("amp_dtype", "bf16") == "bf16"
                          else torch.float16)
        self.scaler = (torch.amp.GradScaler("cuda")
                       if self.use_amp and self.amp_dtype == torch.float16 else None)

        # State
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")

        # Logging
        self.log_every = cfg.train.get("report_every", 50)
        if is_main_process() and _HAS_WANDB and cfg.get("use_wandb", False) and not cfg.get("dry_run", False):
            wb_config = get_wandb_config(cfg)
            wb_entity = cfg.get("wandb_entity", None)
            rprint(f"[wandb] entity='{wb_entity}' project='{wb_config['project']}' "
                   f"group='{wb_config['group']}' run='{wb_config['name']}'")
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
            wandb.watch(self.encoder, log="all", log_freq=max(1, self.log_every // 5), idx=0)
            wandb.watch(self.predictor, log="all", log_freq=max(1, self.log_every // 5), idx=1)
        else:
            self.wandb_run = None

    # --- EMA -------------------------------------------------------------------

    def _get_ema_momentum(self) -> float:
        """Cosine schedule: τ_start at step 0, τ_end at step total_steps."""
        t = self.global_step
        T = max(self.total_steps, 1)
        return self.ema_momentum_end - (self.ema_momentum_end - self.ema_momentum_start) * (
            math.cos(math.pi * t / T) + 1) / 2

    @torch.no_grad()
    def _update_ema(self, momentum: float):
        """Update target_encoder: θ_t ← momentum·θ_t + (1−momentum)·θ_online."""
        online = (self.encoder.module if isinstance(self.encoder, DDP)
                  else self.encoder)
        for p_online, p_target in zip(online.parameters(),
                                       self.target_encoder.parameters()):
            p_target.data.mul_(momentum).add_(p_online.data, alpha=1.0 - momentum)

    # --- Forward / loss --------------------------------------------------------

    def _forward_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        ctx = batch["context"].to(self.device, non_blocking=True)
        tgt = batch["target"].to(self.device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
            ctx_embed = self.encoder(ctx)
            with torch.no_grad():
                tgt_embed = self.target_encoder(tgt)
            pred = self.predictor(ctx_embed)
            loss_dict = ema_loss(pred, tgt_embed)
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
        self.target_encoder.eval()  # always eval mode; no BN running stats to update
        self.optimizer.zero_grad(set_to_none=True)

        t0 = time.time()
        running: Dict[str, list] = defaultdict(list)
        micro_in_accum = 0
        batch_times: list = []

        for step, batch in enumerate(self.train_loader):
            batch_start = time.time()

            loss_dict = self._forward_loss(batch)
            loss = loss_dict["loss"] / self.grad_accum

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

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

                # Capture grad norm before zero_grad wipes the gradients.
                grad_norm = compute_gradient_norm(self.optimizer)

                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                self.global_step += 1
                micro_in_accum = 0

                # EMA update after each optimizer step
                momentum = self._get_ema_momentum()
                self._update_ema(momentum)

                if self.global_step % self.log_every == 0:
                    avg_batch_ms = sum(batch_times) / len(batch_times) * 1000 if batch_times else 0
                    self._log_train(running, epoch, t0, grad_norm=grad_norm,
                                    batch_time_ms=avg_batch_ms)
                    running.clear()
                    batch_times = []
                    t0 = time.time()

        if running:
            avg_batch_ms = sum(batch_times) / len(batch_times) * 1000 if batch_times else 0
            self._log_train(running, epoch, t0, grad_norm=grad_norm,
                            batch_time_ms=avg_batch_ms)

    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict[str, float]:
        self.encoder.eval()
        self.predictor.eval()
        self.target_encoder.eval()

        agg: Dict[str, list] = defaultdict(list)
        val_steps = self.cfg.train.get("val_steps", None)

        for i, batch in enumerate(self.val_loader):
            if val_steps is not None and i >= val_steps:
                break
            loss_dict = self._forward_loss(batch)
            for k, v in loss_dict.items():
                agg[k].append(v.detach())

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
                   grad_norm: float = 0.0, batch_time_ms: float = 0.0):
        metrics = {}
        for k, vals in running.items():
            vals_stacked = torch.stack(vals)
            m = reduce_scalar(vals_stacked.mean())
            metrics[f"train/{k}"] = m.item()
            metrics[f"train/{k}_std"] = reduce_scalar(vals_stacked.std()).item()
            metrics[f"train/{k}_min"] = reduce_scalar(vals_stacked.min()).item()
            metrics[f"train/{k}_max"] = reduce_scalar(vals_stacked.max()).item()

        metrics["train/lr"] = self.scheduler.get_last_lr()
        metrics["train/ema_momentum"] = self._get_ema_momentum()
        metrics["train/grad_norm"] = grad_norm
        metrics["epoch"] = epoch
        steps_per_sec = self.log_every / max(1e-6, time.time() - t0)
        metrics["train/steps_per_sec"] = steps_per_sec

        if batch_time_ms > 0:
            metrics["train/batch_time_ms"] = batch_time_ms

        if is_main_process():
            msg = (f"[train] epoch {epoch} step {self.global_step}/{self.total_steps} | "
                   f"loss={metrics['train/loss']:.4f} | "
                   f"ema_mom={metrics['train/ema_momentum']:.6f} | "
                   f"lr={metrics['train/lr']:.2e} | "
                   f"grad={grad_norm:.2e} | "
                   f"{steps_per_sec:.2f} step/s")
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
            "target_encoder": self.target_encoder.state_dict(),
            "predictor": _unwrap(self.predictor).state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": OmegaConf.to_container(self.cfg, resolve=True),
        }

        path = self.out_path / f"epoch_{epoch}.pt"
        torch.save(ckpt, path)
        rprint(f"[ckpt ] saved {path}")

        if is_best and self.wandb_run is not None:
            artifact = wandb.Artifact(
                name="jepa-ema-best-checkpoint",
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
        if "target_encoder" in ckpt:
            self.target_encoder.load_state_dict(ckpt["target_encoder"])
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
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
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
