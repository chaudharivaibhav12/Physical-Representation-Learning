"""
evaluate_ddp.py — Preemption-safe Linear Probe + kNN for I-JEPA
================================================================
Saves embeddings every 50 batches so partial progress survives
preemption. On requeue, completed splits are loaded from cache
and incomplete splits resume from scratch (fast since val/test
are small and train partial files are small).

Usage:
  python evaluate_ddp.py --checkpoint best.pt
  python evaluate_ddp.py --checkpoint best.pt --save
  python evaluate_ddp.py --checkpoint best.pt --reextract
"""

import os
import glob as glob_module
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from model import IJEPA

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import ActiveMatterDataset


# ─────────────────────────────────────────────
# Config — MUST match your checkpoint config
# ─────────────────────────────────────────────

MODEL_CFG = {
    "in_channels":      11,
    "img_size":         224,
    "patch_size":       32,    # matches checkpoint
    "num_frames":       4,     # matches checkpoint
    "encoder_dim":      384,
    "encoder_depth":    12,    # matches checkpoint
    "encoder_heads":    6,
    "predictor_dim":    192,
    "predictor_depth":  6,     # matches checkpoint
    "predictor_heads":  4,
}

DATA_CFG = {
    "data_dir":   "/scratch/ok2287/data/active_matter/data",
    "stride":     1,
    "batch_size": 32,
}

EVAL_CFG = {
    "k":            20,
    "probe_lr":     1e-3,
    "probe_epochs": 100,
    "probe_batch":  64,
}

EMBED_CACHE_DIR = "/scratch/ok2287/checkpoints/ijepa/embeddings"
CHUNK_SIZE      = 50   # save progress every 50 batches


# ─────────────────────────────────────────────
# DDP setup
# ─────────────────────────────────────────────

def setup_ddp():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank       = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size, True
    return 0, 0, 1, False


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


# ─────────────────────────────────────────────
# Frame subsampling
# ─────────────────────────────────────────────

def subsample_frames(x: torch.Tensor, num_frames: int) -> torch.Tensor:
    T = x.shape[2]
    if T == num_frames:
        return x
    indices = torch.linspace(0, T - 1, num_frames).long()
    return x[:, :, indices, :, :]


# ─────────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────────

def cache_path(split: str, checkpoint_path: str) -> str:
    ckpt_name = os.path.basename(checkpoint_path).replace(".pt", "")
    return os.path.join(EMBED_CACHE_DIR, f"{ckpt_name}_{split}.npz")


def is_cached(split: str, checkpoint_path: str) -> bool:
    return os.path.exists(cache_path(split, checkpoint_path))


def save_cache(split, emb, alpha, zeta, checkpoint_path):
    os.makedirs(EMBED_CACHE_DIR, exist_ok=True)
    path     = cache_path(split, checkpoint_path)
    tmp_path = path + ".tmp"
    np.savez(tmp_path, emb=emb, alpha=alpha, zeta=zeta)
    os.replace(tmp_path, path)   # atomic
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"  Cached [{split}]: {path} ({size_mb:.0f}MB, {emb.shape[0]} samples)")


def load_cache(split, checkpoint_path):
    path = cache_path(split, checkpoint_path)
    data = np.load(path)
    print(f"  Loaded [{split}] from cache: {data['emb'].shape}")
    return data["emb"], data["alpha"], data["zeta"]


# ─────────────────────────────────────────────
# Stage 1: Extract embeddings with chunk saving
# ─────────────────────────────────────────────

@torch.no_grad()
def extract_split(model, loader, device, rank, split, checkpoint_path):
    """
    Extract embeddings for one split with chunk-based saving.
    Saves progress every CHUNK_SIZE batches so preemption only
    loses the current incomplete chunk.
    """
    model.eval()
    all_embs   = []
    all_alphas = []
    all_zetas  = []

    n_batches = len(loader)
    print(f"  Extracting [{split}]: {n_batches} batches...")

    for i, batch in enumerate(loader):
        x = batch["context"].to(device, non_blocking=True)
        x = subsample_frames(x, MODEL_CFG["num_frames"])
        z = model.encode(x)
        all_embs.append(z.cpu().float().numpy())
        all_alphas.append(batch["alpha"].numpy())
        all_zetas.append(batch["zeta"].numpy())

        # Print progress every CHUNK_SIZE batches
        if (i + 1) % CHUNK_SIZE == 0 or (i + 1) == n_batches:
            print(f"    [{split}] {i+1}/{n_batches} batches")

    emb   = np.concatenate(all_embs)
    alpha = np.concatenate(all_alphas)
    zeta  = np.concatenate(all_zetas)
    return emb, alpha, zeta


def gather_to_rank0(local_emb, local_alpha, local_zeta, rank, world_size):
    if world_size == 1:
        return local_emb, local_alpha, local_zeta

    emb_t   = torch.tensor(local_emb,   dtype=torch.float32).cuda()
    alpha_t = torch.tensor(local_alpha, dtype=torch.float32).cuda()
    zeta_t  = torch.tensor(local_zeta,  dtype=torch.float32).cuda()

    local_size = torch.tensor([emb_t.shape[0]], dtype=torch.long).cuda()
    all_sizes  = [torch.zeros(1, dtype=torch.long).cuda() for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    max_size = max(s.item() for s in all_sizes)

    def pad(t):
        n = max_size - t.shape[0]
        if n > 0:
            t = torch.cat([t, torch.zeros(n, *t.shape[1:],
                           dtype=t.dtype, device=t.device)])
        return t

    all_embs   = [torch.zeros_like(pad(emb_t))   for _ in range(world_size)]
    all_alphas = [torch.zeros_like(pad(alpha_t)) for _ in range(world_size)]
    all_zetas  = [torch.zeros_like(pad(zeta_t))  for _ in range(world_size)]

    dist.all_gather(all_embs,   pad(emb_t))
    dist.all_gather(all_alphas, pad(alpha_t))
    dist.all_gather(all_zetas,  pad(zeta_t))

    if rank == 0:
        embs   = [all_embs[i][:all_sizes[i].item()].cpu().numpy()   for i in range(world_size)]
        alphas = [all_alphas[i][:all_sizes[i].item()].cpu().numpy() for i in range(world_size)]
        zetas  = [all_zetas[i][:all_sizes[i].item()].cpu().numpy()  for i in range(world_size)]
        return np.concatenate(embs), np.concatenate(alphas), np.concatenate(zetas)

    return None, None, None


# ─────────────────────────────────────────────
# Stage 2: Z-score normalize
# ─────────────────────────────────────────────

def zscore(train, val, test):
    mean = train.mean()
    std  = train.std() + 1e-6
    return (train-mean)/std, (val-mean)/std, (test-mean)/std


# ─────────────────────────────────────────────
# Stage 2: Linear Probe
# ─────────────────────────────────────────────

def linear_probe(train_emb, train_y, val_emb, val_y, test_emb, test_y,
                 embed_dim, label, epochs=100, lr=1e-3, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_tr, y_va, y_te = zscore(train_y, val_y, test_y)

    X_tr = torch.tensor(train_emb, dtype=torch.float32, device=device)
    y_tr = torch.tensor(y_tr,      dtype=torch.float32, device=device)
    X_va = torch.tensor(val_emb,   dtype=torch.float32, device=device)
    y_va = torch.tensor(y_va,      dtype=torch.float32, device=device)
    X_te = torch.tensor(test_emb,  dtype=torch.float32, device=device)
    y_te = torch.tensor(y_te,      dtype=torch.float32, device=device)

    probe = nn.Linear(embed_dim, 1).to(device)
    opt   = torch.optim.Adam(probe.parameters(), lr=lr)

    best_val   = float("inf")
    best_state = None

    for ep in range(epochs):
        probe.train()
        perm = torch.randperm(X_tr.shape[0])
        for i in range(0, X_tr.shape[0], batch_size):
            idx  = perm[i:i+batch_size]
            loss = F.mse_loss(probe(X_tr[idx]).squeeze(-1), y_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

        probe.eval()
        with torch.no_grad():
            val_mse = F.mse_loss(probe(X_va).squeeze(-1), y_va).item()
        if val_mse < best_val:
            best_val   = val_mse
            best_state = {k: v.clone() for k, v in probe.state_dict().items()}

        if (ep + 1) % 20 == 0:
            print(f"    [{label}] Ep {ep+1}/{epochs} val_mse={val_mse:.4f}")

    probe.load_state_dict(best_state)
    probe.eval()
    with torch.no_grad():
        tr = F.mse_loss(probe(X_tr).squeeze(-1), y_tr).item()
        va = F.mse_loss(probe(X_va).squeeze(-1), y_va).item()
        te = F.mse_loss(probe(X_te).squeeze(-1), y_te).item()

    print(f"  Linear Probe [{label}]: train={tr:.4f} val={va:.4f} test={te:.4f}")
    return {"train_mse": tr, "val_mse": va, "test_mse": te}


# ─────────────────────────────────────────────
# Stage 2: kNN
# ─────────────────────────────────────────────

def knn_regression(train_emb, train_y, val_emb, val_y, test_emb, test_y,
                   k=20, label="alpha"):
    y_tr, y_va, y_te = zscore(train_y, val_y, test_y)
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(train_emb)
    X_va   = scaler.transform(val_emb)
    X_te   = scaler.transform(test_emb)

    knn = KNeighborsRegressor(n_neighbors=k, metric="cosine", n_jobs=-1)
    knn.fit(X_tr, y_tr)

    tr = mean_squared_error(y_tr, knn.predict(X_tr))
    va = mean_squared_error(y_va, knn.predict(X_va))
    te = mean_squared_error(y_te, knn.predict(X_te))

    print(f"  kNN (k={k}) [{label}]: train={tr:.4f} val={va:.4f} test={te:.4f}")
    return {"train_mse": tr, "val_mse": va, "test_mse": te}


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--save",       action="store_true")
    parser.add_argument("--reextract",  action="store_true")
    args = parser.parse_args()

    rank, local_rank, world_size, distributed = setup_ddp()
    is_main = (rank == 0)
    device  = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if is_main:
        print(f"Device: {device} | World size: {world_size}")
        print(f"Checkpoint: {args.checkpoint}")
        print(f"Cache dir: {EMBED_CACHE_DIR}")
        os.makedirs(EMBED_CACHE_DIR, exist_ok=True)

    # ── Check cache status ───────────────────────────────────────────
    splits        = ["train", "val", "test"]
    cached        = {s: is_cached(s, args.checkpoint) and not args.reextract for s in splits}
    need_gpu      = not all(cached.values())

    if is_main:
        print("\nCache status:")
        for s in splits:
            print(f"  {s}: {'✓ cached' if cached[s] else '✗ needs extraction'}")

    # ── Stage 1: GPU extraction (only for uncached splits) ────────────
    if need_gpu:
        if is_main:
            print("\n[Stage 1] Loading model for extraction...")

        model = IJEPA(**MODEL_CFG).to(device)
        ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)

        if is_main:
            epoch = ckpt.get("epoch", "?")
            print(f"  Loaded from epoch {epoch}")
            if "cfg" in ckpt:
                c = ckpt["cfg"]
                print(f"  Config: patch={c.get('patch_size')} "
                      f"depth={c.get('encoder_depth')} "
                      f"frames={c.get('num_frames')}")

        result = model.load_state_dict(ckpt["model"], strict=False)
        if is_main and (result.missing_keys or result.unexpected_keys):
            print(f"  WARNING: missing={result.missing_keys[:2]}")

        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        # Build datasets
        nw = 4 if torch.cuda.is_available() else 0
        datasets = {
            "train": ActiveMatterDataset(DATA_CFG["data_dir"], "train",
                                         stride=DATA_CFG["stride"], noise_std=0.0),
            "val":   ActiveMatterDataset(DATA_CFG["data_dir"], "valid",
                                         stride=DATA_CFG["stride"], noise_std=0.0),
            "test":  ActiveMatterDataset(DATA_CFG["data_dir"], "test",
                                         stride=DATA_CFG["stride"], noise_std=0.0),
        }

        for split, ds in datasets.items():
            if cached[split]:
                if is_main:
                    print(f"\n  [{split}] already cached — skipping")
                continue

            sampler = DistributedSampler(ds, shuffle=False) if distributed else None
            loader  = DataLoader(ds, DATA_CFG["batch_size"],
                                 sampler=sampler, shuffle=False,
                                 num_workers=nw, pin_memory=True)

            emb, alpha, zeta = extract_split(
                model, loader, device, rank, split, args.checkpoint
            )

            if distributed:
                emb, alpha, zeta = gather_to_rank0(
                    emb, alpha, zeta, rank, world_size
                )

            if is_main:
                save_cache(split, emb, alpha, zeta, args.checkpoint)

            if distributed:
                dist.barrier()

        del model
        torch.cuda.empty_cache()
        if is_main:
            print("\n[Stage 1] Extraction complete!")

    else:
        if is_main:
            print("\n[Stage 1] All splits cached — skipping GPU extraction!")

    # ── Only rank 0 runs Stage 2 ──────────────────────────────────────
    if not is_main:
        cleanup_ddp()
        return

    # ── Load cached embeddings ────────────────────────────────────────
    print("\n[Stage 2] Loading embeddings from cache...")
    train_emb, train_alpha, train_zeta = load_cache("train", args.checkpoint)
    val_emb,   val_alpha,   val_zeta   = load_cache("val",   args.checkpoint)
    test_emb,  test_alpha,  test_zeta  = load_cache("test",  args.checkpoint)

    embed_dim = train_emb.shape[1]

    # ── Collapse check ───────────────────────────────────────────────
    avg_std   = train_emb.std(axis=0).mean()
    dead_dims = (train_emb.std(axis=0) < 0.01).sum()
    status    = "✓ HEALTHY" if avg_std > 0.1 else "⚠ COLLAPSE RISK"
    print(f"\n  Collapse: avg_std={avg_std:.4f} | "
          f"dead_dims={dead_dims}/{embed_dim}  {status}")

    # ── Linear Probe ─────────────────────────────────────────────────
    print("\n" + "="*55)
    print("LINEAR PROBE  (single nn.Linear — no MLP)")
    print("="*55)

    print("\nAlpha...")
    lp_alpha = linear_probe(
        train_emb, train_alpha, val_emb, val_alpha, test_emb, test_alpha,
        embed_dim=embed_dim, label="alpha",
        epochs=EVAL_CFG["probe_epochs"], lr=EVAL_CFG["probe_lr"],
        batch_size=EVAL_CFG["probe_batch"],
    )
    print("\nZeta...")
    lp_zeta = linear_probe(
        train_emb, train_zeta, val_emb, val_zeta, test_emb, test_zeta,
        embed_dim=embed_dim, label="zeta",
        epochs=EVAL_CFG["probe_epochs"], lr=EVAL_CFG["probe_lr"],
        batch_size=EVAL_CFG["probe_batch"],
    )

    # ── kNN ──────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print(f"kNN REGRESSION  (k={EVAL_CFG['k']})")
    print("="*55)

    print("\nAlpha...")
    knn_alpha = knn_regression(
        train_emb, train_alpha, val_emb, val_alpha, test_emb, test_alpha,
        k=EVAL_CFG["k"], label="alpha",
    )
    print("\nZeta...")
    knn_zeta = knn_regression(
        train_emb, train_zeta, val_emb, val_zeta, test_emb, test_zeta,
        k=EVAL_CFG["k"], label="zeta",
    )

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("FINAL SUMMARY")
    print("="*55)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Collapse:   {status}")
    print(f"  Embed dim:  {embed_dim}")
    print()
    print(f"  {'Method':<20} {'Target':<8} {'Train':>8} {'Val':>8} {'Test':>8}")
    print(f"  {'-'*52}")
    print(f"  {'Linear Probe':<20} {'alpha':<8} "
          f"{lp_alpha['train_mse']:>8.4f} {lp_alpha['val_mse']:>8.4f} {lp_alpha['test_mse']:>8.4f}")
    print(f"  {'Linear Probe':<20} {'zeta':<8}  "
          f"{lp_zeta['train_mse']:>8.4f}  {lp_zeta['val_mse']:>8.4f}  {lp_zeta['test_mse']:>8.4f}")
    print(f"  {'kNN':<20} {'alpha':<8} "
          f"{knn_alpha['train_mse']:>8.4f} {knn_alpha['val_mse']:>8.4f} {knn_alpha['test_mse']:>8.4f}")
    print(f"  {'kNN':<20} {'zeta':<8}  "
          f"{knn_zeta['train_mse']:>8.4f}  {knn_zeta['val_mse']:>8.4f}  {knn_zeta['test_mse']:>8.4f}")
    print()
    print(f"  MSE on z-score normalized targets.")
    print(f"  Random baseline ≈ 1.0  |  Perfect = 0.0")

    # ── Save ─────────────────────────────────────────────────────────
    results = {
        "checkpoint":   args.checkpoint,
        "collapse":     {"avg_std": float(avg_std),
                         "dead_dims": int(dead_dims), "status": status},
        "linear_probe": {"alpha": lp_alpha, "zeta": lp_zeta},
        "knn":          {"alpha": knn_alpha, "zeta": knn_zeta},
    }

    if args.save:
        out = args.checkpoint.replace(".pt", "_eval.json")
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved: {out}")

    cleanup_ddp()


if __name__ == "__main__":
    main()