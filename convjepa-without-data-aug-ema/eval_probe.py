"""
Linear probe + kNN evaluation for a pretrained JEPA encoder.

What it does:
    1. Loads a saved JEPA checkpoint (encoder weights only).
    2. Freezes the encoder.
    3. Iterates train/val/test splits, encoding each (context, target) sample
       to a dense (B, 128, 14, 14) feature map. Pools to a single 128-dim
       feature vector per sample (mean over space).
    4. Trains TWO heads on the frozen features:
        a) Linear probe: a single nn.Linear(128, 2) regressing to (alpha, zeta).
        b) kNN regressor (sklearn), k chosen by val MSE.
    5. Reports MSE on val/test, both in normalized (z-scored) space and in
       original physical units.
    6. Saves results as JSON for the report.

This is the rubric-compliant evaluation: only Linear and kNN heads on frozen
features. No MLP / attention pooling. No fine-tuning.

Usage:
    python eval_probe.py \
        --checkpoint /scratch/vc2836/DL/checkpoints/activematter-convjepa-ema/best.pt \
        --cache-dir /scratch/vc2836/DL/data/active_matter/cache \
        --output-json conv-ema_results_2.json

Run on the same node/env where you trained. Takes ~5-15 min depending on
GPU availability and dataset size.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Make sure the user's dataset is importable (same env-var trick as train.py).
import os
_DATASET_DIR = os.environ.get("JEPA_DATASET_DIR")
if _DATASET_DIR:
    sys.path.insert(0, _DATASET_DIR)

# These imports come from the jepa_baseline package the script lives in.
from model import ConvEncoder


# ============================================================================
# Helpers
# ============================================================================

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_label_stats() -> Dict[str, torch.Tensor]:
    """Hardcoded normalization stats for active_matter (alpha, zeta)."""
    return {
        "mean": torch.tensor([-3.0, 9.0], dtype=torch.float32),
        "std": torch.tensor([1.4142, 5.1640], dtype=torch.float32),
    }


def normalize_labels(y: torch.Tensor) -> torch.Tensor:
    s = load_label_stats()
    return (y - s["mean"].to(y.device)) / s["std"].to(y.device)


def denormalize_labels(z: torch.Tensor) -> torch.Tensor:
    s = load_label_stats()
    return z * s["std"].to(z.device) + s["mean"].to(z.device)


# ============================================================================
# Encoder loading
# ============================================================================

def build_encoder_from_checkpoint(checkpoint_path: Path,
                                  device: torch.device) -> ConvEncoder:
    """Reconstruct the encoder using shapes from the checkpoint's config."""
    print(f"[load ] checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    m = cfg["model"]
    d = cfg["dataset"]

    encoder = ConvEncoder(
        in_chans=d["num_chans"],
        dims=tuple(m["dims"]),
        num_res_blocks=tuple(m["num_res_blocks"]),
        num_frames=d["num_frames"],
        drop_path_rate=0.0,  # always 0 at eval
    )
    missing, unexpected = encoder.load_state_dict(ckpt["encoder"], strict=False)
    if missing:
        print(f"[warn ] missing keys: {missing[:3]}{'...' if len(missing)>3 else ''}")
    if unexpected:
        print(f"[warn ] unexpected keys: {unexpected[:3]}{'...' if len(unexpected)>3 else ''}")

    encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False

    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"[load ] encoder: {n_params:,} params, frozen, on {device}")
    return encoder


# ============================================================================
# Dataset construction
# ============================================================================

def build_dataset(cache_dir: Path, split: str, num_frames: int):
    """Build ActiveMatterDataset for evaluation: NO augmentation, return labels."""
    import importlib
    mod = importlib.import_module("active_matter_dataset")
    DatasetCls = getattr(mod, "ActiveMatterDataset")
    return DatasetCls(
        cache_dir=str(cache_dir),
        split=split,
        num_frames=num_frames,
        stride=num_frames,           # NON-overlapping windows for eval
        augment=False,               # IMPORTANT: no roll/noise at eval time
        noise_std=0.0,
        return_physical_params=True,
    )


# ============================================================================
# Feature extraction
# ============================================================================

@torch.no_grad()
def extract_features(encoder: ConvEncoder, dataset, device: torch.device,
                     batch_size: int = 8, num_workers: int = 4,
                     use_amp: bool = True,
                     pool: str = "mean") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Encode every sample in the dataset.

    For each sample we encode the context window (16 frames) and pool the
    encoder output (B, 128, 14, 14) into a per-sample feature vector.

    NOTE on label aggregation: every (trajectory, t0) window has the same
    (alpha, zeta), so when we aggregate windows back to trajectories later
    we just take any window's labels.

    Returns:
        features:      (N, D)         where D=128 for mean-pool
        labels:        (N, 2)         [alpha, zeta] per window
        traj_idxs:     (N,)           which trajectory each window came from
    """
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    feats_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []
    traj_list: List[np.ndarray] = []

    encoder.eval()
    amp_dtype = torch.bfloat16 if use_amp and device.type == "cuda" else torch.float32
    n_per_traj = dataset.n_windows_per_traj

    t0 = time.time()
    for batch_idx, batch in enumerate(loader):
        ctx = batch["context"].to(device, non_blocking=True)  # (B, C, T, H, W)

        with torch.amp.autocast(device.type, enabled=use_amp, dtype=amp_dtype):
            embed = encoder(ctx)                               # (B, 128, 14, 14)

        if pool == "mean":
            f = embed.mean(dim=(-1, -2))                       # (B, 128)
        elif pool == "max":
            f = embed.amax(dim=(-1, -2))
        elif pool == "flatten":
            f = embed.flatten(1)                               # (B, 128*14*14)
        else:
            raise ValueError(f"unknown pool: {pool}")

        feats_list.append(f.float().cpu().numpy())
        labels_list.append(batch["physical_params"].numpy())

        # Reconstruct global flat index range for this batch.
        start = batch_idx * batch_size
        end = start + ctx.shape[0]
        idxs = np.arange(start, end)
        traj_list.append(idxs // n_per_traj)

        if (batch_idx + 1) % 25 == 0:
            done = end
            total = len(dataset)
            elapsed = time.time() - t0
            print(f"[feat ] {done}/{total}  ({elapsed:.1f}s, "
                  f"{done/elapsed:.1f} samples/sec)", flush=True)

    features = np.concatenate(feats_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    traj_idxs = np.concatenate(traj_list, axis=0)

    print(f"[feat ] done: features={features.shape} labels={labels.shape} "
          f"in {time.time()-t0:.1f}s")
    return features, labels, traj_idxs


def aggregate_per_trajectory(features: np.ndarray, labels: np.ndarray,
                             traj_idxs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Average features within each trajectory; labels are constant within a
    trajectory so we just take the first.

    This is recommended for the linear probe / kNN: it gives one feature
    vector per physical simulation, matching the (alpha, zeta) label which
    is also per-trajectory. Without aggregation, the same label appears 50x
    in the train set, which inflates kNN scores by trivial nearest neighbors.
    """
    unique_traj = np.unique(traj_idxs)
    pooled_features = np.zeros((len(unique_traj), features.shape[1]), dtype=features.dtype)
    pooled_labels = np.zeros((len(unique_traj), labels.shape[1]), dtype=labels.dtype)
    for i, t in enumerate(unique_traj):
        mask = traj_idxs == t
        pooled_features[i] = features[mask].mean(axis=0)
        pooled_labels[i] = labels[mask][0]   # all rows identical
    return pooled_features, pooled_labels


# ============================================================================
# Linear probe
# ============================================================================

def train_linear_probe(X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray,
                       device: torch.device,
                       epochs: int = 200, batch_size: int = 64,
                       lr: float = 1e-2, weight_decay: float = 1e-4,
                       seed: int = 42) -> Tuple[nn.Module, Dict[str, float]]:
    """
    Single nn.Linear(D, 2) trained with AdamW + MSE on z-scored labels.
    Selects best weights by validation MSE.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    Xt = torch.from_numpy(X_train).float().to(device)
    yt = torch.from_numpy(y_train).float().to(device)
    Xv = torch.from_numpy(X_val).float().to(device)
    yv = torch.from_numpy(y_val).float().to(device)

    # Z-score features using train statistics (purely for stability).
    mu = Xt.mean(dim=0, keepdim=True)
    sd = Xt.std(dim=0, keepdim=True).clamp_min(1e-6)
    Xt = (Xt - mu) / sd
    Xv = (Xv - mu) / sd

    # Z-score labels.
    yt_z = normalize_labels(yt)
    yv_z = normalize_labels(yv)

    D = X_train.shape[1]
    head = nn.Linear(D, 2).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    n = Xt.shape[0]

    for epoch in range(epochs):
        head.train()
        perm = rng.permutation(n)
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            xb = Xt[idx]
            yb = yt_z[idx]
            pred = head(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        head.eval()
        with torch.no_grad():
            val_pred = head(Xv)
            val_loss = loss_fn(val_pred, yv_z).item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}

    head.load_state_dict(best_state)
    metrics = {"best_val_mse_normalized": best_val}
    print(f"[probe] linear: best val MSE (normalized) = {best_val:.4f}")
    return head, metrics


@torch.no_grad()
def evaluate_linear_probe(head: nn.Module,
                          X: np.ndarray, y: np.ndarray,
                          X_train: np.ndarray,
                          device: torch.device) -> Dict[str, float]:
    """Compute MSE in normalized + denormalized space, separately for alpha/zeta."""
    Xt_train = torch.from_numpy(X_train).float().to(device)
    mu = Xt_train.mean(dim=0, keepdim=True)
    sd = Xt_train.std(dim=0, keepdim=True).clamp_min(1e-6)

    X_t = torch.from_numpy(X).float().to(device)
    y_t = torch.from_numpy(y).float().to(device)
    X_t = (X_t - mu) / sd

    head.eval()
    pred_z = head(X_t)
    y_z = normalize_labels(y_t)

    # Normalized MSE (the comparable number for the report).
    mse_alpha_z = ((pred_z[:, 0] - y_z[:, 0]) ** 2).mean().item()
    mse_zeta_z = ((pred_z[:, 1] - y_z[:, 1]) ** 2).mean().item()
    mse_total_z = (((pred_z - y_z) ** 2).mean()).item()

    # Denormalized (in original physical units, for interpretability).
    pred = denormalize_labels(pred_z)
    mse_alpha = ((pred[:, 0] - y_t[:, 0]) ** 2).mean().item()
    mse_zeta = ((pred[:, 1] - y_t[:, 1]) ** 2).mean().item()
    mse_total = (((pred - y_t) ** 2).mean()).item()

    return {
        "mse_normalized_alpha": mse_alpha_z,
        "mse_normalized_zeta": mse_zeta_z,
        "mse_normalized_avg": mse_total_z,
        "mse_alpha": mse_alpha,
        "mse_zeta": mse_zeta,
        "mse_avg": mse_total,
    }


# ============================================================================
# kNN regressor
# ============================================================================

def fit_knn(X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray,
            ks: Tuple[int, ...] = (1, 3, 5, 10, 20)) -> Tuple[object, Dict[str, float]]:
    """
    Sweep k on val MSE, return the best fitted model.

    Uses scikit-learn's KNeighborsRegressor with distance weighting (closer
    neighbors get higher weight; common practice for kNN regression).
    """
    from sklearn.neighbors import KNeighborsRegressor

    # Normalize labels for fair comparison with linear probe.
    y_train_z = (y_train - load_label_stats()["mean"].numpy()) / load_label_stats()["std"].numpy()
    y_val_z = (y_val - load_label_stats()["mean"].numpy()) / load_label_stats()["std"].numpy()

    best = (None, float("inf"), -1)
    for k in ks:
        if k > X_train.shape[0]:
            continue
        knn = KNeighborsRegressor(n_neighbors=k, weights="distance", n_jobs=-1)
        knn.fit(X_train, y_train_z)
        pred = knn.predict(X_val)
        mse = float(np.mean((pred - y_val_z) ** 2))
        print(f"[probe] kNN k={k:>3d}: val MSE (normalized) = {mse:.4f}")
        if mse < best[1]:
            best = (knn, mse, k)

    knn, val_mse, k = best
    print(f"[probe] kNN: best k={k}, val MSE (normalized) = {val_mse:.4f}")
    return knn, {"best_k": k, "best_val_mse_normalized": val_mse}


def evaluate_knn(knn, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Same metric structure as the linear probe."""
    stats = load_label_stats()
    mean = stats["mean"].numpy()
    std = stats["std"].numpy()
    y_z = (y - mean) / std
    pred_z = knn.predict(X)
    pred = pred_z * std + mean

    return {
        "mse_normalized_alpha": float(np.mean((pred_z[:, 0] - y_z[:, 0]) ** 2)),
        "mse_normalized_zeta": float(np.mean((pred_z[:, 1] - y_z[:, 1]) ** 2)),
        "mse_normalized_avg": float(np.mean((pred_z - y_z) ** 2)),
        "mse_alpha": float(np.mean((pred[:, 0] - y[:, 0]) ** 2)),
        "mse_zeta": float(np.mean((pred[:, 1] - y[:, 1]) ** 2)),
        "mse_avg": float(np.mean((pred - y) ** 2)),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="Path to best.pt or latest.pt")
    parser.add_argument("--cache-dir", required=True, type=Path,
                        help="Path to active_matter cache (contains metadata.json)")
    parser.add_argument("--output-json", default="results.json", type=Path)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--num-frames", default=16, type=int)
    parser.add_argument("--pool", default="mean", choices=("mean", "max", "flatten"))
    parser.add_argument("--linear-epochs", default=200, type=int)
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable mixed precision during feature extraction.")
    args = parser.parse_args()

    device = get_device()
    print(f"[main ] device: {device}")

    # 1. Encoder
    encoder = build_encoder_from_checkpoint(args.checkpoint, device)

    # 2. Datasets — we need train, val, AND test (test is rubric-required).
    print("\n[main ] building datasets")
    train_ds = build_dataset(args.cache_dir, "train", args.num_frames)
    val_ds = build_dataset(args.cache_dir, "valid", args.num_frames)
    test_ds = build_dataset(args.cache_dir, "test", args.num_frames)
    print(f"  train: {len(train_ds)} windows, val: {len(val_ds)} windows, "
          f"test: {len(test_ds)} windows")

    # 3. Extract features for all splits
    print("\n[main ] extracting train features")
    Xtr, ytr, ttr = extract_features(encoder, train_ds, device,
                                      batch_size=args.batch_size,
                                      num_workers=args.num_workers,
                                      use_amp=not args.no_amp, pool=args.pool)
    print("\n[main ] extracting val features")
    Xva, yva, tva = extract_features(encoder, val_ds, device,
                                      batch_size=args.batch_size,
                                      num_workers=args.num_workers,
                                      use_amp=not args.no_amp, pool=args.pool)
    print("\n[main ] extracting test features")
    Xte, yte, tte = extract_features(encoder, test_ds, device,
                                      batch_size=args.batch_size,
                                      num_workers=args.num_workers,
                                      use_amp=not args.no_amp, pool=args.pool)

    # 4. Aggregate per-trajectory (so each phys sim contributes one row).
    print("\n[main ] aggregating to per-trajectory features")
    Xtr_agg, ytr_agg = aggregate_per_trajectory(Xtr, ytr, ttr)
    Xva_agg, yva_agg = aggregate_per_trajectory(Xva, yva, tva)
    Xte_agg, yte_agg = aggregate_per_trajectory(Xte, yte, tte)
    print(f"  train: {Xtr_agg.shape}, val: {Xva_agg.shape}, test: {Xte_agg.shape}")

    # 5. Linear probe
    print("\n[main ] training linear probe")
    head, lin_train_info = train_linear_probe(
        Xtr_agg, ytr_agg, Xva_agg, yva_agg, device,
        epochs=args.linear_epochs,
    )
    print("[main ] evaluating linear probe on val/test")
    lin_val = evaluate_linear_probe(head, Xva_agg, yva_agg, Xtr_agg, device)
    lin_test = evaluate_linear_probe(head, Xte_agg, yte_agg, Xtr_agg, device)

    # 6. kNN
    print("\n[main ] training kNN")
    knn, knn_train_info = fit_knn(Xtr_agg, ytr_agg, Xva_agg, yva_agg)
    print("[main ] evaluating kNN on val/test")
    knn_val = evaluate_knn(knn, Xva_agg, yva_agg)
    knn_test = evaluate_knn(knn, Xte_agg, yte_agg)

    # 7. Print headline summary
    print("\n" + "="*72)
    print("RESULTS  (lower MSE = better)")
    print("="*72)
    print(f"  Linear probe — val  MSE (normalized): {lin_val['mse_normalized_avg']:.4f}")
    print(f"  Linear probe — test MSE (normalized): {lin_test['mse_normalized_avg']:.4f}")
    print(f"  kNN (k={knn_train_info['best_k']}) — val  MSE (normalized): {knn_val['mse_normalized_avg']:.4f}")
    print(f"  kNN (k={knn_train_info['best_k']}) — test MSE (normalized): {knn_test['mse_normalized_avg']:.4f}")
    print()
    print(f"  Linear probe — test alpha MSE: {lin_test['mse_alpha']:.4f} (orig units)")
    print(f"  Linear probe — test zeta  MSE: {lin_test['mse_zeta']:.4f} (orig units)")
    print(f"  kNN          — test alpha MSE: {knn_test['mse_alpha']:.4f} (orig units)")
    print(f"  kNN          — test zeta  MSE: {knn_test['mse_zeta']:.4f} (orig units)")
    print("="*72)

    # 8. Save
    results = {
        "checkpoint": str(args.checkpoint),
        "pool": args.pool,
        "n_train": int(Xtr_agg.shape[0]),
        "n_val": int(Xva_agg.shape[0]),
        "n_test": int(Xte_agg.shape[0]),
        "feature_dim": int(Xtr_agg.shape[1]),
        "linear_probe": {
            "train_info": lin_train_info,
            "val": lin_val,
            "test": lin_test,
        },
        "knn": {
            "train_info": knn_train_info,
            "val": knn_val,
            "test": knn_test,
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[main ] results saved -> {args.output_json}")


if __name__ == "__main__":
    main()
