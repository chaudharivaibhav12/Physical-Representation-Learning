"""
Linear probe + kNN evaluation for VideoMAE encoder.

What it does:
    1. Loads a saved VideoMAE checkpoint (encoder weights; decoder discarded).
    2. Freezes the encoder.
    3. Iterates train/val/test splits, encoding each 16-frame window with
       model.encode() → mean-pooled (B, 192) feature vector.
    4. Aggregates per trajectory (mean-pool windows from same simulation).
    5. Trains TWO heads on frozen features:
        a) Linear probe: nn.Linear(192, 2) regressing to (alpha, zeta).
        b) kNN regressor (sklearn), k chosen by val MSE.
    6. Reports MSE on val/test in normalized (z-scored) and original physical units.
    7. Saves results as JSON.

Usage:
    python eval_probe.py \
        --checkpoint /scratch/sb10583/checkpoints/videomae-v1/best.pt \
        --data-dir   /scratch/sb10583/data/data \
        --output-json videomae_results.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model   import VideoMAE
from dataset import VideoMAEEval


# ============================================================================
# Trajectory-aware dataset wrapper
# ============================================================================

class WithSimID(Dataset):
    """Adds sim_id string to each batch for trajectory aggregation."""
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        fpath, sim_idx, *_ = self.ds.samples[idx]
        sample["sim_id"] = f"{fpath}||{sim_idx}"
        return sample


# ============================================================================
# Encoder loading
# ============================================================================

def build_encoder_from_checkpoint(checkpoint_path: Path,
                                  device: torch.device) -> nn.Module:
    print(f"[load ] checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg  = ckpt.get("config", {})

    model = VideoMAE(
        in_channels   = cfg.get("in_channels",    11),
        num_frames    = cfg.get("num_frames",      16),
        img_size      = cfg.get("crop_size",      224),
        enc_embed_dim = cfg.get("enc_embed_dim",  192),
        enc_depth     = cfg.get("enc_depth",       12),
        enc_heads     = cfg.get("enc_heads",        3),
        mlp_ratio     = cfg.get("mlp_ratio",      4.0),
        dropout       = cfg.get("dropout",        0.0),
        patch_size    = cfg.get("patch_size",      16),
        tubelet       = cfg.get("tubelet",          2),
        mask_ratio    = cfg.get("mask_ratio",     0.90),
        dec_embed_dim = cfg.get("dec_embed_dim",   96),
        dec_depth     = cfg.get("dec_depth",        4),
        dec_heads     = cfg.get("dec_heads",        3),
    )
    enc_state = ckpt.get("encoder")
    if enc_state:
        model.encoder.load_state_dict(enc_state)
    else:
        state = ckpt["model"]
        if all(k.startswith("module.") for k in state):
            state = {k[7:]: v for k, v in state.items()}
        model.load_state_dict(state)

    model = model.to(device).eval()
    for p in model.encoder.parameters():
        p.requires_grad = False

    n = sum(p.numel() for p in model.encoder.parameters())
    print(f"[load ] encoder: {n:,} params, frozen, on {device}")
    return model


# ============================================================================
# Feature extraction
# ============================================================================

@torch.no_grad()
def extract_features(model: nn.Module, data_dir: str, split: str,
                     device: torch.device, batch_size: int = 32,
                     num_workers: int = 4, stride: int = 16,
                     use_amp: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    ds = WithSimID(VideoMAEEval(
        data_dir=data_dir, split=split, stride=stride,
    ))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    feats_list:  List[np.ndarray] = []
    labels_list: List[np.ndarray] = []
    sim_ids:     List[str]        = []

    amp_dtype = torch.bfloat16 if use_amp and device.type == "cuda" else torch.float32
    t0 = time.time()

    for batch_idx, batch in enumerate(loader):
        x = batch["frames"].to(device, non_blocking=True)
        with torch.amp.autocast(device.type, enabled=use_amp, dtype=amp_dtype):
            f = model.encode(x)                                # (B, 192)

        feats_list.append(f.float().cpu().numpy())
        labels_list.append(
            np.stack([batch["alpha"].numpy(), batch["zeta"].numpy()], axis=1)
        )
        sim_ids.extend(batch["sim_id"])

        if (batch_idx + 1) % 25 == 0:
            done    = (batch_idx + 1) * batch_size
            elapsed = time.time() - t0
            print(f"[feat ] [{split}] {done}/{len(ds)}  ({elapsed:.1f}s)", flush=True)

    features = np.concatenate(feats_list, axis=0)
    labels   = np.concatenate(labels_list, axis=0)
    print(f"[feat ] {split}: features={features.shape} labels={labels.shape} "
          f"in {time.time()-t0:.1f}s")
    return features, labels, sim_ids


# ============================================================================
# Per-trajectory aggregation
# ============================================================================

def aggregate_per_trajectory(features: np.ndarray, labels: np.ndarray,
                              sim_ids: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    unique  = list(dict.fromkeys(sim_ids))
    idx_map = {s: i for i, s in enumerate(unique)}
    N, D    = len(unique), features.shape[1]
    agg_f   = np.zeros((N, D), dtype=features.dtype)
    agg_l   = np.zeros((N, 2), dtype=labels.dtype)
    cnt     = np.zeros(N)
    for i, sid in enumerate(sim_ids):
        j = idx_map[sid]
        agg_f[j] += features[i]
        agg_l[j]  = labels[i]
        cnt[j]   += 1
    agg_f /= cnt[:, None]
    print(f"  aggregated: {features.shape[0]} windows → {N} trajectories")
    return agg_f, agg_l


# ============================================================================
# Label normalisation (fitted on train set)
# ============================================================================

def fit_label_scaler(y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = y_train.mean(axis=0)
    std  = y_train.std(axis=0) + 1e-6
    return mean, std

def normalize_labels(y: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (y - mean) / std

def denormalize_labels(z: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return z * std + mean


# ============================================================================
# Linear probe
# ============================================================================

def train_linear_probe(X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray,   y_val: np.ndarray,
                       lbl_mean: np.ndarray, lbl_std: np.ndarray,
                       device: torch.device,
                       epochs: int = 200, batch_size: int = 64,
                       lr: float = 1e-2, weight_decay: float = 1e-4,
                       seed: int = 42) -> Tuple[nn.Module, np.ndarray, np.ndarray, Dict]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    mu = X_train.mean(axis=0);  sd = X_train.std(axis=0) + 1e-6
    Xt = torch.from_numpy((X_train - mu) / sd).float().to(device)
    Xv = torch.from_numpy((X_val   - mu) / sd).float().to(device)

    yt_z = torch.from_numpy(normalize_labels(y_train, lbl_mean, lbl_std)).float().to(device)
    yv_z = torch.from_numpy(normalize_labels(y_val,   lbl_mean, lbl_std)).float().to(device)

    head    = nn.Linear(X_train.shape[1], 2).to(device)
    opt     = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_val, best_state = float("inf"), None
    n = Xt.shape[0]

    for _ in range(epochs):
        head.train()
        perm = rng.permutation(n)
        for s in range(0, n, batch_size):
            idx  = perm[s:s + batch_size]
            loss = loss_fn(head(Xt[idx]), yt_z[idx])
            opt.zero_grad(set_to_none=True);  loss.backward();  opt.step()
        head.eval()
        with torch.no_grad():
            v = loss_fn(head(Xv), yv_z).item()
        if v < best_val:
            best_val = v
            best_state = {k: w.detach().clone() for k, w in head.state_dict().items()}

    head.load_state_dict(best_state)
    info = {"best_val_mse_normalized": best_val}
    print(f"[probe] linear: best val MSE (normalized) = {best_val:.4f}")
    return head, mu, sd, info


@torch.no_grad()
def evaluate_linear_probe(head: nn.Module, X: np.ndarray, y: np.ndarray,
                           feat_mu: np.ndarray, feat_sd: np.ndarray,
                           lbl_mean: np.ndarray, lbl_std: np.ndarray,
                           device: torch.device) -> Dict:
    X_t    = torch.from_numpy((X - feat_mu) / feat_sd).float().to(device)
    y_t    = torch.from_numpy(y).float()
    pred_z = head(X_t)
    y_z    = torch.from_numpy(normalize_labels(y, lbl_mean, lbl_std)).float().to(device)
    pred   = torch.from_numpy(denormalize_labels(pred_z.cpu().numpy(), lbl_mean, lbl_std)).float()
    return {
        "mse_normalized_alpha": ((pred_z[:,0] - y_z[:,0])**2).mean().item(),
        "mse_normalized_zeta":  ((pred_z[:,1] - y_z[:,1])**2).mean().item(),
        "mse_normalized_avg":   ((pred_z - y_z)**2).mean().item(),
        "mse_alpha":            ((pred[:,0] - y_t[:,0])**2).mean().item(),
        "mse_zeta":             ((pred[:,1] - y_t[:,1])**2).mean().item(),
        "mse_avg":              ((pred - y_t)**2).mean().item(),
    }


# ============================================================================
# kNN regressor
# ============================================================================

def fit_knn(X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray,   y_val: np.ndarray,
            lbl_mean: np.ndarray, lbl_std: np.ndarray,
            ks: Tuple[int, ...] = (1, 3, 5, 10, 20)):
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors     import KNeighborsRegressor

    scaler = StandardScaler().fit(X_train)
    X_tr_s = scaler.transform(X_train)
    X_va_s = scaler.transform(X_val)
    y_tr_z = normalize_labels(y_train, lbl_mean, lbl_std)
    y_va_z = normalize_labels(y_val,   lbl_mean, lbl_std)

    best = (None, float("inf"), -1)
    for k in ks:
        if k > X_tr_s.shape[0]: continue
        knn = KNeighborsRegressor(n_neighbors=k, weights="distance", n_jobs=-1)
        knn.fit(X_tr_s, y_tr_z)
        mse = float(np.mean((knn.predict(X_va_s) - y_va_z) ** 2))
        print(f"[probe] kNN k={k:>3d}: val MSE (normalized) = {mse:.4f}")
        if mse < best[1]:
            best = (knn, mse, k)

    knn, val_mse, k = best
    print(f"[probe] kNN: best k={k}, val MSE (normalized) = {val_mse:.4f}")
    return knn, scaler, {"best_k": k, "best_val_mse_normalized": val_mse}


def evaluate_knn(knn, scaler, X: np.ndarray, y: np.ndarray,
                 lbl_mean: np.ndarray, lbl_std: np.ndarray) -> Dict:
    X_s    = scaler.transform(X)
    y_z    = normalize_labels(y, lbl_mean, lbl_std)
    pred_z = knn.predict(X_s)
    pred   = denormalize_labels(pred_z, lbl_mean, lbl_std)
    return {
        "mse_normalized_alpha": float(np.mean((pred_z[:,0] - y_z[:,0])**2)),
        "mse_normalized_zeta":  float(np.mean((pred_z[:,1] - y_z[:,1])**2)),
        "mse_normalized_avg":   float(np.mean((pred_z - y_z)**2)),
        "mse_alpha":            float(np.mean((pred[:,0]  - y[:,0])**2)),
        "mse_zeta":             float(np.mean((pred[:,1]  - y[:,1])**2)),
        "mse_avg":              float(np.mean((pred - y)**2)),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",    required=True, type=Path)
    parser.add_argument("--data-dir",      default="/scratch/sb10583/data/data")
    parser.add_argument("--output-json",   default="videomae_results.json", type=Path)
    parser.add_argument("--batch-size",    default=32,  type=int)
    parser.add_argument("--num-workers",   default=4,   type=int)
    parser.add_argument("--stride",        default=16,  type=int)
    parser.add_argument("--linear-epochs", default=200, type=int)
    parser.add_argument("--no-amp",        action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[main ] device: {device}")

    # 1. Encoder
    model = build_encoder_from_checkpoint(args.checkpoint, device)

    # 2. Extract features
    print("\n[main ] extracting features")
    Xtr, ytr, ids_tr = extract_features(model, args.data_dir, "train", device,
                                         args.batch_size, args.num_workers,
                                         args.stride, not args.no_amp)
    Xva, yva, ids_va = extract_features(model, args.data_dir, "valid", device,
                                         args.batch_size, args.num_workers,
                                         args.stride, not args.no_amp)
    Xte, yte, ids_te = extract_features(model, args.data_dir, "test",  device,
                                         args.batch_size, args.num_workers,
                                         args.stride, not args.no_amp)

    # 3. Aggregate per trajectory
    print("\n[main ] aggregating to per-trajectory features")
    Xtr, ytr = aggregate_per_trajectory(Xtr, ytr, ids_tr)
    Xva, yva = aggregate_per_trajectory(Xva, yva, ids_va)
    Xte, yte = aggregate_per_trajectory(Xte, yte, ids_te)
    print(f"  train: {Xtr.shape}, val: {Xva.shape}, test: {Xte.shape}")

    lbl_mean, lbl_std = fit_label_scaler(ytr)

    # 4. Linear probe
    print("\n[main ] training linear probe")
    head, fm, fs, lin_info = train_linear_probe(Xtr, ytr, Xva, yva, lbl_mean, lbl_std, device,
                                                 epochs=args.linear_epochs)
    print("[main ] evaluating linear probe")
    lin_val  = evaluate_linear_probe(head, Xva, yva, fm, fs, lbl_mean, lbl_std, device)
    lin_test = evaluate_linear_probe(head, Xte, yte, fm, fs, lbl_mean, lbl_std, device)

    # 5. kNN
    print("\n[main ] fitting kNN")
    knn, scaler, knn_info = fit_knn(Xtr, ytr, Xva, yva, lbl_mean, lbl_std)
    print("[main ] evaluating kNN")
    knn_val  = evaluate_knn(knn, scaler, Xva, yva, lbl_mean, lbl_std)
    knn_test = evaluate_knn(knn, scaler, Xte, yte, lbl_mean, lbl_std)

    # 6. Summary
    best_k = knn_info["best_k"]
    print("\n" + "="*72)
    print("RESULTS  (lower MSE = better; random baseline ≈ 1.0 normalized)")
    print("="*72)
    print(f"  Linear probe — val  MSE (normalized): {lin_val['mse_normalized_avg']:.4f}")
    print(f"  Linear probe — test MSE (normalized): {lin_test['mse_normalized_avg']:.4f}")
    print(f"  kNN (k={best_k}) — val  MSE (normalized): {knn_val['mse_normalized_avg']:.4f}")
    print(f"  kNN (k={best_k}) — test MSE (normalized): {knn_test['mse_normalized_avg']:.4f}")
    print()
    print(f"  Linear probe — test alpha MSE: {lin_test['mse_alpha']:.4f} (orig units)")
    print(f"  Linear probe — test zeta  MSE: {lin_test['mse_zeta']:.4f} (orig units)")
    print(f"  kNN          — test alpha MSE: {knn_test['mse_alpha']:.4f} (orig units)")
    print(f"  kNN          — test zeta  MSE: {knn_test['mse_zeta']:.4f} (orig units)")
    print("="*72)

    # 7. Save
    results = {
        "checkpoint":  str(args.checkpoint),
        "n_train":     int(Xtr.shape[0]),
        "n_val":       int(Xva.shape[0]),
        "n_test":      int(Xte.shape[0]),
        "feature_dim": int(Xtr.shape[1]),
        "linear_probe": {"train_info": lin_info, "val": lin_val, "test": lin_test},
        "knn":          {"train_info": knn_info,  "val": knn_val, "test": knn_test},
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[main ] results saved → {args.output_json}")


if __name__ == "__main__":
    main()
