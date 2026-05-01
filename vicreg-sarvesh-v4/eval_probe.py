"""
Linear probe + kNN evaluation for VICReg v4 encoder.

Steps:
  1. Load encoder from checkpoint (projector discarded).
  2. Freeze encoder, extract features for train/val/test
     using non-overlapping 16-frame windows (stride=16).
  3. Aggregate per trajectory (mean-pool windows from same simulation).
  4. Linear probe: nn.Linear(384, 2) trained with AdamW + MSE on z-scored labels.
     Best weights chosen by val MSE.
  5. kNN: sweep k in {1,3,5,10,20}, pick best on val, evaluate on test.
  6. Report MSE in normalized space and original physical units.
  7. Save results to JSON.

Usage:
  python eval_probe.py --checkpoint /scratch/sb10583/checkpoints/vicreg-v4/best.pt
  python eval_probe.py --checkpoint best.pt --output-json results.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model   import VICReg
from dataset import ActiveMatterEval


# ─────────────────────────────────────────────
# Trajectory-aware dataset wrapper
# ─────────────────────────────────────────────

class WithSimID(Dataset):
    """Adds 'sim_id' string to each batch for trajectory aggregation."""
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        fpath, sim_idx, *_ = self.ds.samples[idx]
        sample["sim_id"] = f"{fpath}||{sim_idx}"
        return sample


# ─────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────

def load_encoder(checkpoint_path: Path, device: torch.device) -> nn.Module:
    print(f"[load ] {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg  = ckpt.get("config", {})

    model = VICReg(
        in_channels = cfg.get("in_channels",  11),
        embed_dim   = cfg.get("embed_dim",    384),
        depth       = cfg.get("depth",          6),
        num_heads   = cfg.get("num_heads",       6),
        mlp_ratio   = cfg.get("mlp_ratio",     4.0),
        dropout     = cfg.get("dropout",       0.0),
        img_size    = cfg.get("crop_size",     224),
        patch_size  = cfg.get("patch_size",     32),
        tubelet     = cfg.get("tubelet",         2),
        num_frames  = cfg.get("num_frames",     16),
    )
    enc_state = ckpt.get("encoder")
    if enc_state:
        model.encoder.load_state_dict(enc_state)
    else:
        state = ckpt["model"]
        if all(k.startswith("module.") for k in state):
            state = {k[7:]: v for k, v in state.items()}
        model.load_state_dict(state)

    encoder = model.encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False

    n = sum(p.numel() for p in encoder.parameters())
    print(f"[load ] encoder {n:,} params, frozen, on {device}")
    return encoder


# ─────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────

@torch.no_grad()
def extract_features(encoder, data_dir: str, split: str,
                     device: torch.device, batch_size: int,
                     num_workers: int, stride: int = 16):
    ds = WithSimID(ActiveMatterEval(
        data_dir=data_dir, split=split, stride=stride,
    ))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    feats, alphas, zetas, sim_ids = [], [], [], []
    t0 = time.time()
    for i, batch in enumerate(loader):
        x = batch["x"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            f = encoder.forward_pooled(x)          # (B, 384)
        feats.append(f.float().cpu().numpy())
        alphas.append(batch["alpha"].numpy())
        zetas.append(batch["zeta"].numpy())
        sim_ids.extend(batch["sim_id"])
        if (i + 1) % 50 == 0:
            print(f"  [{split}] {(i+1)*batch_size}/{len(ds)}  {time.time()-t0:.1f}s")

    feats  = np.concatenate(feats)
    labels = np.stack([np.concatenate(alphas), np.concatenate(zetas)], axis=1)
    print(f"[feat ] {split}: {feats.shape}  {time.time()-t0:.1f}s")
    return feats, labels, sim_ids


# ─────────────────────────────────────────────
# Per-trajectory aggregation
# ─────────────────────────────────────────────

def aggregate(feats: np.ndarray, labels: np.ndarray, sim_ids: list):
    unique = list(dict.fromkeys(sim_ids))
    idx_map = {s: i for i, s in enumerate(unique)}
    N, D = len(unique), feats.shape[1]
    agg_f = np.zeros((N, D), dtype=feats.dtype)
    agg_l = np.zeros((N, 2), dtype=labels.dtype)
    cnt   = np.zeros(N)
    for i, sid in enumerate(sim_ids):
        j = idx_map[sid]
        agg_f[j] += feats[i]
        agg_l[j]  = labels[i]
        cnt[j]   += 1
    agg_f /= cnt[:, None]
    print(f"  aggregated: {feats.shape[0]} windows → {N} trajectories")
    return agg_f, agg_l


# ─────────────────────────────────────────────
# Label normalisation (computed from train set)
# ─────────────────────────────────────────────

def fit_label_scaler(y_train: np.ndarray):
    mean = y_train.mean(axis=0)
    std  = y_train.std(axis=0) + 1e-6
    return mean, std

def norm_labels(y, mean, std):
    return (y - mean) / std

def denorm_labels(z, mean, std):
    return z * std + mean


# ─────────────────────────────────────────────
# Linear probe
# ─────────────────────────────────────────────

def train_linear_probe(X_tr, y_tr, X_va, y_va, device,
                       epochs=200, batch_size=64, lr=1e-2, wd=1e-4, seed=42):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # Normalise features with train statistics
    mu = X_tr.mean(axis=0);  sd = X_tr.std(axis=0) + 1e-6
    X_tr_n = (X_tr - mu) / sd;  X_va_n = (X_va - mu) / sd

    # Normalise labels
    lm, ls = fit_label_scaler(y_tr)
    y_tr_n = norm_labels(y_tr, lm, ls)
    y_va_n = norm_labels(y_va, lm, ls)

    Xt = torch.from_numpy(X_tr_n).float().to(device)
    yt = torch.from_numpy(y_tr_n).float().to(device)
    Xv = torch.from_numpy(X_va_n).float().to(device)
    yv = torch.from_numpy(y_va_n).float().to(device)

    head = nn.Linear(X_tr.shape[1], 2).to(device)
    opt  = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)

    best_val, best_state = float("inf"), None
    n = Xt.shape[0]
    for epoch in range(epochs):
        head.train()
        for s in range(0, n, batch_size):
            idx = rng.choice(n, min(batch_size, n), replace=False)
            loss = nn.functional.mse_loss(head(Xt[idx]), yt[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            v = nn.functional.mse_loss(head(Xv), yv).item()
        if v < best_val:
            best_val = v
            best_state = {k: w.clone() for k, w in head.state_dict().items()}

    head.load_state_dict(best_state)
    print(f"[probe] linear best val MSE (norm): {best_val:.4f}")
    return head, mu, sd, lm, ls


@torch.no_grad()
def eval_linear(head, X, y, feat_mu, feat_sd, lbl_mean, lbl_std, device):
    X_n  = torch.from_numpy((X - feat_mu) / feat_sd).float().to(device)
    y_t  = torch.from_numpy(y).float().to(device)
    pred_n = head(X_n)
    y_n    = torch.from_numpy(norm_labels(y, lbl_mean, lbl_std)).float().to(device)
    pred   = torch.from_numpy(denorm_labels(pred_n.cpu().numpy(), lbl_mean, lbl_std)).float()
    return {
        "mse_norm_alpha": ((pred_n[:,0] - y_n[:,0])**2).mean().item(),
        "mse_norm_zeta":  ((pred_n[:,1] - y_n[:,1])**2).mean().item(),
        "mse_norm_avg":   ((pred_n - y_n)**2).mean().item(),
        "mse_alpha":      ((pred[:,0] - y_t[:,0].cpu())**2).mean().item(),
        "mse_zeta":       ((pred[:,1] - y_t[:,1].cpu())**2).mean().item(),
    }


# ─────────────────────────────────────────────
# kNN
# ─────────────────────────────────────────────

def fit_knn(X_tr, y_tr, X_va, y_va, lbl_mean, lbl_std,
            ks=(1, 3, 5, 10, 20)):
    from sklearn.preprocessing    import StandardScaler
    from sklearn.neighbors        import KNeighborsRegressor

    scaler   = StandardScaler().fit(X_tr)
    X_tr_s   = scaler.transform(X_tr)
    X_va_s   = scaler.transform(X_va)
    y_tr_n   = norm_labels(y_tr, lbl_mean, lbl_std)
    y_va_n   = norm_labels(y_va, lbl_mean, lbl_std)

    best = (None, float("inf"), -1)
    for k in ks:
        if k > len(X_tr_s): continue
        knn = KNeighborsRegressor(n_neighbors=k, weights="distance", n_jobs=-1)
        knn.fit(X_tr_s, y_tr_n)
        mse = float(np.mean((knn.predict(X_va_s) - y_va_n) ** 2))
        print(f"[probe] kNN k={k:>3d}: val MSE (norm) = {mse:.4f}")
        if mse < best[1]:
            best = (knn, mse, k)

    print(f"[probe] kNN best k={best[2]}, val MSE (norm) = {best[1]:.4f}")
    return best[0], scaler, best[2]


def eval_knn(knn, scaler, X, y, lbl_mean, lbl_std):
    X_s    = scaler.transform(X)
    y_n    = norm_labels(y, lbl_mean, lbl_std)
    pred_n = knn.predict(X_s)
    pred   = denorm_labels(pred_n, lbl_mean, lbl_std)
    return {
        "mse_norm_alpha": float(np.mean((pred_n[:,0] - y_n[:,0])**2)),
        "mse_norm_zeta":  float(np.mean((pred_n[:,1] - y_n[:,1])**2)),
        "mse_norm_avg":   float(np.mean((pred_n - y_n)**2)),
        "mse_alpha":      float(np.mean((pred[:,0]  - y[:,0])**2)),
        "mse_zeta":       float(np.mean((pred[:,1]  - y[:,1])**2)),
    }


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  required=True, type=Path)
    parser.add_argument("--data-dir",    default="/scratch/sb10583/data/data")
    parser.add_argument("--output-json", default="eval_probe_results.json", type=Path)
    parser.add_argument("--batch-size",  default=32,  type=int)
    parser.add_argument("--num-workers", default=4,   type=int)
    parser.add_argument("--stride",      default=16,  type=int)
    parser.add_argument("--linear-epochs", default=200, type=int)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[main ] device: {device}")

    encoder = load_encoder(args.checkpoint, device)

    print("\n[main ] extracting features")
    X_tr, y_tr, ids_tr = extract_features(encoder, args.data_dir, "train",  device, args.batch_size, args.num_workers, args.stride)
    X_va, y_va, ids_va = extract_features(encoder, args.data_dir, "valid",  device, args.batch_size, args.num_workers, args.stride)
    X_te, y_te, ids_te = extract_features(encoder, args.data_dir, "test",   device, args.batch_size, args.num_workers, args.stride)

    print("\n[main ] aggregating per trajectory")
    X_tr, y_tr = aggregate(X_tr, y_tr, ids_tr)
    X_va, y_va = aggregate(X_va, y_va, ids_va)
    X_te, y_te = aggregate(X_te, y_te, ids_te)

    lbl_mean, lbl_std = fit_label_scaler(y_tr)

    print("\n[main ] training linear probe")
    head, fm, fs, lm, ls = train_linear_probe(X_tr, y_tr, X_va, y_va, device, epochs=args.linear_epochs)
    lin_val  = eval_linear(head, X_va, y_va, fm, fs, lm, ls, device)
    lin_test = eval_linear(head, X_te, y_te, fm, fs, lm, ls, device)

    print("\n[main ] fitting kNN")
    knn, scaler, best_k = fit_knn(X_tr, y_tr, X_va, y_va, lbl_mean, lbl_std)
    knn_val  = eval_knn(knn, scaler, X_va, y_va, lbl_mean, lbl_std)
    knn_test = eval_knn(knn, scaler, X_te, y_te, lbl_mean, lbl_std)

    print("\n" + "="*60)
    print("RESULTS  (normalized MSE — random baseline ≈ 1.0)")
    print("="*60)
    print(f"  Linear — val  avg: {lin_val['mse_norm_avg']:.4f}  (alpha {lin_val['mse_norm_alpha']:.4f}  zeta {lin_val['mse_norm_zeta']:.4f})")
    print(f"  Linear — test avg: {lin_test['mse_norm_avg']:.4f}  (alpha {lin_test['mse_norm_alpha']:.4f}  zeta {lin_test['mse_norm_zeta']:.4f})")
    print(f"  kNN k={best_k} — val  avg: {knn_val['mse_norm_avg']:.4f}  (alpha {knn_val['mse_norm_alpha']:.4f}  zeta {knn_val['mse_norm_zeta']:.4f})")
    print(f"  kNN k={best_k} — test avg: {knn_test['mse_norm_avg']:.4f}  (alpha {knn_test['mse_norm_alpha']:.4f}  zeta {knn_test['mse_norm_zeta']:.4f})")
    print("="*60)

    results = {
        "checkpoint": str(args.checkpoint),
        "n_train": int(X_tr.shape[0]), "n_val": int(X_va.shape[0]), "n_test": int(X_te.shape[0]),
        "feature_dim": int(X_tr.shape[1]),
        "linear_probe": {"val": lin_val, "test": lin_test},
        "knn": {"best_k": best_k, "val": knn_val, "test": knn_test},
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2))
    print(f"\n[main ] saved → {args.output_json}")


if __name__ == "__main__":
    main()
