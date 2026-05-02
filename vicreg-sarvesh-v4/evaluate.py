"""
VICReg-V4 Evaluation — Linear Probe + kNN Regression
=====================================================
Loads the frozen VICReg encoder from a checkpoint and evaluates
representation quality by predicting alpha and zeta using:
  1. Linear probe  — Ridge regression (sklearn)
  2. kNN           — k-Nearest Neighbours regression (k=20, cosine)

Targets are z-score normalized (train statistics).  MSE reported for both
val and test splits in a single run.

Usage:
  python evaluate.py --checkpoint /scratch/sb10583/checkpoints/vicreg-v4/best.pt
"""

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from model   import VICReg
from dataset import ActiveMatterEval


CONFIG = {
    "data_dir":    "/scratch/sb10583/data/data",
    "crop_size":   224,
    "batch_size":  32,

    # Must match training config
    "in_channels": 11,
    "embed_dim":   384,
    "depth":       6,
    "num_heads":   6,
    "patch_size":  32,
    "tubelet":     2,
    "num_frames":  16,
    "proj_hidden": 2048,
    "proj_out":    2048,
}


# ─────────────────────────────────────────────────────────────────────────────
# Embedding extraction
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(encoder, loader, device):
    encoder.eval()
    embeddings, alphas, zetas = [], [], []
    for batch in loader:
        x = batch["x"].to(device)
        z = encoder.forward_pooled(x)
        embeddings.append(z.cpu().numpy())
        alphas.append(batch["alpha"].numpy())
        zetas.append(batch["zeta"].numpy())
    return (
        np.concatenate(embeddings, axis=0),
        np.concatenate(alphas,     axis=0),
        np.concatenate(zetas,      axis=0),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Probe + kNN on one target variable
# ─────────────────────────────────────────────────────────────────────────────

def probe_and_knn(X_tr, X_val, X_test, y_tr, y_val, y_test, label, k=20):
    mu, sigma  = y_tr.mean(), y_tr.std() + 1e-8
    y_tr_n     = (y_tr   - mu) / sigma
    y_val_n    = (y_val  - mu) / sigma
    y_test_n   = (y_test - mu) / sigma

    scaler   = StandardScaler()
    X_tr_s   = scaler.fit_transform(X_tr)
    X_val_s  = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_tr_s, y_tr_n)
    lp_val  = mean_squared_error(y_val_n,  ridge.predict(X_val_s))
    lp_test = mean_squared_error(y_test_n, ridge.predict(X_test_s))

    knn = KNeighborsRegressor(n_neighbors=k, metric="cosine", n_jobs=-1)
    knn.fit(X_tr_s, y_tr_n)
    knn_val  = mean_squared_error(y_val_n,  knn.predict(X_val_s))
    knn_test = mean_squared_error(y_test_n, knn.predict(X_test_s))

    print(f"  Linear [{label:5s}] → Val: {lp_val:.4f}  Test: {lp_test:.4f}")
    print(f"  kNN    [{label:5s}] → Val: {knn_val:.4f}  Test: {knn_test:.4f}")
    return lp_val, lp_test, knn_val, knn_test


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args, cfg):
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda    = device.type == "cuda"
    num_workers = 4 if use_cuda else 0
    print(f"Device: {device}\n")

    # ── Load model ────────────────────────────────────────────────────────────
    model = VICReg(
        in_channels = cfg["in_channels"],
        embed_dim   = cfg["embed_dim"],
        depth       = cfg["depth"],
        num_heads   = cfg["num_heads"],
        img_size    = cfg["crop_size"],
        patch_size  = cfg["patch_size"],
        tubelet     = cfg["tubelet"],
        num_frames  = cfg["num_frames"],
        proj_hidden = cfg["proj_hidden"],
        proj_out    = cfg["proj_out"],
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.encoder.load_state_dict(ckpt["encoder"])
    model.eval()
    for param in model.encoder.parameters():
        param.requires_grad = False
    print(f"Loaded checkpoint: {args.checkpoint}  [encoder frozen]")

    # ── Datasets ──────────────────────────────────────────────────────────────
    # stride=1 → 8750/1200/1300 samples matching the project spec
    def make_loader(split):
        ds = ActiveMatterEval(cfg["data_dir"], split=split, crop_size=cfg["crop_size"], stride=1)
        return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False,
                          num_workers=num_workers, pin_memory=use_cuda)

    # ── Extract embeddings ────────────────────────────────────────────────────
    print("\n" + "═" * 55)
    print("EXTRACTING EMBEDDINGS")
    print("═" * 55)
    print("Extracting train embeddings...")
    X_tr, alpha_tr, zeta_tr = extract_embeddings(model.encoder, make_loader("train"), device)
    print("Extracting val embeddings...")
    X_val, alpha_val, zeta_val = extract_embeddings(model.encoder, make_loader("valid"), device)
    print("Extracting test embeddings...")
    X_test, alpha_test, zeta_test = extract_embeddings(model.encoder, make_loader("test"), device)

    print(f"\nShapes: train={X_tr.shape}  val={X_val.shape}  test={X_test.shape}")
    print(f"Embedding std (train): {X_tr.std(axis=0).mean():.4f}  (> 0.1 = healthy)")

    # ── Linear probe ─────────────────────────────────────────────────────────
    print("\n" + "═" * 55)
    print("LINEAR PROBE (Ridge, alpha=1.0)")
    print("═" * 55)
    lp_a_val, lp_a_test, _, _ = probe_and_knn(
        X_tr, X_val, X_test, alpha_tr, alpha_val, alpha_test, "alpha")
    lp_z_val, lp_z_test, _, _ = probe_and_knn(
        X_tr, X_val, X_test, zeta_tr,  zeta_val,  zeta_test,  "zeta")

    # ── kNN ───────────────────────────────────────────────────────────────────
    print("\n" + "═" * 55)
    print("kNN REGRESSION (k=20, cosine)")
    print("═" * 55)
    _, _, knn_a_val, knn_a_test = probe_and_knn(
        X_tr, X_val, X_test, alpha_tr, alpha_val, alpha_test, "alpha")
    _, _, knn_z_val, knn_z_test = probe_and_knn(
        X_tr, X_val, X_test, zeta_tr,  zeta_val,  zeta_test,  "zeta")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "═" * 55)
    print("SUMMARY  (MSE on z-score normalized targets)")
    print("═" * 55)
    print(f"  Linear Probe — alpha:  Val {lp_a_val:.4f}  Test {lp_a_test:.4f}")
    print(f"  Linear Probe — zeta:   Val {lp_z_val:.4f}  Test {lp_z_test:.4f}")
    print(f"  kNN          — alpha:  Val {knn_a_val:.4f}  Test {knn_a_test:.4f}")
    print(f"  kNN          — zeta:   Val {knn_z_val:.4f}  Test {knn_z_test:.4f}")
    print(f"\n  Lower is better.  Random baseline ≈ 1.0 (normalized)")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    payload = {
        "model":         "vicreg-v4",
        "checkpoint":    args.checkpoint,
        "train_samples": len(X_tr),
        "val_samples":   len(X_val),
        "test_samples":  len(X_test),
        "embedding_std": float(X_tr.std(axis=0).mean()),
        "val": {
            "linear_alpha_mse": lp_a_val,
            "linear_zeta_mse":  lp_z_val,
            "knn_alpha_mse":    knn_a_val,
            "knn_zeta_mse":     knn_z_val,
        },
        "test": {
            "linear_alpha_mse": lp_a_test,
            "linear_zeta_mse":  lp_z_test,
            "knn_alpha_mse":    knn_a_test,
            "knn_zeta_mse":     knn_z_test,
        },
    }
    out_path = os.path.join(
        os.path.dirname(args.checkpoint),
        f"eval_{os.path.splitext(os.path.basename(args.checkpoint))[0]}.json",
    )
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved results → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()
    main(args, CONFIG)
