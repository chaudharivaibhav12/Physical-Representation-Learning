"""
Evaluation: Linear Probe + kNN Regression
==========================================
Evaluates frozen VICReg encoder by predicting alpha and zeta using:
  1. Single linear layer  (required by project spec)
  2. kNN regression       (required by project spec)

Both reported with MSE on z-score normalized targets.

Usage:
  python evaluate.py --checkpoint /scratch/sb10583/checkpoints/vicreg/best.pt
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from model   import VICReg
from dataset import ActiveMatterEval


CONFIG = {
    "data_dir":    "/scratch/sb10583/data/data",
    "crop_size":   224,
    "batch_size":  8,

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

    # Eval
    "k":            20,
    "probe_epochs": 100,
    "probe_lr":     1e-3,
}


# ─────────────────────────────────────────────
# Extract Embeddings
# ─────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(encoder, loader, device):
    """Run all samples through the frozen encoder. Returns (N, D), alphas, zetas."""
    encoder.eval()
    embeddings, alphas, zetas = [], [], []

    for batch in loader:
        x  = batch["x"].to(device)
        z  = encoder.forward_pooled(x)       # (B, embed_dim)
        embeddings.append(z.cpu().numpy())
        alphas.append(batch["alpha"].numpy())
        zetas.append(batch["zeta"].numpy())

    return (
        np.concatenate(embeddings, axis=0),
        np.concatenate(alphas,     axis=0),
        np.concatenate(zetas,      axis=0),
    )


# ─────────────────────────────────────────────
# Linear Probe
# ─────────────────────────────────────────────

class LinearProbe(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.linear = nn.Linear(embed_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


def train_linear_probe(
    train_emb, train_targets,
    val_emb,   val_targets,
    embed_dim, epochs=100, lr=1e-3, label="alpha",
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train = torch.tensor(train_emb,     dtype=torch.float32, device=device)
    y_train = torch.tensor(train_targets, dtype=torch.float32, device=device)
    X_val   = torch.tensor(val_emb,       dtype=torch.float32, device=device)
    y_val   = torch.tensor(val_targets,   dtype=torch.float32, device=device)

    # Z-score normalize targets
    mean = y_train.mean()
    std  = y_train.std() + 1e-6
    y_train_n = (y_train - mean) / std
    y_val_n   = (y_val   - mean) / std

    probe     = LinearProbe(embed_dim).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_mse = float("inf")
    for epoch in range(epochs):
        probe.train()
        pred = probe(X_train)
        loss = F.mse_loss(pred, y_train_n)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 20 == 0:
            probe.eval()
            with torch.no_grad():
                val_mse = F.mse_loss(probe(X_val), y_val_n).item()
            best_val_mse = min(best_val_mse, val_mse)

    probe.eval()
    with torch.no_grad():
        train_mse = F.mse_loss(probe(X_train), y_train_n).item()
        val_mse   = F.mse_loss(probe(X_val),   y_val_n).item()

    print(f"  Linear Probe [{label:5s}] → Train MSE: {train_mse:.4f} | Val MSE: {val_mse:.4f}")
    return {"train_mse": train_mse, "val_mse": val_mse}


# ─────────────────────────────────────────────
# kNN Regression
# ─────────────────────────────────────────────

def evaluate_knn(train_emb, train_targets, val_emb, val_targets, k=20, label="alpha") -> dict:
    mean = train_targets.mean()
    std  = train_targets.std() + 1e-6
    y_train_n = (train_targets - mean) / std
    y_val_n   = (val_targets   - mean) / std

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(train_emb)
    X_val   = scaler.transform(val_emb)

    knn = KNeighborsRegressor(n_neighbors=k, metric="cosine", n_jobs=-1)
    knn.fit(X_train, y_train_n)

    train_mse = mean_squared_error(y_train_n, knn.predict(X_train))
    val_mse   = mean_squared_error(y_val_n,   knn.predict(X_val))

    print(f"  kNN (k={k:2d}) [{label:5s}]    → Train MSE: {train_mse:.4f} | Val MSE: {val_mse:.4f}")
    return {"train_mse": train_mse, "val_mse": val_mse}


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def evaluate(args, cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── Load model ────────────────────────────────────────────────────
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

    print(f"[LOAD] Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.encoder.load_state_dict(ckpt["encoder"])
    model.eval()

    for param in model.encoder.parameters():
        param.requires_grad = False

    print("[LOAD] Encoder frozen.\n")

    # ── Datasets ──────────────────────────────────────────────────────
    train_ds = ActiveMatterEval(cfg["data_dir"], split="train",  crop_size=cfg["crop_size"])
    val_ds   = ActiveMatterEval(cfg["data_dir"], split="valid",  crop_size=cfg["crop_size"])
    test_ds  = ActiveMatterEval(cfg["data_dir"], split="test",   crop_size=cfg["crop_size"])

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=cfg["batch_size"], shuffle=False, num_workers=2)

    # ── Extract embeddings ────────────────────────────────────────────
    print("Extracting train embeddings...")
    train_emb, train_alpha, train_zeta = extract_embeddings(model.encoder, train_loader, device)

    print("Extracting val embeddings...")
    val_emb, val_alpha, val_zeta = extract_embeddings(model.encoder, val_loader, device)

    print("Extracting test embeddings...")
    test_emb, test_alpha, test_zeta = extract_embeddings(model.encoder, test_loader, device)

    print(f"\nShapes: train={train_emb.shape}  val={val_emb.shape}  test={test_emb.shape}")
    print(f"Embedding std (train): {train_emb.std(axis=0).mean():.4f}  (> 0.1 = healthy)\n")

    # ── Linear Probe ──────────────────────────────────────────────────
    print("=" * 55)
    print("LINEAR PROBE")
    print("=" * 55)
    lp_alpha = train_linear_probe(
        train_emb, train_alpha, val_emb, val_alpha,
        cfg["embed_dim"], epochs=cfg["probe_epochs"], label="alpha",
    )
    lp_zeta = train_linear_probe(
        train_emb, train_zeta, val_emb, val_zeta,
        cfg["embed_dim"], epochs=cfg["probe_epochs"], label="zeta",
    )

    # ── kNN ───────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("kNN REGRESSION")
    print("=" * 55)
    knn_alpha = evaluate_knn(train_emb, train_alpha, val_emb, val_alpha, k=cfg["k"], label="alpha")
    knn_zeta  = evaluate_knn(train_emb, train_zeta,  val_emb, val_zeta,  k=cfg["k"], label="zeta")

    # ── Test set (final eval only) ────────────────────────────────────
    if args.test:
        print("\n" + "=" * 55)
        print("TEST SET (final evaluation)")
        print("=" * 55)
        evaluate_knn(train_emb, train_alpha, test_emb, test_alpha, k=cfg["k"], label="alpha")
        evaluate_knn(train_emb, train_zeta,  test_emb, test_zeta,  k=cfg["k"], label="zeta")
        train_linear_probe(train_emb, train_alpha, test_emb, test_alpha, cfg["embed_dim"], label="alpha")
        train_linear_probe(train_emb, train_zeta,  test_emb, test_zeta,  cfg["embed_dim"], label="zeta")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("SUMMARY  (Validation MSE, z-score normalized targets)")
    print("=" * 55)
    print(f"  Linear Probe — alpha: {lp_alpha['val_mse']:.4f}")
    print(f"  Linear Probe — zeta:  {lp_zeta['val_mse']:.4f}")
    print(f"  kNN          — alpha: {knn_alpha['val_mse']:.4f}")
    print(f"  kNN          — zeta:  {knn_zeta['val_mse']:.4f}")
    print(f"\n  Lower is better.  Random baseline ≈ 1.0 (normalized)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test", action="store_true", help="Also evaluate on test set")
    args = parser.parse_args()
    evaluate(args, CONFIG)
