"""
Evaluation: Linear Probe + kNN Regression
==========================================
Evaluates frozen encoder representations by predicting
alpha and zeta using:
  1. Single linear layer (as required by project spec)
  2. kNN regression      (as required by project spec)

Both evaluated with MSE loss on z-score normalized targets.

Usage:
  python evaluate.py --checkpoint /scratch/vc2836/DL/checkpoints/vit_jepa_v2/best.pt
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from model   import ViTJEPA
from dataset import ActiveMatterDataset


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return super().default(obj)


CONFIG = {
    "data_dir":        "/scratch/vc2836/DL/data/active_matter/data",
    "checkpoint_dir":  "/scratch/vc2836/DL/checkpoints/vit_jepa_v2",
    "num_frames":  16,
    "crop_size":   224,
    "stride":      4,
    "batch_size":  8,

    # Model (must match training config)
    "in_channels": 11,
    "embed_dim":   384,
    "depth":       6,
    "num_heads":   6,
    "patch_size":  32,
    "tubelet":     2,

    # kNN config
    "k":           20,

    # Linear probe config
    "probe_epochs": 50,
    "probe_lr":     1e-3,
}


# ─────────────────────────────────────────────
# Extract Embeddings
# ─────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(encoder, loader, device):
    """
    Run all samples through the frozen encoder.
    Returns embeddings, alpha values, zeta values.
    """
    encoder.eval()
    embeddings = []
    alphas     = []
    zetas      = []

    for batch in loader:
        ctx = batch["context"].to(device)
        z   = encoder(ctx)                    # (B, num_tokens, D)
        z   = z.mean(dim=1)                   # (B, D) — mean pool over patch tokens
        embeddings.append(z.cpu().numpy())
        alphas.append(batch["alpha"].numpy())
        zetas.append(batch["zeta"].numpy())

    embeddings = np.concatenate(embeddings, axis=0)  # (N, D)
    alphas     = np.concatenate(alphas,     axis=0)  # (N,)
    zetas      = np.concatenate(zetas,      axis=0)  # (N,)
    return embeddings, alphas, zetas


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
    train_emb: np.ndarray,
    train_targets: np.ndarray,
    val_emb: np.ndarray,
    val_targets: np.ndarray,
    embed_dim: int,
    epochs: int = 50,
    lr: float = 1e-3,
    label: str = "alpha",
    test_emb: np.ndarray = None,
    test_targets: np.ndarray = None,
) -> dict:
    """
    Train a single linear layer on frozen embeddings.
    Returns train and val MSE.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert to tensors
    X_train = torch.tensor(train_emb,     dtype=torch.float32, device=device)
    y_train = torch.tensor(train_targets, dtype=torch.float32, device=device)
    X_val   = torch.tensor(val_emb,       dtype=torch.float32, device=device)
    y_val   = torch.tensor(val_targets,   dtype=torch.float32, device=device)

    # Z-score normalize targets as required by spec
    mean = y_train.mean()
    std  = y_train.std() + 1e-6
    y_train_norm = (y_train - mean) / std
    y_val_norm   = (y_val   - mean) / std

    # Build and train probe
    probe     = LinearProbe(embed_dim).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

    best_val_mse = float("inf")
    best_state   = None
    for _ in range(epochs):
        probe.train()
        pred_train = probe(X_train)
        loss       = nn.functional.mse_loss(pred_train, y_train_norm)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        probe.eval()
        with torch.no_grad():
            val_mse = nn.functional.mse_loss(probe(X_val), y_val_norm).item()
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state   = {k: v.clone() for k, v in probe.state_dict().items()}

    probe.load_state_dict(best_state)

    # Final evaluation using best-val checkpoint
    probe.eval()
    with torch.no_grad():
        train_mse = nn.functional.mse_loss(probe(X_train), y_train_norm).item()
        val_mse   = nn.functional.mse_loss(probe(X_val),   y_val_norm).item()
        test_mse  = None
        if test_emb is not None and test_targets is not None:
            X_test   = torch.tensor(test_emb,     dtype=torch.float32, device=device)
            y_test   = torch.tensor(test_targets, dtype=torch.float32, device=device)
            y_test_norm = (y_test - mean) / std
            test_mse = nn.functional.mse_loss(probe(X_test), y_test_norm).item()

    msg = f"  Linear Probe [{label}] → Train MSE: {train_mse:.4f} | Val MSE: {val_mse:.4f}"
    if test_mse is not None:
        msg += f" | Test MSE: {test_mse:.4f}"
    print(msg)
    return {"train_mse": train_mse, "val_mse": val_mse, "test_mse": test_mse}


# ─────────────────────────────────────────────
# kNN Regression
# ─────────────────────────────────────────────

def evaluate_knn(
    train_emb: np.ndarray,
    train_targets: np.ndarray,
    val_emb: np.ndarray,
    val_targets: np.ndarray,
    k: int = 20,
    label: str = "alpha",
    test_emb: np.ndarray = None,
    test_targets: np.ndarray = None,
) -> dict:
    """
    kNN regression on frozen embeddings.
    Targets are z-score normalized as required by spec.
    """
    # Z-score normalize targets
    mean = train_targets.mean()
    std  = train_targets.std() + 1e-6
    y_train_norm = (train_targets - mean) / std
    y_val_norm   = (val_targets   - mean) / std

    # Normalize embeddings for kNN distance computation
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(train_emb)
    X_val   = scaler.transform(val_emb)

    knn = KNeighborsRegressor(n_neighbors=k, metric="cosine", n_jobs=-1)
    knn.fit(X_train, y_train_norm)

    train_mse = mean_squared_error(y_train_norm, knn.predict(X_train))
    val_mse   = mean_squared_error(y_val_norm,   knn.predict(X_val))
    test_mse  = None
    if test_emb is not None and test_targets is not None:
        y_test_norm = (test_targets - mean) / std
        X_test = scaler.transform(test_emb)
        test_mse = mean_squared_error(y_test_norm, knn.predict(X_test))

    msg = f"  kNN (k={k}) [{label}]    → Train MSE: {train_mse:.4f} | Val MSE: {val_mse:.4f}"
    if test_mse is not None:
        msg += f" | Test MSE: {test_mse:.4f}"
    print(msg)
    return {"train_mse": train_mse, "val_mse": val_mse, "test_mse": test_mse}


# ─────────────────────────────────────────────
# Main Evaluation
# ─────────────────────────────────────────────

def evaluate(args, cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── Load model ───────────────────────────────────────────────────
    model = ViTJEPA(
        in_channels = cfg["in_channels"],
        embed_dim   = cfg["embed_dim"],
        depth       = cfg["depth"],
        num_heads   = cfg["num_heads"],
        img_size    = cfg["crop_size"],
        patch_size  = cfg["patch_size"],
        tubelet     = cfg["tubelet"],
        num_frames  = cfg["num_frames"],
    ).to(device)

    ckpt_path = args.checkpoint or os.path.join(cfg["checkpoint_dir"], "best.pt")
    print(f"[LOAD] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.encoder.load_state_dict(ckpt["encoder"])
    model.eval()

    # Freeze encoder completely
    for param in model.encoder.parameters():
        param.requires_grad = False

    print(f"[LOAD] Encoder frozen. Extracting embeddings...\n")

    # ── Datasets ─────────────────────────────────────────────────────
    train_dataset = ActiveMatterDataset(cfg["data_dir"], split="train", stride=cfg["stride"], noise_std=0.0)
    val_dataset   = ActiveMatterDataset(cfg["data_dir"], split="valid", stride=cfg["stride"], noise_std=0.0)
    test_dataset  = ActiveMatterDataset(cfg["data_dir"], split="test",  stride=cfg["stride"], noise_std=0.0)

    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=4)
    val_loader   = DataLoader(val_dataset,   batch_size=cfg["batch_size"], shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_dataset,  batch_size=cfg["batch_size"], shuffle=False, num_workers=2)

    # ── Extract embeddings ───────────────────────────────────────────
    print("Extracting train embeddings...")
    train_emb, train_alpha, train_zeta = extract_embeddings(model.encoder, train_loader, device)

    print("Extracting val embeddings...")
    val_emb, val_alpha, val_zeta = extract_embeddings(model.encoder, val_loader, device)

    print("Extracting test embeddings...")
    test_emb, test_alpha, test_zeta = extract_embeddings(model.encoder, test_loader, device)

    print(f"\nEmbedding shapes: train={train_emb.shape}, val={val_emb.shape}, test={test_emb.shape}")
    print(f"Embedding std (train): {train_emb.std(axis=0).mean():.4f}  (>0.1 = healthy)\n")

    # ── Linear Probe ─────────────────────────────────────────────────
    print("=" * 50)
    print("LINEAR PROBE RESULTS")
    print("=" * 50)

    lp_alpha = train_linear_probe(
        train_emb, train_alpha,
        val_emb,   val_alpha,
        cfg["embed_dim"],
        epochs=cfg["probe_epochs"],
        label="alpha",
        test_emb=test_emb, test_targets=test_alpha,
    )
    lp_zeta = train_linear_probe(
        train_emb, train_zeta,
        val_emb,   val_zeta,
        cfg["embed_dim"],
        epochs=cfg["probe_epochs"],
        label="zeta",
        test_emb=test_emb, test_targets=test_zeta,
    )

    # ── kNN Regression ───────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("kNN REGRESSION RESULTS")
    print("=" * 50)

    knn_alpha = evaluate_knn(train_emb, train_alpha, val_emb, val_alpha, k=cfg["k"], label="alpha",
                             test_emb=test_emb, test_targets=test_alpha)
    knn_zeta  = evaluate_knn(train_emb, train_zeta,  val_emb, val_zeta,  k=cfg["k"], label="zeta",
                             test_emb=test_emb, test_targets=test_zeta)

    # ── Final Summary ────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("FINAL SUMMARY (z-score normalized MSE)")
    print("=" * 50)
    print(f"  Linear Probe — alpha: val={lp_alpha['val_mse']:.4f}  test={lp_alpha['test_mse']:.4f}")
    print(f"  Linear Probe — zeta:  val={lp_zeta['val_mse']:.4f}  test={lp_zeta['test_mse']:.4f}")
    print(f"  kNN          — alpha: val={knn_alpha['val_mse']:.4f}  test={knn_alpha['test_mse']:.4f}")
    print(f"  kNN          — zeta:  val={knn_zeta['val_mse']:.4f}  test={knn_zeta['test_mse']:.4f}")
    print(f"\n  Lower is better. Random baseline ≈ 1.0 (normalized)")

    # ── Save JSON ────────────────────────────────────────────────────
    if args.output_json:
        results = {
            "checkpoint": str(ckpt_path),
            "embed_dim": int(train_emb.shape[1]),
            "n_train": int(train_emb.shape[0]),
            "n_val":   int(val_emb.shape[0]),
            "n_test":  int(test_emb.shape[0]),
            "embed_std": {
                "train": float(train_emb.std(axis=0).mean()),
                "val":   float(val_emb.std(axis=0).mean()),
                "test":  float(test_emb.std(axis=0).mean()),
            },
            "linear_probe": {"alpha": lp_alpha, "zeta": lp_zeta},
            "knn": {"k": cfg["k"], "alpha": knn_alpha, "zeta": knn_zeta},
        }
        out_path = args.output_json
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, cls=_NumpyEncoder)
        print(f"\n[save] results → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (default: best.pt in CONFIG['checkpoint_dir'])")
    parser.add_argument("--output-json", type=str, default=None, help="Optional path to save results as JSON")
    args = parser.parse_args()
    evaluate(args, CONFIG)
