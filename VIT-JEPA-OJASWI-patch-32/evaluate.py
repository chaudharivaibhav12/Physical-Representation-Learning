"""
Evaluation: Linear Probe + kNN Regression
==========================================
Evaluates frozen encoder representations by predicting
alpha and zeta using:
  1. Single linear layer  (required by project spec)
  2. kNN regression       (required by project spec)

Both evaluated with MSE loss on z-score normalized targets.

Usage:
  # Evaluate best checkpoint
  python evaluate.py --checkpoint /scratch/ok2287/checkpoints/vit_jepa/best.pt

  # Evaluate specific epoch checkpoint
  python evaluate.py --checkpoint /scratch/ok2287/checkpoints/vit_jepa/epoch_30.pt

  # Evaluate and save results to file
  python evaluate.py --checkpoint /scratch/ok2287/checkpoints/vit_jepa/best.pt --save
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from model   import ViTJEPA
from dataset import ActiveMatterDataset


# ─────────────────────────────────────────────
# Config — must match training config
# ─────────────────────────────────────────────

CFG = {
    "data_dir":      "/scratch/ok2287/data/active_matter/data",
    "num_frames":    16,
    "crop_size":     224,
    "stride":        4,
    "batch_size":    8,

    # Model (must match training)
    "in_channels":   11,
    "embed_dim":     384,
    "depth":         8,
    "num_heads":     6,
    "patch_size":    32,
    "tubelet":       2,
    "predictor_dim": 192,
    "pred_depth":    2,
    "pred_heads":    4,

    # kNN
    "k":             20,

    # Linear probe
    "probe_lr":      1e-3,
    "probe_epochs":  100,
    "probe_batch":   64,
}


# ─────────────────────────────────────────────
# 1. Extract Embeddings from Frozen Encoder
# ─────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(encoder, loader, device, split_name=""):
    """
    Run all samples through the frozen encoder.
    Returns embeddings, alpha values, zeta values.
    """
    encoder.eval()
    embeddings = []
    alphas     = []
    zetas      = []

    print(f"  Extracting {split_name} embeddings...")
    for i, batch in enumerate(loader):
        ctx = batch["context"].to(device)
        z   = encoder.forward_pooled(ctx)      # (B, D)
        embeddings.append(z.cpu().numpy())
        alphas.append(batch["alpha"].numpy())
        zetas.append(batch["zeta"].numpy())

        if (i + 1) % 50 == 0:
            print(f"    Batch {i+1}/{len(loader)}")

    embeddings = np.concatenate(embeddings, axis=0)   # (N, D)
    alphas     = np.concatenate(alphas,     axis=0)   # (N,)
    zetas      = np.concatenate(zetas,      axis=0)   # (N,)

    print(f"  {split_name}: {embeddings.shape[0]} samples, dim={embeddings.shape[1]}")
    print(f"  Embedding std (avg across dims): {embeddings.std(axis=0).mean():.4f}")
    return embeddings, alphas, zetas


# ─────────────────────────────────────────────
# 2. Z-score Normalize Targets
# ─────────────────────────────────────────────

def zscore_normalize(train_targets, val_targets, test_targets):
    """
    Z-score normalize targets using training set statistics.
    As required by project spec.
    """
    mean = train_targets.mean()
    std  = train_targets.std() + 1e-6
    return (
        (train_targets - mean) / std,
        (val_targets   - mean) / std,
        (test_targets  - mean) / std,
        mean, std,
    )


# ─────────────────────────────────────────────
# 3. Linear Probe
# ─────────────────────────────────────────────

def train_linear_probe(
    train_emb:     np.ndarray,
    train_targets: np.ndarray,
    val_emb:       np.ndarray,
    val_targets:   np.ndarray,
    test_emb:      np.ndarray,
    test_targets:  np.ndarray,
    embed_dim:     int,
    label:         str = "alpha",
    epochs:        int = 100,
    lr:            float = 1e-3,
    batch_size:    int = 64,
) -> dict:
    """
    Train a single nn.Linear(embed_dim, 1) on frozen embeddings.
    As required by project spec — no MLP, no attention, just one linear layer.
    Targets are z-score normalized.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Z-score normalize targets
    y_train_norm, y_val_norm, y_test_norm, mean, std = zscore_normalize(
        train_targets, val_targets, test_targets
    )

    # Convert to tensors
    X_train = torch.tensor(train_emb,    dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train_norm, dtype=torch.float32, device=device)
    X_val   = torch.tensor(val_emb,      dtype=torch.float32, device=device)
    y_val   = torch.tensor(y_val_norm,   dtype=torch.float32, device=device)
    X_test  = torch.tensor(test_emb,     dtype=torch.float32, device=device)
    y_test  = torch.tensor(y_test_norm,  dtype=torch.float32, device=device)

    # Single linear layer — as required by spec
    probe     = nn.Linear(embed_dim, 1).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

    best_val_mse  = float("inf")
    best_state    = None

    for epoch in range(epochs):
        probe.train()

        # Mini-batch training
        perm     = torch.randperm(X_train.shape[0])
        ep_loss  = 0.0
        n_batches = 0

        for i in range(0, X_train.shape[0], batch_size):
            idx     = perm[i:i + batch_size]
            pred    = probe(X_train[idx]).squeeze(-1)
            loss    = F.mse_loss(pred, y_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep_loss  += loss.item()
            n_batches += 1

        # Validation check
        probe.eval()
        with torch.no_grad():
            val_mse = F.mse_loss(probe(X_val).squeeze(-1), y_val).item()

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state   = {k: v.clone() for k, v in probe.state_dict().items()}

        if (epoch + 1) % 20 == 0:
            print(f"    [{label}] Epoch {epoch+1}/{epochs} | "
                  f"Train MSE: {ep_loss/n_batches:.4f} | Val MSE: {val_mse:.4f}")

    # Load best probe and evaluate
    probe.load_state_dict(best_state)
    probe.eval()
    with torch.no_grad():
        train_mse = F.mse_loss(probe(X_train).squeeze(-1), y_train).item()
        val_mse   = F.mse_loss(probe(X_val).squeeze(-1),   y_val).item()
        test_mse  = F.mse_loss(probe(X_test).squeeze(-1),  y_test).item()

    print(f"  Linear Probe [{label}]:")
    print(f"    Train MSE: {train_mse:.4f}")
    print(f"    Val   MSE: {val_mse:.4f}")
    print(f"    Test  MSE: {test_mse:.4f}")

    return {
        "train_mse": train_mse,
        "val_mse":   val_mse,
        "test_mse":  test_mse,
    }


# ─────────────────────────────────────────────
# 4. kNN Regression
# ─────────────────────────────────────────────

def evaluate_knn(
    train_emb:     np.ndarray,
    train_targets: np.ndarray,
    val_emb:       np.ndarray,
    val_targets:   np.ndarray,
    test_emb:      np.ndarray,
    test_targets:  np.ndarray,
    k:             int = 20,
    label:         str = "alpha",
) -> dict:
    """
    kNN regression on frozen embeddings.
    Targets z-score normalized as required by spec.
    """
    # Z-score normalize targets
    y_train_norm, y_val_norm, y_test_norm, _, _ = zscore_normalize(
        train_targets, val_targets, test_targets
    )

    # Normalize embeddings for distance computation
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(train_emb)
    X_val   = scaler.transform(val_emb)
    X_test  = scaler.transform(test_emb)

    knn = KNeighborsRegressor(n_neighbors=k, metric="cosine", n_jobs=-1)
    knn.fit(X_train, y_train_norm)

    train_mse = mean_squared_error(y_train_norm, knn.predict(X_train))
    val_mse   = mean_squared_error(y_val_norm,   knn.predict(X_val))
    test_mse  = mean_squared_error(y_test_norm,  knn.predict(X_test))

    print(f"  kNN (k={k}) [{label}]:")
    print(f"    Train MSE: {train_mse:.4f}")
    print(f"    Val   MSE: {val_mse:.4f}")
    print(f"    Test  MSE: {test_mse:.4f}")

    return {
        "train_mse": train_mse,
        "val_mse":   val_mse,
        "test_mse":  test_mse,
    }


# ─────────────────────────────────────────────
# 5. Collapse Check
# ─────────────────────────────────────────────

def check_collapse(embeddings: np.ndarray) -> dict:
    """
    Check for representation collapse.
    Healthy representations have high variance across dimensions.
    """
    std_per_dim = embeddings.std(axis=0)
    avg_std     = std_per_dim.mean()
    min_std     = std_per_dim.min()
    dead_dims   = (std_per_dim < 0.01).sum()

    status = "✓ HEALTHY" if avg_std > 0.1 else "⚠ COLLAPSE RISK"
    print(f"  Collapse check: avg_std={avg_std:.4f} | "
          f"min_std={min_std:.4f} | "
          f"dead_dims={dead_dims}/{len(std_per_dim)}  {status}")

    return {
        "avg_std":   float(avg_std),
        "min_std":   float(min_std),
        "dead_dims": int(dead_dims),
        "status":    status,
    }


# ─────────────────────────────────────────────
# 6. Main Evaluation
# ─────────────────────────────────────────────

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}\n")

    # ── Load model ───────────────────────────────────────────────────
    model = ViTJEPA(
        in_channels   = CFG["in_channels"],
        embed_dim     = CFG["embed_dim"],
        depth         = CFG["depth"],
        num_heads     = CFG["num_heads"],
        img_size      = CFG["crop_size"],
        patch_size    = CFG["patch_size"],
        tubelet       = CFG["tubelet"],
        num_frames    = CFG["num_frames"],
        predictor_dim = CFG["predictor_dim"],
        pred_depth    = CFG["pred_depth"],
        pred_heads    = CFG["pred_heads"],
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.encoder.load_state_dict(ckpt["encoder"])
    epoch = ckpt.get("epoch", "?")
    print(f"Loaded encoder from epoch {epoch}\n")

    # Freeze encoder completely
    model.encoder.eval()
    for param in model.encoder.parameters():
        param.requires_grad = False

    # ── Datasets ─────────────────────────────────────────────────────
    num_workers = 4 if torch.cuda.is_available() else 0

    train_dataset = ActiveMatterDataset(
        CFG["data_dir"], split="train",
        stride=CFG["stride"], noise_std=0.0,  # no noise during eval
    )
    val_dataset = ActiveMatterDataset(
        CFG["data_dir"], split="valid",
        stride=CFG["stride"], noise_std=0.0,
    )
    test_dataset = ActiveMatterDataset(
        CFG["data_dir"], split="test",
        stride=CFG["stride"], noise_std=0.0,
    )

    train_loader = DataLoader(train_dataset, batch_size=CFG["batch_size"],
                              shuffle=False, num_workers=num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=CFG["batch_size"],
                              shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=CFG["batch_size"],
                              shuffle=False, num_workers=num_workers)

    # ── Extract embeddings ───────────────────────────────────────────
    print("=" * 55)
    print("EXTRACTING EMBEDDINGS")
    print("=" * 55)
    train_emb, train_alpha, train_zeta = extract_embeddings(
        model.encoder, train_loader, device, "train")
    val_emb,   val_alpha,   val_zeta   = extract_embeddings(
        model.encoder, val_loader,   device, "val")
    test_emb,  test_alpha,  test_zeta  = extract_embeddings(
        model.encoder, test_loader,  device, "test")

    # ── Collapse check ───────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("COLLAPSE CHECK")
    print("=" * 55)
    collapse_info = check_collapse(train_emb)

    # ── Linear Probe ─────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("LINEAR PROBE  (single nn.Linear — no MLP)")
    print("=" * 55)

    print("\nTraining linear probe for alpha...")
    lp_alpha = train_linear_probe(
        train_emb, train_alpha,
        val_emb,   val_alpha,
        test_emb,  test_alpha,
        embed_dim  = CFG["embed_dim"],
        label      = "alpha",
        epochs     = CFG["probe_epochs"],
        lr         = CFG["probe_lr"],
        batch_size = CFG["probe_batch"],
    )

    print("\nTraining linear probe for zeta...")
    lp_zeta = train_linear_probe(
        train_emb, train_zeta,
        val_emb,   val_zeta,
        test_emb,  test_zeta,
        embed_dim  = CFG["embed_dim"],
        label      = "zeta",
        epochs     = CFG["probe_epochs"],
        lr         = CFG["probe_lr"],
        batch_size = CFG["probe_batch"],
    )

    # ── kNN Regression ───────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"kNN REGRESSION  (k={CFG['k']}, metric=cosine)")
    print("=" * 55)

    print("\nkNN for alpha...")
    knn_alpha = evaluate_knn(
        train_emb, train_alpha,
        val_emb,   val_alpha,
        test_emb,  test_alpha,
        k=CFG["k"], label="alpha",
    )

    print("\nkNN for zeta...")
    knn_zeta = evaluate_knn(
        train_emb, train_zeta,
        val_emb,   val_zeta,
        test_emb,  test_zeta,
        k=CFG["k"], label="zeta",
    )

    # ── Final Summary ────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"FINAL SUMMARY  (epoch {epoch})")
    print("=" * 55)
    print(f"  Checkpoint:  {args.checkpoint}")
    print(f"  Embed dim:   {CFG['embed_dim']}")
    print(f"  Collapse:    {collapse_info['status']}")
    print()
    print(f"  {'Method':<20} {'Target':<8} {'Train':>8} {'Val':>8} {'Test':>8}")
    print(f"  {'-'*52}")
    print(f"  {'Linear Probe':<20} {'alpha':<8} {lp_alpha['train_mse']:>8.4f} {lp_alpha['val_mse']:>8.4f} {lp_alpha['test_mse']:>8.4f}")
    print(f"  {'Linear Probe':<20} {'zeta':<8} {lp_zeta['train_mse']:>8.4f} {lp_zeta['val_mse']:>8.4f} {lp_zeta['test_mse']:>8.4f}")
    print(f"  {'kNN':<20} {'alpha':<8} {knn_alpha['train_mse']:>8.4f} {knn_alpha['val_mse']:>8.4f} {knn_alpha['test_mse']:>8.4f}")
    print(f"  {'kNN':<20} {'zeta':<8} {knn_zeta['train_mse']:>8.4f} {knn_zeta['val_mse']:>8.4f} {knn_zeta['test_mse']:>8.4f}")
    print()
    print(f"  Note: MSE on z-score normalized targets.")
    print(f"        Random baseline ≈ 1.0  |  Perfect = 0.0")

    # ── Save Results ─────────────────────────────────────────────────
    results = {
        "checkpoint":    args.checkpoint,
        "epoch":         epoch,
        "collapse":      collapse_info,
        "linear_probe":  {"alpha": lp_alpha, "zeta": lp_zeta},
        "knn":           {"alpha": knn_alpha, "zeta": knn_zeta},
    }

    if args.save:
        out_path = args.checkpoint.replace(".pt", "_eval_results.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to: {out_path}")

    return results


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate ViT-JEPA encoder with linear probe and kNN"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to checkpoint .pt file (use best.pt for final evaluation)"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save results to JSON file next to checkpoint"
    )
    args = parser.parse_args()
    evaluate(args)