"""
evaluate.py  —  Resumable Linear Probe + kNN for I-JEPA
======================================================

Key Features:
- Embedding caching (critical for Slurm preemption)
- Stage-based execution: extract / probe / knn
- Uses TARGET encoder (correct per JEPA paper)
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

from model import IJEPA

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import ActiveMatterDataset


# ─────────────────────────────────────────────
# Config (must match training)
# ─────────────────────────────────────────────

CFG = {
    "data_dir":    "/scratch/ok2287/data/active_matter/data",
    "stride":      1,
    "num_frames":  4,
    "batch_size":  32,

    "in_channels":      11,
    "img_size":         224,
    "patch_size":       32,
    "encoder_dim":      384,
    "encoder_depth":    12,
    "encoder_heads":    6,
    "predictor_dim":    192,
    "predictor_depth":  6,
    "predictor_heads":  4,

    "k": 20,
    "probe_lr":     1e-3,
    "probe_epochs": 30,
    "probe_batch":  64,
}


# ─────────────────────────────────────────────
# Utils
# ─────────────────────────────────────────────

def subsample_frames(x, num_frames):
    T = x.shape[2]
    if T == num_frames:
        return x
    idx = torch.linspace(0, T - 1, num_frames).long()
    return x[:, :, idx, :, :]


@torch.no_grad()
def extract_embeddings(model, loader, device, split_name="", cache_file=None):
    model.eval()

    embeddings = []
    alphas = []
    zetas = []

    start_idx = 0

    if cache_file and os.path.exists(cache_file):
        print(f"Resuming partial embeddings for {split_name}")
        data = np.load(cache_file, allow_pickle=True)
        embeddings = list(data["embeddings"])
        alphas     = list(data["alphas"])
        zetas      = list(data["zetas"])
        start_idx  = int(data["last_idx"])

    for i, batch in enumerate(loader):
        if i < start_idx:
            continue

        x = subsample_frames(batch["context"].to(device), CFG["num_frames"])
        z = model.encode(x)

        embeddings.append(z.cpu().numpy())
        alphas.append(batch["alpha"].numpy())
        zetas.append(batch["zeta"].numpy())

        # 🔥 SAVE PROGRESS EVERY ~20 BATCHES
        if i % 20 == 0:
            np.savez(cache_file,
                embeddings=np.array(embeddings, dtype=object),
                alphas=np.array(alphas, dtype=object),
                zetas=np.array(zetas, dtype=object),
                last_idx=i
            )
            print(f"[{split_name}] checkpoint saved at batch {i}")

    return (
        np.concatenate(embeddings),
        np.concatenate(alphas),
        np.concatenate(zetas),
    )

# ─────────────────────────────────────────────
# Linear Probe
# ─────────────────────────────────────────────

def train_linear_probe(train_emb, train_t, val_emb, val_t, test_emb, test_t, label):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    y_train, y_val, y_test = zscore(train_t, val_t, test_t)

    X_train = torch.tensor(train_emb, device=device, dtype=torch.float32)
    y_train = torch.tensor(y_train, device=device, dtype=torch.float32)

    X_val = torch.tensor(val_emb, device=device, dtype=torch.float32)
    y_val = torch.tensor(y_val, device=device, dtype=torch.float32)

    X_test = torch.tensor(test_emb, device=device, dtype=torch.float32)
    y_test = torch.tensor(y_test, device=device, dtype=torch.float32)

    probe = nn.Linear(CFG["encoder_dim"], 1).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=CFG["probe_lr"])

    best_val = float("inf")
    best_state = None

    for ep in range(CFG["probe_epochs"]):
        probe.train()
        perm = torch.randperm(X_train.shape[0])

        for i in range(0, X_train.shape[0], CFG["probe_batch"]):
            idx = perm[i:i+CFG["probe_batch"]]
            pred = probe(X_train[idx]).squeeze()
            loss = F.mse_loss(pred, y_train[idx])

            opt.zero_grad()
            loss.backward()
            opt.step()

        probe.eval()
        with torch.no_grad():
            val_mse = F.mse_loss(probe(X_val).squeeze(), y_val).item()

        if val_mse < best_val:
            best_val = val_mse
            best_state = probe.state_dict()

        if (ep+1) % 10 == 0:
            print(f"[{label}] Epoch {ep+1} | Val MSE: {val_mse:.4f}")

    probe.load_state_dict(best_state)

    with torch.no_grad():
        return {
            "train": F.mse_loss(probe(X_train).squeeze(), y_train).item(),
            "val":   F.mse_loss(probe(X_val).squeeze(),   y_val).item(),
            "test":  F.mse_loss(probe(X_test).squeeze(),  y_test).item(),
        }


# ─────────────────────────────────────────────
# kNN
# ─────────────────────────────────────────────

def run_knn(train_emb, train_t, val_emb, val_t, test_emb, test_t, label):

    y_train, y_val, y_test = zscore(train_t, val_t, test_t)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_emb)
    X_val   = scaler.transform(val_emb)
    X_test  = scaler.transform(test_emb)

    knn = KNeighborsRegressor(n_neighbors=CFG["k"], metric="cosine", n_jobs=-1)
    knn.fit(X_train, y_train)

    return {
        "train": mean_squared_error(y_train, knn.predict(X_train)),
        "val":   mean_squared_error(y_val,   knn.predict(X_val)),
        "test":  mean_squared_error(y_test,  knn.predict(X_test)),
    }


# ─────────────────────────────────────────────
# Collapse check
# ─────────────────────────────────────────────

def check_collapse(emb):
    std = emb.std(axis=0)
    avg = std.mean()
    dead = (std < 0.01).sum()
    status = "HEALTHY" if avg > 0.1 else "COLLAPSE"

    print(f"Collapse: avg_std={avg:.4f} | dead_dims={dead} → {status}")
    return status


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading model...")
    model = IJEPA(**{k: CFG[k] for k in [
        "in_channels","img_size","patch_size","num_frames",
        "encoder_dim","encoder_depth","encoder_heads",
        "predictor_dim","predictor_depth","predictor_heads"
    ]}).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    # ──────────────── DATA ────────────────
    train_ds = ActiveMatterDataset(CFG["data_dir"], "train", stride=1, noise_std=0)
    val_ds   = ActiveMatterDataset(CFG["data_dir"], "valid", stride=1, noise_std=0)
    test_ds  = ActiveMatterDataset(CFG["data_dir"], "test",  stride=1, noise_std=0)

    loader_args = dict(batch_size=CFG["batch_size"], shuffle=False, num_workers=8)

    train_loader = DataLoader(train_ds, **loader_args)
    val_loader   = DataLoader(val_ds,   **loader_args)
    test_loader  = DataLoader(test_ds,  **loader_args)

    # ──────────────── EMBEDDINGS ────────────────
    if os.path.exists(args.cache):
        print(f"Loading cached embeddings from {args.cache}")
        data = np.load(args.cache)

        train_emb = data["train_emb"]
        val_emb   = data["val_emb"]
        test_emb  = data["test_emb"]

        train_alpha = data["train_alpha"]
        val_alpha   = data["val_alpha"]
        test_alpha  = data["test_alpha"]

        train_zeta = data["train_zeta"]
        val_zeta   = data["val_zeta"]
        test_zeta  = data["test_zeta"]

    else:
        print("Extracting embeddings...")

        train_emb, train_alpha, train_zeta = extract_embeddings(model, train_loader, device, "train")
        val_emb,   val_alpha,   val_zeta   = extract_embeddings(model, val_loader,   device, "val")
        test_emb,  test_alpha,  test_zeta  = extract_embeddings(model, test_loader,  device, "test")

        print(f"Saving embeddings → {args.cache}")
        np.savez(args.cache,
            train_emb=train_emb, val_emb=val_emb, test_emb=test_emb,
            train_alpha=train_alpha, val_alpha=val_alpha, test_alpha=test_alpha,
            train_zeta=train_zeta, val_zeta=val_zeta, test_zeta=test_zeta
        )

    check_collapse(train_emb)

    # ──────────────── LINEAR PROBE ────────────────
    if args.stage in ["probe", "all"]:
        print("\nRunning Linear Probe...")
        lp_alpha = train_linear_probe(train_emb, train_alpha, val_emb, val_alpha, test_emb, test_alpha, "alpha")
        lp_zeta  = train_linear_probe(train_emb, train_zeta,  val_emb, val_zeta,  test_emb, test_zeta,  "zeta")

        print("Linear Probe Results:", lp_alpha, lp_zeta)

    # ──────────────── KNN ────────────────
    if args.stage in ["knn", "all"]:
        print("\nRunning kNN...")
        knn_alpha = run_knn(train_emb, train_alpha, val_emb, val_alpha, test_emb, test_alpha, "alpha")
        knn_zeta  = run_knn(train_emb, train_zeta,  val_emb, val_zeta,  test_emb, test_zeta,  "zeta")

        print("kNN Results:", knn_alpha, knn_zeta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--cache", type=str, default="/scratch/ok2287/embeddings.npz")
    parser.add_argument("--stage", type=str, default="all")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    main(args)