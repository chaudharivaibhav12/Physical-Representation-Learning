"""
Evaluation for ViT-JEPA: linear probe + kNN regression on alpha and zeta.

Usage:
  python eval.py --checkpoint /scratch/sb10583/checkpoints/vit-jepa-sarvesh/best.pt
  python eval.py --checkpoint best.pt --split test
"""

import argparse
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from dataset import ActiveMatterDataset
from model   import ViTJEPA


@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    zs, alphas, zetas = [], [], []
    for batch in loader:
        frames = batch["frames"].to(device)
        with torch.autocast(
            device_type="cuda" if device.type == "cuda" else "cpu",
            dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        ):
            z = model.encode(frames)
        zs.append(z.float().cpu().numpy())
        alphas.append(batch["alpha"].numpy())
        zetas.append(batch["zeta"].numpy())
    return (
        np.concatenate(zs),
        np.concatenate(alphas),
        np.concatenate(zetas),
    )


def evaluate(cfg, checkpoint_path, split, batch_size=32):
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda  = torch.cuda.is_available()
    num_workers = 4 if use_cuda else 0

    # ── Load model ────────────────────────────────────────────────────────────
    m = cfg["model"]
    model = ViTJEPA(
        in_channels=m["in_channels"],
        embed_dim=m["embed_dim"],
        depth=m["depth"],
        num_heads=m["num_heads"],
        mlp_ratio=m["mlp_ratio"],
        patch_size=m["patch_size"],
        tubelet=m["tubelet"],
        num_frames=m["num_frames"],
        pred_dim=m["pred_dim"],
        pred_depth=m["pred_depth"],
        pred_heads=m["pred_heads"],
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    print(f"loaded checkpoint: {checkpoint_path}  [encoder frozen]")

    # ── Extract embeddings ────────────────────────────────────────────────────
    d = cfg["data"]

    def make_loader(sp):
        ds = ActiveMatterDataset(d["data_dir"], split=sp, stride=d["stride"], noise_std=0.0, augment=False)
        return DataLoader(ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=use_cuda)

    print("extracting train embeddings...")
    Z_tr, alpha_tr, zeta_tr = extract_embeddings(model, make_loader("train"), device)

    print(f"extracting {split} embeddings...")
    Z_ev, alpha_ev, zeta_ev = extract_embeddings(model, make_loader(split), device)

    # Normalize labels (z-score from train set)
    def norm_labels(y_tr, y_ev):
        mu, sigma = y_tr.mean(), y_tr.std() + 1e-8
        return (y_tr - mu) / sigma, (y_ev - mu) / sigma

    alpha_tr_n, alpha_ev_n = norm_labels(alpha_tr, alpha_ev)
    zeta_tr_n,  zeta_ev_n  = norm_labels(zeta_tr,  zeta_ev)

    # Normalize embeddings (zero-mean unit-variance per dimension)
    scaler = StandardScaler().fit(Z_tr)
    Z_tr_s = scaler.transform(Z_tr)
    Z_ev_s = scaler.transform(Z_ev)

    results = {}

    # ── Linear probe (Ridge) ──────────────────────────────────────────────────
    print("\nlinear probe (Ridge)...")
    for name, y_tr, y_ev in [
        ("alpha", alpha_tr_n, alpha_ev_n),
        ("zeta",  zeta_tr_n,  zeta_ev_n),
    ]:
        reg = Ridge(alpha=1.0).fit(Z_tr_s, y_tr)
        mse = mean_squared_error(y_ev, reg.predict(Z_ev_s))
        print(f"  linear probe {name}: MSE = {mse:.4f}")
        results[f"linear_{name}_mse"] = mse

    # ── kNN regression ────────────────────────────────────────────────────────
    print("\nkNN regression (k=20)...")
    knn = KNeighborsRegressor(n_neighbors=20, n_jobs=-1)
    for name, y_tr, y_ev in [
        ("alpha", alpha_tr_n, alpha_ev_n),
        ("zeta",  zeta_tr_n,  zeta_ev_n),
    ]:
        knn.fit(Z_tr_s, y_tr)
        mse = mean_squared_error(y_ev, knn.predict(Z_ev_s))
        print(f"  kNN {name}: MSE = {mse:.4f}")
        results[f"knn_{name}_mse"] = mse

    print(f"\n── summary ({split}) ──")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split",      type=str, default="valid", choices=["valid", "test"])
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    evaluate(cfg, args.checkpoint, args.split, args.batch_size)
