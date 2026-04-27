"""
VideoMAE Evaluation — Linear Probe + kNN Regression
=====================================================
Loads the frozen VideoMAE encoder from a checkpoint, extracts per-sample
embeddings, and evaluates:
  1. Linear probe  — Ridge regression (sklearn)
  2. kNN           — k-Nearest Neighbours regression (sklearn)

Targets: alpha and zeta (z-score normalized before regression).
Metric:  MSE on val and test sets (reported in original space as well).

Usage:
  python evaluate.py --checkpoint /scratch/sb10583/checkpoints/videomae-v1/best.pt
  python evaluate.py --checkpoint best.pt --data-dir /scratch/sb10583/data/data
  python evaluate.py --checkpoint best.pt --knn-k 20 --stride 2
"""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from sklearn.linear_model import Ridge
from sklearn.neighbors    import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline      import Pipeline

from model   import VideoMAE
from dataset import VideoMAEEval


# ─────────────────────────────────────────────
# Embedding extraction
# ─────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(model, loader, device, amp_dtype):
    """
    Returns:
      embeddings: (N, D) numpy array
      alphas:     (N,)   numpy array
      zetas:      (N,)   numpy array
    """
    model.eval()
    all_emb, all_alpha, all_zeta = [], [], []

    for batch in loader:
        x     = batch["frames"].to(device, non_blocking=True)
        alpha = batch["alpha"]
        zeta  = batch["zeta"]

        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            emb = model.encode(x)   # (B, D)

        all_emb.append(emb.cpu().float().numpy())
        all_alpha.append(alpha.numpy())
        all_zeta.append(zeta.numpy())

    return (
        np.concatenate(all_emb,   axis=0),
        np.concatenate(all_alpha, axis=0),
        np.concatenate(all_zeta,  axis=0),
    )


# ─────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────

def evaluate_regression(
    X_train, y_train,
    X_eval,  y_eval,
    label: str,
    knn_k: int = 20,
):
    """
    Runs linear probe + kNN on a single target variable.
    Reports MSE in the normalized space and the original space.
    """
    # Z-score normalize targets
    y_scaler = StandardScaler()
    y_train_n = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

    # ── Linear probe (Ridge) ───────────────────────────────────────────
    linear = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge",  Ridge(alpha=1.0)),
    ])
    linear.fit(X_train, y_train_n)
    pred_lin_n = linear.predict(X_eval)
    pred_lin   = y_scaler.inverse_transform(pred_lin_n.reshape(-1, 1)).ravel()

    mse_lin_norm = np.mean((pred_lin_n - y_scaler.transform(y_eval.reshape(-1, 1)).ravel()) ** 2)
    mse_lin      = np.mean((pred_lin   - y_eval) ** 2)

    # ── kNN regression ────────────────────────────────────────────────
    knn = Pipeline([
        ("scaler", StandardScaler()),
        ("knn",    KNeighborsRegressor(n_neighbors=knn_k, metric="euclidean", weights="uniform")),
    ])
    knn.fit(X_train, y_train_n)
    pred_knn_n = knn.predict(X_eval)
    pred_knn   = y_scaler.inverse_transform(pred_knn_n.reshape(-1, 1)).ravel()

    mse_knn_norm = np.mean((pred_knn_n - y_scaler.transform(y_eval.reshape(-1, 1)).ravel()) ** 2)
    mse_knn      = np.mean((pred_knn   - y_eval) ** 2)

    print(f"\n  [{label}]")
    print(f"    Linear probe  — MSE (norm): {mse_lin_norm:.4f}   MSE (orig): {mse_lin:.4f}")
    print(f"    kNN (k={knn_k:2d})      — MSE (norm): {mse_knn_norm:.4f}   MSE (orig): {mse_knn:.4f}")

    return {
        f"{label}/linear_mse_norm": mse_lin_norm,
        f"{label}/linear_mse":      mse_lin,
        f"{label}/knn_mse_norm":    mse_knn_norm,
        f"{label}/knn_mse":         mse_knn,
    }


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main(args):
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16

    # ── Load model ────────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg  = ckpt.get("config", {})

    model = VideoMAE(
        in_channels=cfg.get("in_channels",    11),
        num_frames=cfg.get("num_frames",      16),
        img_size=cfg.get("crop_size",         224),
        enc_embed_dim=cfg.get("enc_embed_dim", 192),
        enc_depth=cfg.get("enc_depth",         12),
        enc_heads=cfg.get("enc_heads",          3),
        mlp_ratio=cfg.get("mlp_ratio",         4.0),
        dropout=cfg.get("dropout",             0.0),
        patch_size=cfg.get("patch_size",       16),
        tubelet=cfg.get("tubelet",              2),
        mask_ratio=cfg.get("mask_ratio",       0.90),
        dec_embed_dim=cfg.get("dec_embed_dim", 96),
        dec_depth=cfg.get("dec_depth",          4),
        dec_heads=cfg.get("dec_heads",          3),
        norm_pix_loss=cfg.get("norm_pix_loss", True),
    ).to(device)

    # Load encoder weights only (decoder is discarded)
    enc_state = ckpt.get("encoder", None)
    if enc_state is None:
        # Fall back to full model state dict
        full_state = ckpt["model"]
        if all(k.startswith("module.") for k in full_state):
            full_state = {k[len("module."):]: v for k, v in full_state.items()}
        model.load_state_dict(full_state)
    else:
        model.encoder.load_state_dict(enc_state)

    model.eval()
    print(f"Encoder parameters: {sum(p.numel() for p in model.encoder.parameters()):,}")

    # ── Build eval datasets ────────────────────────────────────────────
    data_dir  = args.data_dir or cfg.get("data_dir", "/scratch/sb10583/data/data")
    num_frames= cfg.get("num_frames", 16)
    crop_size = cfg.get("crop_size",  224)
    batch_size= args.batch_size

    def make_loader(split):
        ds = VideoMAEEval(
            data_dir=data_dir, split=split,
            num_frames=num_frames, crop_size=crop_size, stride=args.stride,
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print("\nExtracting train embeddings...")
    X_train, alpha_train, zeta_train = extract_embeddings(model, make_loader("train"), device, amp_dtype)
    print(f"  X_train: {X_train.shape}")

    print("Extracting val embeddings...")
    X_val,   alpha_val,   zeta_val   = extract_embeddings(model, make_loader("valid"), device, amp_dtype)
    print(f"  X_val:   {X_val.shape}")

    print("Extracting test embeddings...")
    X_test,  alpha_test,  zeta_test  = extract_embeddings(model, make_loader("test"),  device, amp_dtype)
    print(f"  X_test:  {X_test.shape}")

    # ── Evaluate on val ───────────────────────────────────────────────
    print("\n═══ Validation ══════════════════════════════")
    val_results = {}
    val_results.update(evaluate_regression(X_train, alpha_train, X_val, alpha_val,   "val/alpha", args.knn_k))
    val_results.update(evaluate_regression(X_train, zeta_train,  X_val, zeta_val,    "val/zeta",  args.knn_k))

    # ── Evaluate on test ──────────────────────────────────────────────
    print("\n═══ Test ════════════════════════════════════")
    test_results = {}
    test_results.update(evaluate_regression(X_train, alpha_train, X_test, alpha_test, "test/alpha", args.knn_k))
    test_results.update(evaluate_regression(X_train, zeta_train,  X_test, zeta_test,  "test/zeta",  args.knn_k))

    # ── Summary ───────────────────────────────────────────────────────
    print("\n═══ Summary ═════════════════════════════════")
    all_results = {**val_results, **test_results}
    for k, v in sorted(all_results.items()):
        print(f"  {k:45s} = {v:.6f}")

    # Optionally save results
    if args.out:
        np.save(args.out, all_results)
        print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,  required=True, help="path to .pt checkpoint")
    parser.add_argument("--data-dir",   type=str,  default=None,  help="override data_dir from config")
    parser.add_argument("--batch-size", type=int,  default=32,    help="embedding extraction batch size")
    parser.add_argument("--stride",     type=int,  default=2,     help="eval dataset stride")
    parser.add_argument("--knn-k",      type=int,  default=20,    help="k for kNN regression")
    parser.add_argument("--out",        type=str,  default=None,  help="save results dict as .npy")
    args = parser.parse_args()
    main(args)
