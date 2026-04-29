"""
Representation collapse diagnostic for a pretrained JEPA encoder.

Quantifies how much of the feature space is actually used by computing:
  1. Per-channel statistics (stds, fraction of "dead" channels)
  2. Effective rank of the feature matrix (entropy of singular value spectrum)
  3. Participation ratio (how many dimensions carry meaningful energy)
  4. Nearest-neighbor identity rate (do distinct trajectories produce distinct features?)

A healthy representation has:
  - Channel stds roughly uniform and above some floor
  - Effective rank close to the feature dimension (~128)
  - Participation ratio > D/2
  - Low NN-identity rate (different trajectories map to different features)

A collapsed or partially-collapsed representation shows:
  - Channel stds very close to 1.0 everywhere (VICReg hinge saturation) OR near 0
  - Effective rank much smaller than D
  - Participation ratio small
  - High NN-identity rate (many trajectories collapse to near-identical features)

Usage:
    python collapse_check.py \
        --checkpoint /path/to/epoch_N.pt \
        --cache-dir /path/to/cache \
        --output-json /path/to/collapse_epochN.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

_DATASET_DIR = os.environ.get("JEPA_DATASET_DIR")
if _DATASET_DIR:
    sys.path.insert(0, _DATASET_DIR)

from model import ConvEncoder


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_encoder_from_checkpoint(ckpt_path: Path, device: torch.device) -> ConvEncoder:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    m, d = cfg["model"], cfg["dataset"]
    enc = ConvEncoder(
        in_chans=d["num_chans"],
        dims=tuple(m["dims"]),
        num_res_blocks=tuple(m["num_res_blocks"]),
        num_frames=d["num_frames"],
        drop_path_rate=0.0,
    )
    enc.load_state_dict(ckpt["encoder"], strict=False)
    enc.to(device).eval()
    for p in enc.parameters():
        p.requires_grad = False
    return enc


def build_dataset(cache_dir: Path, split: str, num_frames: int):
    import importlib
    mod = importlib.import_module("active_matter_dataset_no_aug")
    Cls = getattr(mod, "ActiveMatterDatasetNoAug")
    return Cls(
        cache_dir=str(cache_dir),
        split=split,
        num_frames=num_frames,
        stride=num_frames,
        augment=False,
        noise_std=0.0,
        return_physical_params=True,
    )


@torch.no_grad()
def extract_features(encoder, dataset, device, batch_size=8, num_workers=4,
                     pool="mean") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    feats, labels, trajs = [], [], []
    n_per_traj = dataset.n_windows_per_traj
    for batch_idx, batch in enumerate(loader):
        ctx = batch["context"].to(device, non_blocking=True)
        with torch.amp.autocast(device.type, enabled=(device.type == "cuda"),
                                dtype=torch.bfloat16):
            e = encoder(ctx)
        if pool == "mean":
            f = e.mean(dim=(-1, -2))
        else:
            f = e.flatten(1)
        feats.append(f.float().cpu().numpy())
        labels.append(batch["physical_params"].numpy())
        start = batch_idx * batch_size
        end = start + ctx.shape[0]
        trajs.append(np.arange(start, end) // n_per_traj)
    return (np.concatenate(feats), np.concatenate(labels),
            np.concatenate(trajs))


def aggregate_per_trajectory(features, labels, traj_idxs):
    unique = np.unique(traj_idxs)
    pf = np.zeros((len(unique), features.shape[1]), dtype=features.dtype)
    pl = np.zeros((len(unique), labels.shape[1]), dtype=labels.dtype)
    for i, t in enumerate(unique):
        mask = traj_idxs == t
        pf[i] = features[mask].mean(axis=0)
        pl[i] = labels[mask][0]
    return pf, pl


# ----------------------------------------------------------------------------
# Collapse metrics
# ----------------------------------------------------------------------------

def channel_stats(features: np.ndarray, eps: float = 1e-4) -> Dict:
    """
    Compute per-channel statistics:
      - stds: per-channel standard deviation (N, D) -> (D,)
      - dead_channels: fraction of channels with std < eps (essentially dead)
      - near_unit_fraction: fraction of channels with std in [0.9, 1.1]
        (VICReg pushes stds to be just above 1.0; if most channels are tightly
        packed at 1.0, the encoder is only barely satisfying the hinge, which
        is characteristic of partial collapse)
    """
    stds = features.std(axis=0)
    return {
        "channel_std_mean": float(stds.mean()),
        "channel_std_median": float(np.median(stds)),
        "channel_std_min": float(stds.min()),
        "channel_std_max": float(stds.max()),
        "channel_std_p10": float(np.percentile(stds, 10)),
        "channel_std_p90": float(np.percentile(stds, 90)),
        "dead_channel_fraction": float((stds < eps).mean()),
        "near_unit_channel_fraction": float(((stds >= 0.9) & (stds <= 1.1)).mean()),
        "total_channels": int(features.shape[1]),
        "_channel_stds": stds.tolist(),
    }


def effective_rank(features: np.ndarray) -> Dict:
    """
    Effective rank via entropy of the normalized singular value spectrum.
    Also compute participation ratio (inverse of sum of squared normalized
    eigenvalues), which is another "how many dimensions matter?" measure.

    For a rank-D matrix with uniform singular values, eff_rank = D.
    For a rank-1 matrix (full collapse), eff_rank = 1.
    """
    # Center the features.
    X = features - features.mean(axis=0, keepdims=True)
    # SVD of (N, D). singular values are shared between X and X.T.
    # For X (N, D) with N >= D, we get D singular values.
    if X.shape[0] < X.shape[1]:
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
    else:
        # Equivalent but faster for tall matrix: eigendecomp of X.T @ X.
        cov = (X.T @ X) / max(1, X.shape[0] - 1)
        eigvals = np.linalg.eigvalsh(cov)[::-1]
        eigvals = np.clip(eigvals, 1e-12, None)
        s = np.sqrt(eigvals * (X.shape[0] - 1))

    s2 = s ** 2
    p = s2 / s2.sum()
    # Shannon entropy over the normalized spectrum, in nats; eff_rank = exp(H).
    entropy = -np.sum(p * np.log(p + 1e-12))
    eff_rank = float(np.exp(entropy))

    # Participation ratio: (sum s^2)^2 / sum s^4.
    part_ratio = float((s2.sum() ** 2) / ((s2 ** 2).sum() + 1e-12))

    # How many singular values carry >=90%, >=99% of variance?
    cumulative = np.cumsum(p)
    rank_90 = int(np.argmax(cumulative >= 0.90) + 1) if (cumulative >= 0.90).any() else len(s)
    rank_99 = int(np.argmax(cumulative >= 0.99) + 1) if (cumulative >= 0.99).any() else len(s)

    return {
        "effective_rank": eff_rank,
        "participation_ratio": part_ratio,
        "rank_90pct_variance": rank_90,
        "rank_99pct_variance": rank_99,
        "max_singular_value": float(s.max()),
        "min_singular_value": float(s.min()),
        "condition_number": float(s.max() / max(s.min(), 1e-12)),
        "_top_singular_values": s[:20].tolist(),
    }


def nearest_neighbor_identity(features: np.ndarray, labels: np.ndarray,
                              k: int = 5) -> Dict:
    """
    For each sample, find its k nearest neighbors in feature space, then
    measure:
      - nn_same_label_rate: fraction of neighbors with the same (alpha, zeta).
        High values would indicate features respect label identity.
      - nn_feature_distance_median: typical distance to nearest neighbor.
        Very small distances across many pairs indicate partial collapse
        (multiple trajectories mapping to near-identical features).
    """
    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(n_neighbors=k + 1)  # +1 for self
    nbrs.fit(features)
    dists, idxs = nbrs.kneighbors(features)
    # Drop the first column (self).
    dists = dists[:, 1:]
    idxs = idxs[:, 1:]

    # Label-identity rate.
    same_label = np.zeros((features.shape[0], k), dtype=bool)
    for i in range(features.shape[0]):
        for j in range(k):
            same_label[i, j] = np.all(labels[i] == labels[idxs[i, j]])

    return {
        "k": k,
        "nn_same_label_rate": float(same_label.mean()),
        "nn_distance_mean": float(dists.mean()),
        "nn_distance_median": float(np.median(dists)),
        "nn_distance_min": float(dists.min()),
        "nn_distance_p10": float(np.percentile(dists, 10)),
    }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--split", default="train",
                        help="Which split to compute stats on. Default train "
                             "for largest sample size.")
    parser.add_argument("--num-frames", default=16, type=int)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--pool", default="mean", choices=("mean", "flatten"))
    args = parser.parse_args()

    device = get_device()
    print(f"[main ] device: {device}")
    print(f"[main ] checkpoint: {args.checkpoint}")

    encoder = build_encoder_from_checkpoint(args.checkpoint, device)
    dataset = build_dataset(args.cache_dir, args.split, args.num_frames)
    print(f"[main ] {args.split} dataset: {len(dataset)} windows")

    t0 = time.time()
    features, labels, trajs = extract_features(
        encoder, dataset, device, batch_size=args.batch_size, pool=args.pool
    )
    print(f"[main ] features: {features.shape} in {time.time()-t0:.1f}s")

    # Aggregate per-trajectory for NN analysis (identity at traj level).
    pf, pl = aggregate_per_trajectory(features, labels, trajs)
    print(f"[main ] aggregated: {pf.shape} (one per trajectory)")

    # Compute metrics.
    print("\n[calc ] channel stats")
    ch = channel_stats(features)

    print("[calc ] effective rank")
    rk = effective_rank(features)

    print("[calc ] nearest-neighbor analysis")
    nn = nearest_neighbor_identity(pf, pl, k=5)

    # ---- Print headline ----
    print("\n" + "=" * 72)
    print("COLLAPSE DIAGNOSTIC")
    print("=" * 72)
    print(f"  Feature dim (total):           {ch['total_channels']}")
    print(f"  Effective rank:                {rk['effective_rank']:.2f}  "
          f"(higher is better; max = {ch['total_channels']})")
    print(f"  Participation ratio:           {rk['participation_ratio']:.2f}")
    print(f"  Rank at 90% variance:          {rk['rank_90pct_variance']}")
    print(f"  Rank at 99% variance:          {rk['rank_99pct_variance']}")
    print()
    print(f"  Channel std mean:              {ch['channel_std_mean']:.4f}")
    print(f"  Channel std [min, median, max]: "
          f"[{ch['channel_std_min']:.4f}, {ch['channel_std_median']:.4f}, "
          f"{ch['channel_std_max']:.4f}]")
    print(f"  Dead channel fraction:         {ch['dead_channel_fraction']:.3f}  "
          f"(std < 1e-4, lower is better)")
    print(f"  Near-unit channel fraction:    {ch['near_unit_channel_fraction']:.3f}  "
          f"(std in [0.9, 1.1]; high = hinge-saturation)")
    print()
    print(f"  NN same-label rate (k=5):      {nn['nn_same_label_rate']:.3f}  "
          f"(1.0 = perfect, 0.022 = chance)")
    print(f"  NN distance [median, min]:     "
          f"[{nn['nn_distance_median']:.4f}, {nn['nn_distance_min']:.4f}]")
    print(f"  Condition number:              {rk['condition_number']:.2e}")
    print("=" * 72)

    # ---- Interpretation ----
    print("\n[interp] reading the numbers:")
    eff = rk["effective_rank"]
    D = ch["total_channels"]
    near_unit = ch["near_unit_channel_fraction"]
    if eff > D * 0.7:
        print(f"  - effective rank = {eff:.1f}/{D} is HEALTHY (>70% of D)")
    elif eff > D * 0.4:
        print(f"  - effective rank = {eff:.1f}/{D} is MODERATE (40-70% of D)")
    else:
        print(f"  - effective rank = {eff:.1f}/{D} is LOW (<40% of D) - possible partial collapse")
    if near_unit > 0.8:
        print(f"  - near_unit_channel_fraction = {near_unit:.2f} is high: "
              f"encoder is hugging the VICReg std hinge, which can indicate partial collapse")
    elif near_unit < 0.3:
        print(f"  - near_unit_channel_fraction = {near_unit:.2f} is low: "
              f"channels vary in scale, encoder is using different magnitudes per channel")

    # ---- Save ----
    results = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "n_windows": int(features.shape[0]),
        "n_trajectories": int(pf.shape[0]),
        "feature_dim": int(features.shape[1]),
        "channel_stats": ch,
        "effective_rank_analysis": rk,
        "nearest_neighbor_analysis": nn,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[main ] saved -> {args.output_json}")


if __name__ == "__main__":
    main()