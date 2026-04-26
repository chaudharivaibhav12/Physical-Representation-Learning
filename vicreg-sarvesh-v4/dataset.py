"""
ActiveMatterDataset (VICReg two-view version)
=============================================
Dataset structure identical to the ViT-JEPA branch:
  - Same HDF5 loading
  - Same sliding window: 32 frames (16 + 16), stride=4
  - Same per-sample z-score normalization
  - Same spatial crop (random 224x224 train, center crop val/test)
  - Same Gaussian noise std

The only difference: instead of returning (context, target) with the same
augmentation applied to the full 32-frame clip, we split into view1 (first 16)
and view2 (last 16) and apply independent augmentations to each.
This gives VICReg two temporally distant views (16-frame gap) with
independent spatial crops and noise — ideal for invariance learning.
"""

import os
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class ActiveMatterDataset(Dataset):
    """
    Two-view VICReg dataset for active matter simulations.

    Returns dicts with:
      "view1": (C=11, T=16, H=224, W=224)  -- first 16 frames, independently augmented
      "view2": (C=11, T=16, H=224, W=224)  -- next  16 frames, independently augmented
      "alpha": float                         -- NOT used during training
      "zeta":  float                         -- NOT used during training

    Args:
        data_dir:   path to active_matter/data/
        split:      "train" | "valid" | "test"
        num_frames: frames per clip half (default 16)
        crop_size:  spatial crop size (default 224)
        stride:     sliding window stride (default 4)
        noise_std:  Gaussian noise std per view (default 1.0, 0 to disable)
        normalize:  per-sample z-score normalization (default True)
    """
    def __init__(
        self,
        data_dir:   str,
        split:      str   = "train",
        num_frames: int   = 16,
        crop_size:  int   = 224,
        stride:     int   = 4,
        noise_std:  float = 1.0,
        normalize:  bool  = True,
    ):
        self.data_dir   = os.path.join(data_dir, split)
        self.num_frames = num_frames
        self.crop_size  = crop_size
        self.stride     = stride
        self.noise_std  = noise_std if split == "train" else 0.0
        self.normalize  = normalize
        self.is_train   = split == "train"

        self.files = sorted(glob.glob(os.path.join(self.data_dir, "*.hdf5")))
        assert len(self.files) > 0, f"No HDF5 files found in {self.data_dir}"
        print(f"[{split}] Found {len(self.files)} files")

        # Build sample index — identical to ViT-JEPA
        # Each sample = (file_path, sim_idx, start_frame)
        # Window = 2 * num_frames = 32 consecutive frames
        self.samples = []
        window = 2 * num_frames  # 32

        for fpath in self.files:
            alpha, zeta = self._parse_params(fpath)
            with h5py.File(fpath, "r") as f:
                num_sims   = f["t0_fields/concentration"].shape[0]   # 3
                num_tsteps = f["t0_fields/concentration"].shape[1]   # 81

            for sim_idx in range(num_sims):
                max_start = num_tsteps - window  # 81 - 32 = 49
                for start in range(0, max_start + 1, stride):
                    self.samples.append((fpath, sim_idx, start, alpha, zeta))

        print(f"[{split}] Total samples: {len(self.samples)}")

    def _parse_params(self, fpath: str):
        """Extract alpha and zeta from filename."""
        name  = os.path.basename(fpath)
        parts = name.replace(".hdf5", "").split("_")
        zeta_idx  = parts.index("zeta")
        alpha_idx = parts.index("alpha")
        zeta  = float(parts[zeta_idx  + 1])
        alpha = float(parts[alpha_idx + 1])
        return alpha, zeta

    def _load_clip(self, fpath: str, sim_idx: int, start: int) -> np.ndarray:
        """
        Load 32 consecutive frames and stack all 11 channels.
        Returns: (32, 11, 256, 256) — identical to ViT-JEPA
        """
        end = start + 2 * self.num_frames  # start + 32

        with h5py.File(fpath, "r") as f:
            conc = f["t0_fields/concentration"][sim_idx, start:end]          # (32, 256, 256)
            conc = conc[:, np.newaxis, :, :]                                  # (32, 1, 256, 256)

            vel  = f["t1_fields/velocity"][sim_idx, start:end]               # (32, 256, 256, 2)
            vel  = vel.transpose(0, 3, 1, 2)                                  # (32, 2, 256, 256)

            D    = f["t2_fields/D"][sim_idx, start:end]                      # (32, 256, 256, 2, 2)
            D    = D.reshape(2 * self.num_frames, 256, 256, 4)
            D    = D.transpose(0, 3, 1, 2)                                    # (32, 4, 256, 256)

            E    = f["t2_fields/E"][sim_idx, start:end]                      # (32, 256, 256, 2, 2)
            E    = E.reshape(2 * self.num_frames, 256, 256, 4)
            E    = E.transpose(0, 3, 1, 2)                                    # (32, 4, 256, 256)

        clip = np.concatenate([conc, vel, D, E], axis=1)                     # (32, 11, 256, 256)
        return clip.astype(np.float32)

    def _random_crop(self, clip: np.ndarray) -> np.ndarray:
        """Random 224x224 spatial crop from 256x256."""
        _, _, H, W = clip.shape
        top  = np.random.randint(0, H - self.crop_size + 1)
        left = np.random.randint(0, W - self.crop_size + 1)
        return clip[:, :, top:top + self.crop_size, left:left + self.crop_size]

    def _center_crop(self, clip: np.ndarray) -> np.ndarray:
        """Center 224x224 crop for val/test."""
        _, _, H, W = clip.shape
        top  = (H - self.crop_size) // 2
        left = (W - self.crop_size) // 2
        return clip[:, :, top:top + self.crop_size, left:left + self.crop_size]

    def _random_flip_rotate(self, clip: np.ndarray) -> np.ndarray:
        """Random horizontal/vertical flip and 90° rotation — valid for isotropic simulations."""
        if np.random.rand() > 0.5:
            clip = clip[:, :, :, ::-1].copy()   # horizontal flip
        if np.random.rand() > 0.5:
            clip = clip[:, :, ::-1, :].copy()   # vertical flip
        k = np.random.randint(0, 4)
        if k > 0:
            clip = np.rot90(clip, k=k, axes=(2, 3)).copy()  # 90/180/270° rotation
        return clip

    def _normalize(self, clip: np.ndarray) -> np.ndarray:
        """Per-sample, per-channel z-score normalization — identical to ViT-JEPA."""
        mean = clip.mean(axis=(0, 2, 3), keepdims=True)   # (1, 11, 1, 1)
        std  = clip.std(axis=(0, 2, 3),  keepdims=True) + 1e-6
        return (clip - mean) / std

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        fpath, sim_idx, start, alpha, zeta = self.samples[idx]

        # Load full 32-frame clip: (32, 11, 256, 256)
        clip = self._load_clip(fpath, sim_idx, start)

        # Normalize full clip (same as ViT-JEPA)
        if self.normalize:
            clip = self._normalize(clip)

        # Split into view1 (first 16) and view2 (last 16) — 16 frame temporal gap
        half1 = clip[:self.num_frames]   # (16, 11, 256, 256)
        half2 = clip[self.num_frames:]   # (16, 11, 256, 256)

        # Apply INDEPENDENT augmentations to each view
        if self.is_train:
            half1 = self._random_crop(half1)
            half2 = self._random_crop(half2)
            half1 = self._random_flip_rotate(half1)
            half2 = self._random_flip_rotate(half2)
            if self.noise_std > 0:
                half1 = half1 + np.random.randn(*half1.shape).astype(np.float32) * self.noise_std
                half2 = half2 + np.random.randn(*half2.shape).astype(np.float32) * self.noise_std
        else:
            half1 = self._center_crop(half1)
            half2 = self._center_crop(half2)

        # (T, C, H, W) → (C, T, H, W) for Conv3D
        view1 = torch.from_numpy(half1).permute(1, 0, 2, 3)  # (11, 16, 224, 224)
        view2 = torch.from_numpy(half2).permute(1, 0, 2, 3)  # (11, 16, 224, 224)

        return {
            "view1": view1,
            "view2": view2,
            "alpha": torch.tensor(alpha, dtype=torch.float32),
            "zeta":  torch.tensor(zeta,  dtype=torch.float32),
        }


# ──────────────────────────────────────────────────────────────────────
# Eval-only single-view dataset (for linear probe / kNN)
# ──────────────────────────────────────────────────────────────────────

class ActiveMatterEval(Dataset):
    """
    Single-view dataset for evaluation.
    Uses the same sliding window as training but returns only view1 (first 16 frames).
    No augmentation — center crop only.
    """
    def __init__(
        self,
        data_dir:   str,
        split:      str  = "valid",
        num_frames: int  = 16,
        crop_size:  int  = 224,
        stride:     int  = 4,
    ):
        self.num_frames = num_frames
        self.crop_size  = crop_size

        split_dir = os.path.join(data_dir, split)
        files     = sorted(glob.glob(os.path.join(split_dir, "*.hdf5")))
        assert len(files) > 0, f"No HDF5 files found in {split_dir}"

        self.samples = []
        window = 2 * num_frames

        for fpath in files:
            alpha, zeta = self._parse_params(fpath)
            with h5py.File(fpath, "r") as f:
                num_sims   = f["t0_fields/concentration"].shape[0]
                num_tsteps = f["t0_fields/concentration"].shape[1]
            for sim_idx in range(num_sims):
                max_start = num_tsteps - window
                for start in range(0, max_start + 1, stride):
                    self.samples.append((fpath, sim_idx, start, alpha, zeta))

        print(f"[eval/{split}] {len(self.samples)} samples")

    def _parse_params(self, fpath):
        name  = os.path.basename(fpath).replace(".hdf5", "")
        parts = name.split("_")
        zeta  = float(parts[parts.index("zeta")  + 1])
        alpha = float(parts[parts.index("alpha") + 1])
        return alpha, zeta

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, sim_idx, start, alpha, zeta = self.samples[idx]

        with h5py.File(fpath, "r") as f:
            end  = start + self.num_frames
            conc = f["t0_fields/concentration"][sim_idx, start:end][:, np.newaxis]
            vel  = f["t1_fields/velocity"][sim_idx, start:end].transpose(0, 3, 1, 2)
            D    = f["t2_fields/D"][sim_idx, start:end].reshape(self.num_frames, 256, 256, 4).transpose(0, 3, 1, 2)
            E    = f["t2_fields/E"][sim_idx, start:end].reshape(self.num_frames, 256, 256, 4).transpose(0, 3, 1, 2)

        clip = np.concatenate([conc, vel, D, E], axis=1).astype(np.float32)  # (16, 11, 256, 256)

        # Normalize + center crop
        mean = clip.mean(axis=(0, 2, 3), keepdims=True)
        std  = clip.std(axis=(0, 2, 3),  keepdims=True) + 1e-6
        clip = (clip - mean) / std

        top  = (256 - self.crop_size) // 2
        left = (256 - self.crop_size) // 2
        clip = clip[:, :, top:top + self.crop_size, left:left + self.crop_size]

        x = torch.from_numpy(clip).permute(1, 0, 2, 3)  # (11, 16, 224, 224)
        return {
            "x":     x,
            "alpha": torch.tensor(alpha, dtype=torch.float32),
            "zeta":  torch.tensor(zeta,  dtype=torch.float32),
        }


# ──────────────────────────────────────────────────────────────────────
# Quick sanity check
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "/scratch/sb10583/data/data"

    ds = ActiveMatterDataset(data_dir, split="train", stride=4)
    sample = ds[0]
    print(f"view1: {sample['view1'].shape}  {sample['view1'].dtype}")
    print(f"view2: {sample['view2'].shape}  {sample['view2'].dtype}")
    print(f"alpha: {sample['alpha'].item():.3f}   zeta: {sample['zeta'].item():.3f}")
    print(f"view1 mean: {sample['view1'].mean():.4f}  std: {sample['view1'].std():.4f}")
    print(f"view2 mean: {sample['view2'].mean():.4f}  std: {sample['view2'].std():.4f}")
