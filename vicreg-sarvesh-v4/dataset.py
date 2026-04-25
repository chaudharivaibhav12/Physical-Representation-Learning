"""
ActiveMatterDataset — VICReg V4
================================
Follows the project spec exactly:
  - 8,750 training samples (window=32, stride=1, max_start=49, 50 windows/sim × 175 sims)
  - Each sample = 32 consecutive frames split into two 16-frame views
  - view1 = frames 0–15, view2 = frames 16–31 (temporally adjacent, different states)

vs V1/V2: same temporal split but stride=1 (4× more data) and stronger augmentations.
vs V3:    back to temporal views (avoids same-clip collapse).

Independent spatial augmentations applied to each view (train only):
  - Random 224x224 spatial crop
  - Random horizontal / vertical flip
  - Random 90° rotation
  - Gaussian noise (std=1.0)
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
      "view1": (C=11, T=16, H=224, W=224)  -- frames 0-15, augmented
      "view2": (C=11, T=16, H=224, W=224)  -- frames 16-31, augmented
      "alpha": float
      "zeta":  float
    """
    def __init__(
        self,
        data_dir:   str,
        split:      str   = "train",
        num_frames: int   = 16,
        crop_size:  int   = 224,
        noise_std:  float = 1.0,
        normalize:  bool  = True,
    ):
        self.data_dir   = os.path.join(data_dir, split)
        self.num_frames = num_frames
        self.crop_size  = crop_size
        self.noise_std  = noise_std if split == "train" else 0.0
        self.normalize  = normalize
        self.is_train   = split == "train"

        self.files = sorted(glob.glob(os.path.join(self.data_dir, "*.hdf5")))
        assert len(self.files) > 0, f"No HDF5 files found in {self.data_dir}"
        print(f"[{split}] Found {len(self.files)} files")

        # window=32, stride=1 → max_start=49 → 50 windows/sim → 8,750 train samples
        window = 2 * num_frames  # 32
        self.samples = []
        for fpath in self.files:
            alpha, zeta = self._parse_params(fpath)
            with h5py.File(fpath, "r") as f:
                num_sims   = f["t0_fields/concentration"].shape[0]
                num_tsteps = f["t0_fields/concentration"].shape[1]

            for sim_idx in range(num_sims):
                max_start = num_tsteps - window  # 81 - 32 = 49
                for start in range(0, max_start + 1):  # stride=1, 50 windows
                    self.samples.append((fpath, sim_idx, start, alpha, zeta))

        print(f"[{split}] Total samples: {len(self.samples)}")

    def _parse_params(self, fpath: str):
        name  = os.path.basename(fpath)
        parts = name.replace(".hdf5", "").split("_")
        zeta  = float(parts[parts.index("zeta")  + 1])
        alpha = float(parts[parts.index("alpha") + 1])
        return alpha, zeta

    def _load_clip(self, fpath: str, sim_idx: int, start: int) -> np.ndarray:
        """Load 32 consecutive frames, all 11 channels. Returns (32, 11, 256, 256)."""
        end = start + 2 * self.num_frames

        with h5py.File(fpath, "r") as f:
            conc = f["t0_fields/concentration"][sim_idx, start:end][:, np.newaxis]
            vel  = f["t1_fields/velocity"][sim_idx, start:end].transpose(0, 3, 1, 2)
            D    = f["t2_fields/D"][sim_idx, start:end].reshape(2 * self.num_frames, 256, 256, 4).transpose(0, 3, 1, 2)
            E    = f["t2_fields/E"][sim_idx, start:end].reshape(2 * self.num_frames, 256, 256, 4).transpose(0, 3, 1, 2)

        return np.concatenate([conc, vel, D, E], axis=1).astype(np.float32)  # (32, 11, 256, 256)

    def _normalize(self, clip: np.ndarray) -> np.ndarray:
        mean = clip.mean(axis=(0, 2, 3), keepdims=True)
        std  = clip.std(axis=(0, 2, 3),  keepdims=True) + 1e-6
        return (clip - mean) / std

    def _augment(self, clip: np.ndarray) -> np.ndarray:
        """Random crop → flip → rotate → noise."""
        _, _, H, W = clip.shape
        top  = np.random.randint(0, H - self.crop_size + 1)
        left = np.random.randint(0, W - self.crop_size + 1)
        clip = clip[:, :, top:top + self.crop_size, left:left + self.crop_size]
        if np.random.rand() > 0.5:
            clip = clip[:, :, :, ::-1].copy()
        if np.random.rand() > 0.5:
            clip = clip[:, :, ::-1, :].copy()
        k = np.random.randint(0, 4)
        if k > 0:
            clip = np.rot90(clip, k=k, axes=(2, 3)).copy()
        if self.noise_std > 0:
            clip = clip + np.random.randn(*clip.shape).astype(np.float32) * self.noise_std
        return clip

    def _center_crop(self, clip: np.ndarray) -> np.ndarray:
        _, _, H, W = clip.shape
        top  = (H - self.crop_size) // 2
        left = (W - self.crop_size) // 2
        return clip[:, :, top:top + self.crop_size, left:left + self.crop_size]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        fpath, sim_idx, start, alpha, zeta = self.samples[idx]

        clip = self._load_clip(fpath, sim_idx, start)  # (32, 11, 256, 256)

        if self.normalize:
            clip = self._normalize(clip)

        half1 = clip[:self.num_frames]   # (16, 11, 256, 256) — frames 0-15
        half2 = clip[self.num_frames:]   # (16, 11, 256, 256) — frames 16-31

        if self.is_train:
            half1 = self._augment(half1)
            half2 = self._augment(half2)
        else:
            half1 = self._center_crop(half1)
            half2 = self._center_crop(half2)

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
    """Single-view eval dataset. First 16 frames, center crop, no augmentation, stride=1."""
    def __init__(
        self,
        data_dir:   str,
        split:      str  = "valid",
        num_frames: int  = 16,
        crop_size:  int  = 224,
    ):
        self.num_frames = num_frames
        self.crop_size  = crop_size

        split_dir = os.path.join(data_dir, split)
        files     = sorted(glob.glob(os.path.join(split_dir, "*.hdf5")))
        assert len(files) > 0, f"No HDF5 files found in {split_dir}"

        self.samples = []
        for fpath in files:
            alpha, zeta = self._parse_params(fpath)
            with h5py.File(fpath, "r") as f:
                num_sims   = f["t0_fields/concentration"].shape[0]
                num_tsteps = f["t0_fields/concentration"].shape[1]
            for sim_idx in range(num_sims):
                max_start = num_tsteps - num_frames
                for start in range(0, max_start + 1):  # stride=1
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

        clip = np.concatenate([conc, vel, D, E], axis=1).astype(np.float32)
        mean = clip.mean(axis=(0, 2, 3), keepdims=True)
        std  = clip.std(axis=(0, 2, 3),  keepdims=True) + 1e-6
        clip = (clip - mean) / std
        top  = (256 - self.crop_size) // 2
        left = (256 - self.crop_size) // 2
        clip = clip[:, :, top:top + self.crop_size, left:left + self.crop_size]

        x = torch.from_numpy(clip).permute(1, 0, 2, 3)
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

    ds = ActiveMatterDataset(data_dir, split="train")
    print(f"Train samples: {len(ds)}  (expected ~8750)")
    sample = ds[0]
    print(f"view1: {sample['view1'].shape}  view2: {sample['view2'].shape}")
    print(f"alpha: {sample['alpha'].item():.3f}   zeta: {sample['zeta'].item():.3f}")
    print(f"view1 mean: {sample['view1'].mean():.4f}  std: {sample['view1'].std():.4f}")
    print(f"view2 mean: {sample['view2'].mean():.4f}  std: {sample['view2'].std():.4f}")
    print(f"views identical: {(sample['view1'] == sample['view2']).all().item()} (should be False)")
