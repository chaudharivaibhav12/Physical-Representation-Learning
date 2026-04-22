"""
ActiveMatterDataset (VICReg two-view version)
=============================================
Returns two independently augmented views of the same simulation clip
for VICReg self-supervised training.

Each HDF5 file contains:
  - 3 independent simulations
  - 81 time steps each
  - 256x256 spatial resolution
  - 11 physical channels (concentration, velocity, D tensor, E tensor)

Two views are sampled from temporally distant windows (min gap = 16 frames)
so the encoder must learn invariant physics features, not just temporal proximity.

Returns:
  "view1": (C=11, T=16, H=224, W=224)
  "view2": (C=11, T=16, H=224, W=224)
  "alpha": float  -- NOT used during training
  "zeta":  float  -- NOT used during training
"""

import os
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


NUM_FRAMES_TOTAL = 81
NUM_FRAMES_CLIP  = 16
RAW_SIZE         = 256
CROP_SIZE        = 224
MIN_TEMPORAL_GAP = 16   # minimum frame gap between view1 and view2 start


class ActiveMatterVICReg(Dataset):
    """
    Two-view dataset for VICReg training on active_matter simulations.

    Args:
        data_dir:   path to active_matter/data/
        split:      "train" | "valid" | "test"
        crop_size:  spatial crop (default 224)
        noise_std:  Gaussian noise std applied per-view (default 0.05, 0 = off)
        min_gap:    minimum temporal gap between view1 and view2 start frames
    """
    def __init__(
        self,
        data_dir:   str,
        split:      str   = "train",
        crop_size:  int   = CROP_SIZE,
        noise_std:  float = 0.05,
        min_gap:    int   = MIN_TEMPORAL_GAP,
        stride:     int   = 4,
    ):
        self.split     = split
        self.crop_size = crop_size
        self.noise_std = noise_std if split == "train" else 0.0
        self.min_gap   = min_gap
        self.is_train  = (split == "train")

        split_dir  = os.path.join(data_dir, split)
        self.files = sorted(glob.glob(os.path.join(split_dir, "*.hdf5")))
        assert len(self.files) > 0, f"No HDF5 files found in {split_dir}"

        # Index: sliding window over time — (file, sim_idx, anchor_start, alpha, zeta)
        # anchor_start is the base position for view1; view2 is sampled far from it
        self.samples = []
        max_start = NUM_FRAMES_TOTAL - NUM_FRAMES_CLIP  # 65
        for fpath in self.files:
            alpha, zeta = self._parse_params(fpath)
            with h5py.File(fpath, "r") as f:
                n_sims = f["t0_fields/concentration"].shape[0]
            for sim_idx in range(n_sims):
                for start in range(0, max_start + 1, stride):
                    self.samples.append((fpath, sim_idx, start, alpha, zeta))

        print(f"[{split}] {len(self.files)} files → {len(self.samples)} samples")

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────

    def _parse_params(self, fpath: str):
        name  = os.path.basename(fpath).replace(".hdf5", "")
        parts = name.split("_")
        zeta  = float(parts[parts.index("zeta")  + 1])
        alpha = float(parts[parts.index("alpha") + 1])
        return alpha, zeta

    def _load_clip(self, fpath: str, sim_idx: int, start: int) -> np.ndarray:
        """Load 16 consecutive frames → (16, 11, 256, 256) float32."""
        end = start + NUM_FRAMES_CLIP
        with h5py.File(fpath, "r") as f:
            conc = f["t0_fields/concentration"][sim_idx, start:end]          # (16, 256, 256)
            conc = conc[:, np.newaxis, :, :]                                  # (16, 1, 256, 256)

            vel  = f["t1_fields/velocity"][sim_idx, start:end]               # (16, 256, 256, 2)
            vel  = vel.transpose(0, 3, 1, 2)                                  # (16, 2, 256, 256)

            D    = f["t2_fields/D"][sim_idx, start:end]                      # (16, 256, 256, 2, 2)
            D    = D.reshape(NUM_FRAMES_CLIP, RAW_SIZE, RAW_SIZE, 4)
            D    = D.transpose(0, 3, 1, 2)                                    # (16, 4, 256, 256)

            E    = f["t2_fields/E"][sim_idx, start:end]                      # (16, 256, 256, 2, 2)
            E    = E.reshape(NUM_FRAMES_CLIP, RAW_SIZE, RAW_SIZE, 4)
            E    = E.transpose(0, 3, 1, 2)                                    # (16, 4, 256, 256)

        clip = np.concatenate([conc, vel, D, E], axis=1)                     # (16, 11, 256, 256)
        return clip.astype(np.float32)

    def _sample_two_starts(self) -> tuple:
        """
        Sample two start frames with minimum temporal gap.
        Both windows must fit within [0, NUM_FRAMES_TOTAL - NUM_FRAMES_CLIP].
        """
        max_start = NUM_FRAMES_TOTAL - NUM_FRAMES_CLIP  # 65

        for _ in range(50):  # retry until valid pair found
            s1 = np.random.randint(0, max_start + 1)
            s2 = np.random.randint(0, max_start + 1)
            if abs(s1 - s2) >= self.min_gap:
                return s1, s2

        # Fallback: deterministic distant pair
        return 0, max_start

    def _random_crop(self, clip: np.ndarray) -> np.ndarray:
        """Random 224x224 crop from 256x256."""
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

    def _normalize(self, clip: np.ndarray) -> np.ndarray:
        """Per-sample, per-channel z-score normalization."""
        mean = clip.mean(axis=(0, 2, 3), keepdims=True)   # (1, 11, 1, 1)
        std  = clip.std(axis=(0, 2, 3),  keepdims=True) + 1e-6
        return (clip - mean) / std

    def _augment(self, clip: np.ndarray) -> np.ndarray:
        """Independent augmentation per view (training only)."""
        # Random spatial crop
        clip = self._random_crop(clip)

        # Gaussian noise
        if self.noise_std > 0:
            clip = clip + np.random.randn(*clip.shape).astype(np.float32) * self.noise_std

        return clip

    # ──────────────────────────────────────────────────────────────────
    # Dataset interface
    # ──────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        fpath, sim_idx, anchor, alpha, zeta = self.samples[idx]

        if self.is_train:
            # view1 anchored at sliding window position, view2 sampled far away
            s1 = anchor
            max_start = NUM_FRAMES_TOTAL - NUM_FRAMES_CLIP
            for _ in range(50):
                s2 = np.random.randint(0, max_start + 1)
                if abs(s2 - s1) >= self.min_gap:
                    break
            else:
                s2 = max_start - s1 if s1 < max_start // 2 else 0
        else:
            # Val/test: deterministic — anchor for view1, midpoint for view2
            s1 = anchor
            s2 = (NUM_FRAMES_TOTAL - NUM_FRAMES_CLIP) // 2  # 32

        clip1 = self._load_clip(fpath, sim_idx, s1)   # (16, 11, 256, 256)
        clip2 = self._load_clip(fpath, sim_idx, s2)   # (16, 11, 256, 256)

        # Normalize each clip independently
        clip1 = self._normalize(clip1)
        clip2 = self._normalize(clip2)

        # Augment (independent per view, training only)
        if self.is_train:
            clip1 = self._augment(clip1)
            clip2 = self._augment(clip2)
        else:
            clip1 = self._center_crop(clip1)
            clip2 = self._center_crop(clip2)

        # (T, C, H, W) → (C, T, H, W) for Conv3D
        view1 = torch.from_numpy(clip1).permute(1, 0, 2, 3)  # (11, 16, 224, 224)
        view2 = torch.from_numpy(clip2).permute(1, 0, 2, 3)  # (11, 16, 224, 224)

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
    Single-view dataset for evaluation. Returns one deterministic clip
    per simulation (frames 32:48, center crop). No augmentation.
    """
    def __init__(self, data_dir: str, split: str = "valid", crop_size: int = CROP_SIZE):
        self.crop_size = crop_size

        split_dir  = os.path.join(data_dir, split)
        files      = sorted(glob.glob(os.path.join(split_dir, "*.hdf5")))
        assert len(files) > 0, f"No HDF5 files found in {split_dir}"

        self.sims = []
        for fpath in files:
            alpha, zeta = self._parse_params(fpath)
            with h5py.File(fpath, "r") as f:
                n_sims = f["t0_fields/concentration"].shape[0]
            for sim_idx in range(n_sims):
                self.sims.append((fpath, sim_idx, alpha, zeta))

        print(f"[eval/{split}] {len(self.sims)} simulations")

    def _parse_params(self, fpath):
        name  = os.path.basename(fpath).replace(".hdf5", "")
        parts = name.split("_")
        zeta  = float(parts[parts.index("zeta")  + 1])
        alpha = float(parts[parts.index("alpha") + 1])
        return alpha, zeta

    def __len__(self):
        return len(self.sims)

    def __getitem__(self, idx):
        fpath, sim_idx, alpha, zeta = self.sims[idx]
        start = (NUM_FRAMES_TOTAL - NUM_FRAMES_CLIP) // 2  # frame 32

        with h5py.File(fpath, "r") as f:
            end  = start + NUM_FRAMES_CLIP
            conc = f["t0_fields/concentration"][sim_idx, start:end][:, np.newaxis]
            vel  = f["t1_fields/velocity"][sim_idx, start:end].transpose(0, 3, 1, 2)
            D    = f["t2_fields/D"][sim_idx, start:end].reshape(NUM_FRAMES_CLIP, RAW_SIZE, RAW_SIZE, 4).transpose(0, 3, 1, 2)
            E    = f["t2_fields/E"][sim_idx, start:end].reshape(NUM_FRAMES_CLIP, RAW_SIZE, RAW_SIZE, 4).transpose(0, 3, 1, 2)

        clip = np.concatenate([conc, vel, D, E], axis=1).astype(np.float32)  # (16, 11, 256, 256)

        # Normalize + center crop
        mean = clip.mean(axis=(0, 2, 3), keepdims=True)
        std  = clip.std(axis=(0, 2, 3),  keepdims=True) + 1e-6
        clip = (clip - mean) / std

        top  = (RAW_SIZE - self.crop_size) // 2
        left = (RAW_SIZE - self.crop_size) // 2
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

    ds = ActiveMatterVICReg(data_dir, split="train")
    sample = ds[0]
    print(f"view1: {sample['view1'].shape}  {sample['view1'].dtype}")
    print(f"view2: {sample['view2'].shape}  {sample['view2'].dtype}")
    print(f"alpha: {sample['alpha'].item():.3f}   zeta: {sample['zeta'].item():.3f}")
    print(f"view1 mean: {sample['view1'].mean():.4f}  std: {sample['view1'].std():.4f}")
    print(f"view2 mean: {sample['view2'].mean():.4f}  std: {sample['view2'].std():.4f}")
