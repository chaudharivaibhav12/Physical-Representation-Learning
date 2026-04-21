"""
ActiveMatterDataset
===================
Loads the active_matter HDF5 files and returns (context, target) pairs
for ViT-JEPA training.

Each HDF5 file contains:
  - 3 independent simulations
  - 81 time steps each
  - 256x256 spatial resolution
  - 11 physical channels (concentration, velocity, D tensor, E tensor)

We extract samples using a sliding window of 32 frames (16 context + 16 target)
and apply a random 224x224 spatial crop.
"""

import os
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class ActiveMatterDataset(Dataset):
    """
    Dataset for active matter physical simulations.

    Returns dicts with:
      "context": (C=11, T=16, H=224, W=224)  -- first 16 frames
      "target":  (C=11, T=16, H=224, W=224)  -- next  16 frames
      "alpha":   float                         -- physical parameter (NOT used in training)
      "zeta":    float                         -- physical parameter (NOT used in training)

    Args:
        data_dir:   path to active_matter/data/
        split:      "train" | "valid" | "test"
        num_frames: frames per clip half (default 16)
        crop_size:  spatial crop size (default 224)
        stride:     sliding window stride (default 4)
        noise_std:  Gaussian noise augmentation std (default 1.0, 0 to disable)
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

        # Gather all HDF5 files
        self.files = sorted(glob.glob(os.path.join(self.data_dir, "*.hdf5")))
        assert len(self.files) > 0, f"No HDF5 files found in {self.data_dir}"
        print(f"[{split}] Found {len(self.files)} files")

        # Build sample index
        # Each sample = (file_path, sim_idx, start_frame)
        # We need 2*num_frames = 32 consecutive frames per sample
        self.samples = []
        window = 2 * num_frames  # 32 frames total (16 context + 16 target)

        for fpath in self.files:
            # Parse alpha and zeta from filename
            # e.g. active_matter_L_10.0_zeta_1.0_alpha_-3.0.hdf5
            alpha, zeta = self._parse_params(fpath)

            with h5py.File(fpath, "r") as f:
                num_sims   = f["t0_fields/concentration"].shape[0]  # 3
                num_tsteps = f["t0_fields/concentration"].shape[1]  # 81

            for sim_idx in range(num_sims):
                # Slide window across time dimension
                max_start = num_tsteps - window  # 81 - 32 = 49
                for start in range(0, max_start + 1, stride):
                    self.samples.append((fpath, sim_idx, start, alpha, zeta))

        print(f"[{split}] Total samples: {len(self.samples)}")

    def _parse_params(self, fpath: str):
        """Extract alpha and zeta from filename."""
        name  = os.path.basename(fpath)
        parts = name.replace(".hdf5", "").split("_")
        # Format: active_matter_L_10.0_zeta_1.0_alpha_-3.0
        zeta_idx  = parts.index("zeta")
        alpha_idx = parts.index("alpha")
        zeta  = float(parts[zeta_idx + 1])
        alpha = float(parts[alpha_idx + 1])
        return alpha, zeta

    def _load_clip(self, fpath: str, sim_idx: int, start: int) -> np.ndarray:
        """
        Load 32 consecutive frames and stack all 11 channels.

        Returns: (32, 11, 256, 256)
        """
        end = start + 2 * self.num_frames  # start + 32

        with h5py.File(fpath, "r") as f:
            # Concentration: (3, 81, 256, 256) -> (32, 256, 256)
            conc = f["t0_fields/concentration"][sim_idx, start:end]   # (32, 256, 256)
            conc = conc[:, np.newaxis, :, :]                           # (32, 1, 256, 256)

            # Velocity: (3, 81, 256, 256, 2) -> (32, 2, 256, 256)
            vel  = f["t1_fields/velocity"][sim_idx, start:end]         # (32, 256, 256, 2)
            vel  = vel.transpose(0, 3, 1, 2)                           # (32, 2, 256, 256)

            # D tensor: (3, 81, 256, 256, 2, 2) -> (32, 4, 256, 256)
            D    = f["t2_fields/D"][sim_idx, start:end]                # (32, 256, 256, 2, 2)
            D    = D.reshape(2 * self.num_frames, 256, 256, 4)
            D    = D.transpose(0, 3, 1, 2)                             # (32, 4, 256, 256)

            # E tensor: same as D
            E    = f["t2_fields/E"][sim_idx, start:end]                # (32, 256, 256, 2, 2)
            E    = E.reshape(2 * self.num_frames, 256, 256, 4)
            E    = E.transpose(0, 3, 1, 2)                             # (32, 4, 256, 256)

        # Stack all channels: (32, 11, 256, 256)
        clip = np.concatenate([conc, vel, D, E], axis=1)
        return clip.astype(np.float32)

    def _random_crop(self, clip: np.ndarray) -> np.ndarray:
        """Random 224x224 spatial crop from 256x256."""
        _, _, H, W = clip.shape
        top  = np.random.randint(0, H - self.crop_size + 1)
        left = np.random.randint(0, W - self.crop_size + 1)
        return clip[:, :, top:top + self.crop_size, left:left + self.crop_size]

    def _center_crop(self, clip: np.ndarray) -> np.ndarray:
        """Center 224x224 crop for validation/test."""
        _, _, H, W = clip.shape
        top  = (H - self.crop_size) // 2
        left = (W - self.crop_size) // 2
        return clip[:, :, top:top + self.crop_size, left:left + self.crop_size]

    def _normalize(self, clip: np.ndarray) -> np.ndarray:
        """
        Per-sample, per-channel z-score normalization.
        Normalizes each channel independently across T, H, W.
        """
        # clip: (T, C, H, W)
        mean = clip.mean(axis=(0, 2, 3), keepdims=True)   # (1, C, 1, 1)
        std  = clip.std(axis=(0, 2, 3),  keepdims=True) + 1e-6
        return (clip - mean) / std

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        fpath, sim_idx, start, alpha, zeta = self.samples[idx]

        # Load 32-frame clip: (32, 11, 256, 256)
        clip = self._load_clip(fpath, sim_idx, start)

        # Spatial crop
        if self.is_train:
            clip = self._random_crop(clip)   # (32, 11, 224, 224)
        else:
            clip = self._center_crop(clip)   # (32, 11, 224, 224)

        # Normalize
        if self.normalize:
            clip = self._normalize(clip)

        # Add Gaussian noise augmentation (training only)
        if self.noise_std > 0:
            clip = clip + np.random.randn(*clip.shape).astype(np.float32) * self.noise_std

        # Split into context (first 16) and target (last 16)
        context = clip[:self.num_frames]    # (16, 11, 224, 224)
        target  = clip[self.num_frames:]    # (16, 11, 224, 224)

        # Convert to tensor and reorder to (C, T, H, W) for Conv3D
        context = torch.from_numpy(context).permute(1, 0, 2, 3)  # (11, 16, 224, 224)
        target  = torch.from_numpy(target).permute(1, 0, 2, 3)   # (11, 16, 224, 224)

        return {
            "context": context,
            "target":  target,
            "alpha":   torch.tensor(alpha, dtype=torch.float32),
            "zeta":    torch.tensor(zeta,  dtype=torch.float32),
        }


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    data_dir = "/scratch/ok2287/data/active_matter/data"
    dataset  = ActiveMatterDataset(data_dir, split="train", stride=4)

    sample = dataset[0]
    print(f"context shape: {sample['context'].shape}")  # (11, 16, 224, 224)
    print(f"target  shape: {sample['target'].shape}")   # (11, 16, 224, 224)
    print(f"alpha:         {sample['alpha'].item()}")
    print(f"zeta:          {sample['zeta'].item()}")
    print(f"context mean:  {sample['context'].mean():.4f}")
    print(f"context std:   {sample['context'].std():.4f}")
