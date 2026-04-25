"""
ActiveMatterDataset v2
======================
Changes from v1:
  - Physics-aware spatial flip augmentations (horizontal + vertical)
    Velocity components and tensor off-diagonals are sign-corrected
    to preserve physical consistency under coordinate reflection.

Each HDF5 file contains:
  - 3 independent simulations
  - 81 time steps each
  - 256x256 spatial resolution
  - 11 physical channels (concentration, velocity, D tensor, E tensor)

Channel layout after stacking (indices 0–10):
  0        : concentration (scalar)
  1        : vx  (velocity x-component)
  2        : vy  (velocity y-component)
  3        : Dxx (orientation tensor, row-major flatten of 2x2)
  4        : Dxy
  5        : Dyx
  6        : Dyy
  7        : Exx (strain-rate tensor)
  8        : Exy
  9        : Eyx
  10       : Eyy

Flip sign conventions:
  Horizontal flip (x → -x, flip W axis):
    vx → -vx  (channel 1)
    Dxy, Dyx, Exy, Eyx → negate  (channels 4, 5, 8, 9)
  Vertical flip (y → -y, flip H axis):
    vy → -vy  (channel 2)
    Dxy, Dyx, Exy, Eyx → negate  (channels 4, 5, 8, 9)
"""

import os
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


# Sign masks applied after flipping — shape (11,) broadcast over (T, 11, H, W)
# 1 = no sign change, -1 = negate
_HFLIP_SIGN = np.array([1, -1,  1, 1, -1, -1, 1, 1, -1, -1, 1], dtype=np.float32)
_VFLIP_SIGN = np.array([1,  1, -1, 1, -1, -1, 1, 1, -1, -1, 1], dtype=np.float32)


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
        stride:     sliding window stride (default 1)
        noise_std:  Gaussian noise augmentation std (default 1.0, 0 to disable)
        normalize:  per-sample z-score normalization (default True)
        hflip_prob: probability of horizontal flip augmentation (train only)
        vflip_prob: probability of vertical flip augmentation (train only)
    """
    def __init__(
        self,
        data_dir:   str,
        split:      str   = "train",
        num_frames: int   = 16,
        crop_size:  int   = 224,
        stride:     int   = 1,
        noise_std:  float = 1.0,
        normalize:  bool  = True,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
    ):
        self.data_dir   = os.path.join(data_dir, split)
        self.num_frames = num_frames
        self.crop_size  = crop_size
        self.stride     = stride
        self.noise_std  = noise_std if split == "train" else 0.0
        self.normalize  = normalize
        self.is_train   = split == "train"
        self.hflip_prob = hflip_prob if split == "train" else 0.0
        self.vflip_prob = vflip_prob if split == "train" else 0.0

        self.files = sorted(glob.glob(os.path.join(self.data_dir, "*.hdf5")))
        assert len(self.files) > 0, f"No HDF5 files found in {self.data_dir}"
        print(f"[{split}] Found {len(self.files)} files")

        self.samples = []
        window = 2 * num_frames

        for fpath in self.files:
            alpha, zeta = self._parse_params(fpath)

            with h5py.File(fpath, "r") as f:
                num_sims   = f["t0_fields/concentration"].shape[0]
                num_tsteps = f["t0_fields/concentration"].shape[1]

            for sim_idx in range(num_sims):
                max_start = num_tsteps - window
                for start in range(0, max_start + 1, stride):
                    self.samples.append((fpath, sim_idx, start, alpha, zeta))

        print(f"[{split}] Total samples: {len(self.samples)}")

    def _parse_params(self, fpath: str):
        name  = os.path.basename(fpath)
        parts = name.replace(".hdf5", "").split("_")
        zeta_idx  = parts.index("zeta")
        alpha_idx = parts.index("alpha")
        zeta  = float(parts[zeta_idx + 1])
        alpha = float(parts[alpha_idx + 1])
        return alpha, zeta

    def _load_clip(self, fpath: str, sim_idx: int, start: int) -> np.ndarray:
        """Load 32 consecutive frames and stack all 11 channels. Returns (32, 11, 256, 256)."""
        end = start + 2 * self.num_frames

        with h5py.File(fpath, "r") as f:
            conc = f["t0_fields/concentration"][sim_idx, start:end]
            conc = conc[:, np.newaxis, :, :]                           # (32, 1, 256, 256)

            vel  = f["t1_fields/velocity"][sim_idx, start:end]         # (32, 256, 256, 2)
            vel  = vel.transpose(0, 3, 1, 2)                           # (32, 2, 256, 256)

            D    = f["t2_fields/D"][sim_idx, start:end]                # (32, 256, 256, 2, 2)
            D    = D.reshape(2 * self.num_frames, 256, 256, 4)
            D    = D.transpose(0, 3, 1, 2)                             # (32, 4, 256, 256)

            E    = f["t2_fields/E"][sim_idx, start:end]
            E    = E.reshape(2 * self.num_frames, 256, 256, 4)
            E    = E.transpose(0, 3, 1, 2)                             # (32, 4, 256, 256)

        clip = np.concatenate([conc, vel, D, E], axis=1)               # (32, 11, 256, 256)
        return clip.astype(np.float32)

    def _random_crop(self, clip: np.ndarray) -> np.ndarray:
        _, _, H, W = clip.shape
        top  = np.random.randint(0, H - self.crop_size + 1)
        left = np.random.randint(0, W - self.crop_size + 1)
        return clip[:, :, top:top + self.crop_size, left:left + self.crop_size]

    def _center_crop(self, clip: np.ndarray) -> np.ndarray:
        _, _, H, W = clip.shape
        top  = (H - self.crop_size) // 2
        left = (W - self.crop_size) // 2
        return clip[:, :, top:top + self.crop_size, left:left + self.crop_size]

    def _normalize(self, clip: np.ndarray) -> np.ndarray:
        """Per-sample, per-channel z-score normalization across T, H, W."""
        mean = clip.mean(axis=(0, 2, 3), keepdims=True)
        std  = clip.std(axis=(0, 2, 3),  keepdims=True) + 1e-6
        return (clip - mean) / std

    def _hflip(self, clip: np.ndarray) -> np.ndarray:
        """
        Horizontal flip (reflect x-axis, flip W dimension).
        Negates vx and off-diagonal tensor components to preserve physics.
        clip shape: (T, 11, H, W)
        """
        clip = clip[:, :, :, ::-1].copy()
        clip *= _HFLIP_SIGN[None, :, None, None]
        return clip

    def _vflip(self, clip: np.ndarray) -> np.ndarray:
        """
        Vertical flip (reflect y-axis, flip H dimension).
        Negates vy and off-diagonal tensor components to preserve physics.
        clip shape: (T, 11, H, W)
        """
        clip = clip[:, :, ::-1, :].copy()
        clip *= _VFLIP_SIGN[None, :, None, None]
        return clip

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        fpath, sim_idx, start, alpha, zeta = self.samples[idx]

        clip = self._load_clip(fpath, sim_idx, start)   # (32, 11, 256, 256)

        if self.is_train:
            clip = self._random_crop(clip)
        else:
            clip = self._center_crop(clip)

        if self.normalize:
            clip = self._normalize(clip)

        # Physics-aware spatial flips (training only)
        if self.hflip_prob > 0 and np.random.random() < self.hflip_prob:
            clip = self._hflip(clip)
        if self.vflip_prob > 0 and np.random.random() < self.vflip_prob:
            clip = self._vflip(clip)

        # Gaussian noise (training only, applied after flips)
        if self.noise_std > 0:
            clip = clip + np.random.randn(*clip.shape).astype(np.float32) * self.noise_std

        context = clip[:self.num_frames]
        target  = clip[self.num_frames:]

        context = torch.from_numpy(context).permute(1, 0, 2, 3)   # (11, 16, 224, 224)
        target  = torch.from_numpy(target).permute(1, 0, 2, 3)

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
    dataset  = ActiveMatterDataset(data_dir, split="train", stride=1)

    sample = dataset[0]
    print(f"context shape: {sample['context'].shape}")
    print(f"target  shape: {sample['target'].shape}")
    print(f"alpha:         {sample['alpha'].item()}")
    print(f"zeta:          {sample['zeta'].item()}")
    print(f"context mean:  {sample['context'].mean():.4f}")
    print(f"context std:   {sample['context'].std():.4f}")
    print(f"Total samples: {len(dataset)}")
