"""
ActiveMatterDataset v2
======================
Changes from v1:
  - Per-simulation disk cache: one .pt file per (hdf5_file, sim_idx) storing
    the full 81-frame simulation (81, 11, 256, 256) float32.
    Eliminates slow HDF5 multi-dataset reads on every __getitem__ call.
    Window slicing, normalization, crop, and augmentations still run at
    runtime so augmentation diversity is preserved.
  - Physics-aware spatial flip augmentations (horizontal + vertical)
    with proper sign corrections for velocity and off-diagonal tensor components.

Cache design:
  - Cache unit : one simulation  →  (81, 11, 256, 256) float32
  - Cache file : {cache_dir}/{filestem}_sim{sim_idx}.pt
  - Disk usage : ~233 MB per sim; ~157 GB for all 672 train simulations
  - Stride-independent: same cache files used regardless of stride value

Channel layout (indices 0–10):
  0   : concentration (scalar)
  1   : vx  (velocity x)
  2   : vy  (velocity y)
  3   : Dxx | 4: Dxy | 5: Dyx | 6: Dyy  (orientation tensor, row-major)
  7   : Exx | 8: Exy | 9: Eyx | 10: Eyy (strain-rate tensor, row-major)

Flip sign conventions:
  Horizontal flip (x → -x, flip W axis): negate channels 1, 4, 5, 8, 9
  Vertical   flip (y → -y, flip H axis): negate channels 2, 4, 5, 8, 9
"""

import os
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


_HFLIP_SIGN = np.array([1, -1,  1, 1, -1, -1, 1, 1, -1, -1, 1], dtype=np.float32)
_VFLIP_SIGN = np.array([1,  1, -1, 1, -1, -1, 1, 1, -1, -1, 1], dtype=np.float32)


class ActiveMatterDataset(Dataset):
    """
    Dataset for active matter physical simulations.

    Returns dicts with:
      "context": (C=11, T=16, H=224, W=224)
      "target":  (C=11, T=16, H=224, W=224)
      "alpha":   float
      "zeta":    float

    Args:
        data_dir:   path to active_matter/data/{split}/
        split:      "train" | "valid" | "test"
        num_frames: frames per clip half (default 16)
        crop_size:  spatial crop size (default 224)
        stride:     sliding window stride (default 1)
        noise_std:  Gaussian noise std (default 1.0, 0 to disable)
        normalize:  per-sample z-score normalization (default True)
        hflip_prob: horizontal flip probability (train only)
        vflip_prob: vertical flip probability (train only)
        cache_dir:  directory for per-simulation .pt cache files.
                    If None, reads directly from HDF5 every call.
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
        cache_dir:  str   = None,
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
        self.cache_dir  = cache_dir

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        self.files = sorted(glob.glob(os.path.join(self.data_dir, "*.hdf5")))
        assert len(self.files) > 0, f"No HDF5 files found in {self.data_dir}"
        print(f"[{split}] Found {len(self.files)} files")

        self.samples = []
        window = 2 * num_frames   # 32 frames total

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
        if cache_dir:
            print(f"[{split}] Cache directory: {cache_dir}")

    # ─────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────

    def _parse_params(self, fpath: str):
        name  = os.path.basename(fpath)
        parts = name.replace(".hdf5", "").split("_")
        zeta  = float(parts[parts.index("zeta")  + 1])
        alpha = float(parts[parts.index("alpha") + 1])
        return alpha, zeta

    def _get_sim_cache_path(self, fpath: str, sim_idx: int) -> str:
        stem = os.path.splitext(os.path.basename(fpath))[0]
        return os.path.join(self.cache_dir, f"{stem}_sim{sim_idx}.pt")

    # ─────────────────────────────────────────────────────────────
    # Simulation-level loading (cache or HDF5)
    # ─────────────────────────────────────────────────────────────

    def _load_sim_from_hdf5(self, fpath: str, sim_idx: int) -> torch.Tensor:
        """
        Load a full simulation (all 81 frames) from HDF5 and stack channels.
        Returns: (81, 11, 256, 256) float32 tensor.
        """
        with h5py.File(fpath, "r") as f:
            T = f["t0_fields/concentration"].shape[1]   # 81

            conc = f["t0_fields/concentration"][sim_idx]          # (81, 256, 256)
            conc = conc[:, np.newaxis, :, :]                      # (81, 1, 256, 256)

            vel  = f["t1_fields/velocity"][sim_idx]               # (81, 256, 256, 2)
            vel  = vel.transpose(0, 3, 1, 2)                      # (81, 2, 256, 256)

            D    = f["t2_fields/D"][sim_idx]                      # (81, 256, 256, 2, 2)
            D    = D.reshape(T, 256, 256, 4).transpose(0, 3, 1, 2)

            E    = f["t2_fields/E"][sim_idx]
            E    = E.reshape(T, 256, 256, 4).transpose(0, 3, 1, 2)

        sim = np.concatenate([conc, vel, D, E], axis=1).astype(np.float32)
        return torch.from_numpy(sim)   # (81, 11, 256, 256)

    def _load_sim(self, fpath: str, sim_idx: int) -> torch.Tensor:
        """
        Load a full simulation from cache if available, otherwise from HDF5.
        On first access, writes the .pt cache file for future epochs.
        Returns: (81, 11, 256, 256) float32 tensor.
        """
        if self.cache_dir is None:
            return self._load_sim_from_hdf5(fpath, sim_idx)

        cache_path = self._get_sim_cache_path(fpath, sim_idx)

        if os.path.exists(cache_path):
            return torch.load(cache_path, map_location="cpu", weights_only=True)

        # Cache miss: read from HDF5 and write to disk for future epochs
        sim = self._load_sim_from_hdf5(fpath, sim_idx)
        torch.save(sim, cache_path)
        return sim

    # ─────────────────────────────────────────────────────────────
    # Window extraction and preprocessing
    # ─────────────────────────────────────────────────────────────

    def _extract_window(self, sim: torch.Tensor, start: int) -> np.ndarray:
        """Slice a 32-frame window from the cached simulation tensor."""
        window = sim[start : start + 2 * self.num_frames]   # (32, 11, 256, 256)
        return window.numpy()

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
        """Horizontal flip with velocity/tensor sign correction."""
        clip = clip[:, :, :, ::-1].copy()
        clip *= _HFLIP_SIGN[None, :, None, None]
        return clip

    def _vflip(self, clip: np.ndarray) -> np.ndarray:
        """Vertical flip with velocity/tensor sign correction."""
        clip = clip[:, :, ::-1, :].copy()
        clip *= _VFLIP_SIGN[None, :, None, None]
        return clip

    # ─────────────────────────────────────────────────────────────
    # Dataset interface
    # ─────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        fpath, sim_idx, start, alpha, zeta = self.samples[idx]

        # Load full simulation (from cache .pt or HDF5) then slice window
        sim  = self._load_sim(fpath, sim_idx)          # (81, 11, 256, 256)
        clip = self._extract_window(sim, start)        # (32, 11, 256, 256)

        # Spatial crop
        clip = self._random_crop(clip) if self.is_train else self._center_crop(clip)

        # Normalize
        if self.normalize:
            clip = self._normalize(clip)

        # Physics-aware spatial flips (training only)
        if self.hflip_prob > 0 and np.random.random() < self.hflip_prob:
            clip = self._hflip(clip)
        if self.vflip_prob > 0 and np.random.random() < self.vflip_prob:
            clip = self._vflip(clip)

        # Gaussian noise (training only)
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
    import sys
    data_dir  = sys.argv[1] if len(sys.argv) > 1 else "/scratch/vc2836/DL/data/active_matter/data"
    cache_dir = sys.argv[2] if len(sys.argv) > 2 else "/scratch/vc2836/DL/data/active_matter/data_vit_v2"

    dataset = ActiveMatterDataset(
        data_dir  = data_dir,
        split     = "train",
        stride    = 1,
        cache_dir = cache_dir,
    )

    sample = dataset[0]
    print(f"context shape: {sample['context'].shape}")
    print(f"target  shape: {sample['target'].shape}")
    print(f"alpha:         {sample['alpha'].item()}")
    print(f"zeta:          {sample['zeta'].item()}")
    print(f"context mean:  {sample['context'].mean():.4f}")
    print(f"context std:   {sample['context'].std():.4f}")
    print(f"Total samples: {len(dataset)}")
