"""
ActiveMatterDataset for ViT-JEPA.
Returns single 16-frame clips; token-level masking happens in the training loop.
"""

import os
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class ActiveMatterDataset(Dataset):
    """
    Single-clip dataset for ViT-JEPA self-supervised training.

    Returns dicts with:
      "frames": (C=11, T=16, H=224, W=224)
      "alpha":  float
      "zeta":   float
    """
    def __init__(
        self,
        data_dir:   str,
        split:      str   = "train",
        num_frames: int   = 16,
        crop_size:  int   = 224,
        stride:     int   = 1,
        noise_std:  float = 1.0,
    ):
        self.data_dir   = os.path.join(data_dir, split)
        self.num_frames = num_frames
        self.crop_size  = crop_size
        self.noise_std  = noise_std if split == "train" else 0.0
        self.is_train   = split == "train"

        self.files = sorted(glob.glob(os.path.join(self.data_dir, "*.hdf5")))
        assert len(self.files) > 0, f"No HDF5 files found in {self.data_dir}"
        print(f"[{split}] Found {len(self.files)} files")

        self.samples = []
        for fpath in self.files:
            alpha, zeta = self._parse_params(fpath)
            with h5py.File(fpath, "r") as f:
                num_sims   = f["t0_fields/concentration"].shape[0]
                num_tsteps = f["t0_fields/concentration"].shape[1]
            for sim_idx in range(num_sims):
                for start in range(0, num_tsteps - num_frames + 1, stride):
                    self.samples.append((fpath, sim_idx, start, alpha, zeta))

        print(f"[{split}] Total samples: {len(self.samples)}")

    def _parse_params(self, fpath: str):
        parts = os.path.basename(fpath).replace(".hdf5", "").split("_")
        zeta  = float(parts[parts.index("zeta")  + 1])
        alpha = float(parts[parts.index("alpha") + 1])
        return alpha, zeta

    def _load_clip(self, fpath: str, sim_idx: int, start: int) -> np.ndarray:
        end = start + self.num_frames
        with h5py.File(fpath, "r") as f:
            conc = f["t0_fields/concentration"][sim_idx, start:end][:, np.newaxis]
            vel  = f["t1_fields/velocity"][sim_idx, start:end].transpose(0, 3, 1, 2)
            D    = f["t2_fields/D"][sim_idx, start:end].reshape(self.num_frames, 256, 256, 4).transpose(0, 3, 1, 2)
            E    = f["t2_fields/E"][sim_idx, start:end].reshape(self.num_frames, 256, 256, 4).transpose(0, 3, 1, 2)
        return np.concatenate([conc, vel, D, E], axis=1).astype(np.float32)  # (T, 11, 256, 256)

    def _normalize(self, clip: np.ndarray) -> np.ndarray:
        # Per-sample, per-channel z-score across T, H, W
        mean = clip.mean(axis=(0, 2, 3), keepdims=True)
        std  = clip.std(axis=(0, 2, 3),  keepdims=True) + 1e-6
        return (clip - mean) / std

    def _augment(self, clip: np.ndarray) -> np.ndarray:
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
            clip += np.random.randn(*clip.shape).astype(np.float32) * self.noise_std
        return clip

    def _center_crop(self, clip: np.ndarray) -> np.ndarray:
        _, _, H, W = clip.shape
        top  = (H - self.crop_size) // 2
        left = (W - self.crop_size) // 2
        return clip[:, :, top:top + self.crop_size, left:left + self.crop_size]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, sim_idx, start, alpha, zeta = self.samples[idx]
        clip = self._load_clip(fpath, sim_idx, start)  # (T, 11, 256, 256)
        clip = self._normalize(clip)

        if self.is_train:
            clip = self._augment(clip)
        else:
            clip = self._center_crop(clip)

        frames = torch.from_numpy(clip).permute(1, 0, 2, 3)  # (11, T, 224, 224)

        return {
            "frames": frames,
            "alpha":  torch.tensor(alpha, dtype=torch.float32),
            "zeta":   torch.tensor(zeta,  dtype=torch.float32),
        }


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "/scratch/sb10583/data/data"
    ds = ActiveMatterDataset(data_dir, split="train", stride=1)
    s  = ds[0]
    print(f"frames: {s['frames'].shape}  alpha: {s['alpha'].item():.3f}  zeta: {s['zeta'].item():.3f}")
    print(f"mean: {s['frames'].mean():.4f}  std: {s['frames'].std():.4f}")
