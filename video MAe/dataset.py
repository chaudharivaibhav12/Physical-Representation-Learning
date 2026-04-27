"""
VideoMAE Dataset for Active Matter Physics Simulations
======================================================
Single-view dataset (no two-view split — masking is the only augmentation).

Returns dicts with:
  "frames": (C=11, T=16, H=224, W=224)  Tensor
  "alpha":  float scalar (not used during training)
  "zeta":   float scalar (not used during training)

HDF5 structure (identical to ViT-JEPA / VICReg datasets):
  t0_fields/concentration  (num_sims, num_tsteps, 256, 256)       → 1 channel
  t1_fields/velocity       (num_sims, num_tsteps, 256, 256, 2)    → 2 channels
  t2_fields/D              (num_sims, num_tsteps, 256, 256, 2, 2) → 4 channels
  t2_fields/E              (num_sims, num_tsteps, 256, 256, 2, 2) → 4 channels
  Total: 11 channels, 81 timesteps, 256x256 spatial
"""

import os
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class VideoMAEDataset(Dataset):
    """
    Single-view active matter dataset for VideoMAE training.

    Args:
        data_dir:   path to active_matter/data/ (contains train/, valid/, test/)
        split:      "train" | "valid" | "test"
        num_frames: frames per clip (default 16)
        crop_size:  spatial crop size (default 224)
        stride:     sliding window stride (default 1)
        normalize:  per-sample per-channel z-score normalization (default True)
    """
    def __init__(
        self,
        data_dir:   str,
        split:      str  = "train",
        num_frames: int  = 16,
        crop_size:  int  = 224,
        stride:     int  = 1,
        normalize:  bool = True,
    ):
        self.data_dir   = os.path.join(data_dir, split)
        self.num_frames = num_frames
        self.crop_size  = crop_size
        self.stride     = stride
        self.normalize  = normalize
        self.is_train   = (split == "train")

        self.files = sorted(glob.glob(os.path.join(self.data_dir, "*.hdf5")))
        assert len(self.files) > 0, f"No HDF5 files found in {self.data_dir}"
        print(f"[{split}] Found {len(self.files)} HDF5 files")

        # Build sample index: (file_path, sim_idx, start_frame, alpha, zeta)
        self.samples = []
        for fpath in self.files:
            alpha, zeta = self._parse_params(fpath)
            with h5py.File(fpath, "r") as f:
                num_sims   = f["t0_fields/concentration"].shape[0]   # 3
                num_tsteps = f["t0_fields/concentration"].shape[1]   # 81
            max_start = num_tsteps - num_frames  # 81 - 16 = 65
            for sim_idx in range(num_sims):
                for start in range(0, max_start + 1, stride):
                    self.samples.append((fpath, sim_idx, start, alpha, zeta))

        print(f"[{split}] Total samples: {len(self.samples)}")

    def _parse_params(self, fpath: str):
        name  = os.path.basename(fpath).replace(".hdf5", "")
        parts = name.split("_")
        zeta  = float(parts[parts.index("zeta")  + 1])
        alpha = float(parts[parts.index("alpha") + 1])
        return alpha, zeta

    def _load_clip(self, fpath: str, sim_idx: int, start: int) -> np.ndarray:
        """Load num_frames consecutive frames and stack all 11 channels.
        Returns: (num_frames, 11, 256, 256)
        """
        end = start + self.num_frames
        with h5py.File(fpath, "r") as f:
            conc = f["t0_fields/concentration"][sim_idx, start:end]            # (T, 256, 256)
            conc = conc[:, np.newaxis]                                          # (T, 1, 256, 256)

            vel  = f["t1_fields/velocity"][sim_idx, start:end]                 # (T, 256, 256, 2)
            vel  = vel.transpose(0, 3, 1, 2)                                   # (T, 2, 256, 256)

            D    = f["t2_fields/D"][sim_idx, start:end]                        # (T, 256, 256, 2, 2)
            D    = D.reshape(self.num_frames, 256, 256, 4).transpose(0, 3, 1, 2)  # (T, 4, 256, 256)

            E    = f["t2_fields/E"][sim_idx, start:end]                        # (T, 256, 256, 2, 2)
            E    = E.reshape(self.num_frames, 256, 256, 4).transpose(0, 3, 1, 2)  # (T, 4, 256, 256)

        return np.concatenate([conc, vel, D, E], axis=1).astype(np.float32)    # (T, 11, 256, 256)

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

    def _random_flip_rotate(self, clip: np.ndarray) -> np.ndarray:
        if np.random.rand() > 0.5:
            clip = clip[:, :, :, ::-1].copy()
        if np.random.rand() > 0.5:
            clip = clip[:, :, ::-1, :].copy()
        k = np.random.randint(0, 4)
        if k > 0:
            clip = np.rot90(clip, k=k, axes=(2, 3)).copy()
        return clip

    def _normalize(self, clip: np.ndarray) -> np.ndarray:
        mean = clip.mean(axis=(0, 2, 3), keepdims=True)   # (1, 11, 1, 1)
        std  = clip.std(axis=(0, 2, 3),  keepdims=True) + 1e-6
        return (clip - mean) / std

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        fpath, sim_idx, start, alpha, zeta = self.samples[idx]

        clip = self._load_clip(fpath, sim_idx, start)  # (T, 11, 256, 256)

        if self.normalize:
            clip = self._normalize(clip)

        if self.is_train:
            clip = self._random_crop(clip)
            clip = self._random_flip_rotate(clip)
        else:
            clip = self._center_crop(clip)

        # (T, C, H, W) → (C, T, H, W)
        frames = torch.from_numpy(clip).permute(1, 0, 2, 3)  # (11, 16, 224, 224)

        return {
            "frames": frames,
            "alpha":  torch.tensor(alpha, dtype=torch.float32),
            "zeta":   torch.tensor(zeta,  dtype=torch.float32),
        }


# ──────────────────────────────────────────────────────────────
# Eval-only dataset: same as above but always no augmentation
# (used by evaluate.py to extract embeddings)
# ──────────────────────────────────────────────────────────────

class VideoMAEEval(Dataset):
    """
    Eval dataset — center crop, no augmentation, stride=2 for speed.
    Used by evaluate.py for linear probe / kNN extraction.
    """
    def __init__(
        self,
        data_dir:   str,
        split:      str  = "valid",
        num_frames: int  = 16,
        crop_size:  int  = 224,
        stride:     int  = 2,
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
            max_start = num_tsteps - num_frames
            for sim_idx in range(num_sims):
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
        end = start + self.num_frames

        with h5py.File(fpath, "r") as f:
            conc = f["t0_fields/concentration"][sim_idx, start:end][:, np.newaxis]
            vel  = f["t1_fields/velocity"][sim_idx, start:end].transpose(0, 3, 1, 2)
            D    = f["t2_fields/D"][sim_idx, start:end].reshape(self.num_frames, 256, 256, 4).transpose(0, 3, 1, 2)
            E    = f["t2_fields/E"][sim_idx, start:end].reshape(self.num_frames, 256, 256, 4).transpose(0, 3, 1, 2)

        clip = np.concatenate([conc, vel, D, E], axis=1).astype(np.float32)   # (T, 11, 256, 256)

        mean = clip.mean(axis=(0, 2, 3), keepdims=True)
        std  = clip.std(axis=(0, 2, 3),  keepdims=True) + 1e-6
        clip = (clip - mean) / std

        top  = (256 - self.crop_size) // 2
        left = (256 - self.crop_size) // 2
        clip = clip[:, :, top:top + self.crop_size, left:left + self.crop_size]

        frames = torch.from_numpy(clip).permute(1, 0, 2, 3)  # (11, 16, 224, 224)
        return {
            "frames": frames,
            "alpha":  torch.tensor(alpha, dtype=torch.float32),
            "zeta":   torch.tensor(zeta,  dtype=torch.float32),
        }


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "/scratch/sb10583/data/data"

    ds = VideoMAEDataset(data_dir, split="train", stride=1)
    s  = ds[0]
    print(f"frames: {s['frames'].shape}  dtype={s['frames'].dtype}")
    print(f"alpha: {s['alpha'].item():.3f}   zeta: {s['zeta'].item():.3f}")
    print(f"mean: {s['frames'].mean():.4f}  std: {s['frames'].std():.4f}")
