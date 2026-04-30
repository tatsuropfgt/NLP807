"""Audio datasets for self-supervised pretraining.

Returns mono waveforms windowed to a fixed length. Used for both music
(POP909-rendered) and the non-music control data, since the file layout is
just a directory of wav files.

The dataset supports a deterministic train/val split by seeded shuffle of the
file list, and a deterministic windowing mode for stable validation loss.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

Split = Literal["train", "val", "all"]


class WavFolderDataset(Dataset):
    """Recursively load wavs from ``root`` and return fixed-length windows.

    Args:
        root: directory to scan for ``*.wav``.
        sample_rate: expected sample rate. Files at a different rate raise.
        window_seconds: length of returned audio window.
        pad_short: if True, pad files shorter than the window with zeros;
            otherwise, raise on encountering one.
        split: ``"train"``, ``"val"``, or ``"all"``. Splits are made by a
            seeded shuffle of the sorted file list, then taking the first
            ``val_ratio`` fraction as val.
        val_ratio: fraction of files reserved for ``"val"``.
        split_seed: seed for the train/val split (independent from window
            randomness). Same seed across train/val ensures complementary
            file lists.
        deterministic_window: if True, always return the *middle* window of
            each file. Used for validation so that the loss is reproducible.
    """

    def __init__(
        self,
        root: str | Path,
        sample_rate: int = 16000,
        window_seconds: float = 5.0,
        pad_short: bool = True,
        split: Split = "all",
        val_ratio: float = 0.05,
        split_seed: int = 42,
        deterministic_window: bool = False,
    ) -> None:
        self.root = Path(root)
        self.sample_rate = sample_rate
        self.window_samples = int(round(window_seconds * sample_rate))
        self.pad_short = pad_short
        self.deterministic_window = deterministic_window

        all_files = sorted(self.root.rglob("*.wav"))
        if not all_files:
            raise FileNotFoundError(f"No .wav files under {self.root}")

        if split == "all":
            files = all_files
        else:
            order = list(range(len(all_files)))
            random.Random(split_seed).shuffle(order)
            n_val = max(1, int(round(len(all_files) * val_ratio)))
            val_idx = set(order[:n_val])
            if split == "train":
                files = [f for i, f in enumerate(all_files) if i not in val_idx]
            elif split == "val":
                files = [f for i, f in enumerate(all_files) if i in val_idx]
            else:
                raise ValueError(f"unknown split: {split}")
        self.files: list[Path] = files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.files[idx]
        info = sf.info(str(path))
        if info.samplerate != self.sample_rate:
            raise ValueError(
                f"sample rate mismatch: got {info.samplerate}, expected {self.sample_rate} ({path})"
            )
        n = info.frames

        if n >= self.window_samples:
            if self.deterministic_window:
                start = max(0, (n - self.window_samples) // 2)
            else:
                start = random.randint(0, n - self.window_samples)
            wav, _ = sf.read(
                str(path),
                start=start,
                stop=start + self.window_samples,
                dtype="float32",
                always_2d=False,
            )
        else:
            if not self.pad_short:
                raise ValueError(f"file shorter than window: {path} ({n} < {self.window_samples})")
            wav, _ = sf.read(str(path), dtype="float32", always_2d=False)
            pad = self.window_samples - wav.shape[-1]
            wav = np.pad(wav, (0, pad))

        if wav.ndim > 1:
            wav = wav.mean(axis=-1)
        return torch.from_numpy(wav)


def collate_waveforms(batch: list[torch.Tensor]) -> torch.Tensor:
    """Stack equal-length waveforms into (B, T)."""
    return torch.stack(batch, dim=0)
