"""LibriSpeech datasets paired with per-frame supervision.

Two dataset variants share most of the file-discovery / windowing logic:
- :class:`LibriSpeechPhonemeDataset` — wav + per-frame phoneme labels (from
  MFA TextGrid). Used by the phoneme classification probe and, by deriving
  boundary labels at training time, also by the boundary detection probe.
- :class:`LibriSpeechF0Dataset` — wav + per-frame F0 (Hz) array
  pre-computed by ``src/data/extract_f0.py``. Used by the F0 regression
  probe. Unvoiced frames are 0.0 in the F0 array.

Both yield variable-length items; use ``collate_padded`` /
``collate_padded_f0`` for batching.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from src.data.alignments import (
    SILENCE_ID,
    alignment_path_for_utt,
    parse_textgrid_phones,
    segments_to_frame_labels,
)

PAD_LABEL = -100  # ignore_index for cross-entropy on padded frames.
F0_PAD = 0.0      # padded F0 frames (also marks "unvoiced" naturally)


def utt_id_to_relpath(utt_id: str) -> Path:
    """``"1272-128104-0000"`` -> ``Path("1272/128104/1272-128104-0000")``."""
    spk, chap, _ = utt_id.split("-", 2)
    return Path(spk) / chap / utt_id


@dataclass
class LibriSpeechItem:
    audio_path: Path
    textgrid_path: Path
    utt_id: str


class LibriSpeechPhonemeDataset(Dataset):
    """LibriSpeech audio + frame-level phoneme labels.

    Args:
        librispeech_dir: split-level audio dir, e.g. ``.../LibriSpeech/dev-clean``.
        alignments_dir: split-level alignments dir, e.g.
            ``.../alignments/LibriSpeech/dev-clean``. Must contain per-utterance
            ``*.TextGrid`` files mirroring the audio layout.
        sample_rate: required audio sample rate (errors on mismatch).
        frame_hop_seconds: hop in seconds for frame-level labels (default
            10 ms = matches the encoder's mel-spec frame rate at hop=160).
        max_seconds: optional truncation per utterance.
        min_seconds: drop utterances shorter than this.
    """

    def __init__(
        self,
        librispeech_dir: Path,
        alignments_dir: Path,
        sample_rate: int = 16000,
        frame_hop_seconds: float = 0.01,
        max_seconds: float | None = 16.0,
        min_seconds: float = 1.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_hop_seconds = frame_hop_seconds
        self.max_seconds = max_seconds
        self.min_seconds = min_seconds

        librispeech_dir = Path(librispeech_dir)
        alignments_dir = Path(alignments_dir)
        flac_files = sorted(librispeech_dir.rglob("*.flac"))

        items: list[LibriSpeechItem] = []
        skipped_no_align = 0
        skipped_short = 0
        for f in flac_files:
            utt_id = f.stem  # e.g. "1272-128104-0000"
            tg_path = alignment_path_for_utt(alignments_dir, utt_id)
            if not tg_path.exists():
                skipped_no_align += 1
                continue
            info = sf.info(str(f))
            if info.duration < self.min_seconds:
                skipped_short += 1
                continue
            items.append(
                LibriSpeechItem(audio_path=f, textgrid_path=tg_path, utt_id=utt_id)
            )

        if not items:
            raise ValueError(
                f"No LibriSpeech utterances with matching TextGrid alignments under "
                f"{librispeech_dir} (alignments from {alignments_dir})."
            )
        self.items = items
        print(
            f"LibriSpeechPhonemeDataset: {len(items)} utts loaded "
            f"(skipped: no_align={skipped_no_align}, short={skipped_short})"
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        wav, sr = sf.read(str(item.audio_path), dtype="float32", always_2d=False)
        if sr != self.sample_rate:
            raise ValueError(f"sample rate {sr} != {self.sample_rate} ({item.audio_path})")
        if wav.ndim > 1:
            wav = wav.mean(axis=-1)
        if self.max_seconds is not None:
            max_samples = int(self.max_seconds * self.sample_rate)
            if wav.shape[0] > max_samples:
                wav = wav[:max_samples]

        hop_samples = int(round(self.sample_rate * self.frame_hop_seconds))
        # Match torchaudio MelSpectrogram(center=True) frame count exactly.
        n_frames = wav.shape[0] // hop_samples + 1
        segments = parse_textgrid_phones(item.textgrid_path)
        labels = segments_to_frame_labels(
            segments,
            n_frames=n_frames,
            frame_hop_seconds=self.frame_hop_seconds,
            silence_id=SILENCE_ID,
        )
        return {
            "wav": torch.from_numpy(wav),
            "labels": torch.tensor(labels, dtype=torch.long),
            "n_samples": int(wav.shape[0]),
            "n_frames": int(n_frames),
            "utt_id": item.utt_id,
        }


def collate_padded(batch: list[dict]) -> dict:
    """Right-pad waveforms (zeros) and phoneme labels (PAD_LABEL = -100)."""
    max_samples = max(b["n_samples"] for b in batch)
    max_frames = max(b["n_frames"] for b in batch)

    wavs = torch.zeros(len(batch), max_samples, dtype=torch.float32)
    labels = torch.full((len(batch), max_frames), PAD_LABEL, dtype=torch.long)
    label_mask = torch.zeros(len(batch), max_frames, dtype=torch.bool)

    for i, b in enumerate(batch):
        wavs[i, : b["n_samples"]] = b["wav"]
        labels[i, : b["n_frames"]] = b["labels"]
        label_mask[i, : b["n_frames"]] = True

    return {
        "wav": wavs,
        "labels": labels,
        "label_mask": label_mask,
        "utt_ids": [b["utt_id"] for b in batch],
    }


# ---------------------------------------------------------------------------
# F0 dataset / collate
# ---------------------------------------------------------------------------


@dataclass
class LibriSpeechF0Item:
    audio_path: Path
    f0_path: Path
    utt_id: str


class LibriSpeechF0Dataset(Dataset):
    """LibriSpeech audio + per-frame F0 (Hz) for the F0 regression probe.

    F0 arrays are pre-computed by ``src/data/extract_f0.py`` and stored as
    ``.npy`` files under ``f0_dir``, mirroring the audio layout::

        f0_dir/<spk>/<chapter>/<utt>.npy

    Each ``.npy`` is float32 F0 in Hz at the same 10 ms hop the encoder
    operates at. Unvoiced frames are 0.0 (also matches ``F0_PAD``).
    """

    def __init__(
        self,
        librispeech_dir: Path,
        f0_dir: Path,
        sample_rate: int = 16000,
        frame_hop_seconds: float = 0.01,
        max_seconds: float | None = 16.0,
        min_seconds: float = 1.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_hop_seconds = frame_hop_seconds
        self.max_seconds = max_seconds
        self.min_seconds = min_seconds

        librispeech_dir = Path(librispeech_dir)
        f0_dir = Path(f0_dir)
        flac_files = sorted(librispeech_dir.rglob("*.flac"))

        items: list[LibriSpeechF0Item] = []
        skipped_no_f0 = 0
        skipped_short = 0
        for f in flac_files:
            utt_id = f.stem
            f0_path = f0_dir / utt_id_to_relpath(utt_id).with_suffix(".npy")
            if not f0_path.exists():
                skipped_no_f0 += 1
                continue
            info = sf.info(str(f))
            if info.duration < self.min_seconds:
                skipped_short += 1
                continue
            items.append(
                LibriSpeechF0Item(audio_path=f, f0_path=f0_path, utt_id=utt_id)
            )
        if not items:
            raise ValueError(
                f"No LibriSpeech utterances with matching F0 npy under "
                f"{librispeech_dir} (F0 from {f0_dir})."
            )
        self.items = items
        print(
            f"LibriSpeechF0Dataset: {len(items)} utts loaded "
            f"(skipped: no_f0={skipped_no_f0}, short={skipped_short})"
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        wav, sr = sf.read(str(item.audio_path), dtype="float32", always_2d=False)
        if sr != self.sample_rate:
            raise ValueError(f"sample rate {sr} != {self.sample_rate} ({item.audio_path})")
        if wav.ndim > 1:
            wav = wav.mean(axis=-1)
        if self.max_seconds is not None:
            max_samples = int(self.max_seconds * self.sample_rate)
            if wav.shape[0] > max_samples:
                wav = wav[:max_samples]

        f0 = np.load(str(item.f0_path)).astype(np.float32)
        # Truncate F0 array if we truncated the audio. Keep equal-length
        # slicing simple: number of F0 frames at this hop.
        hop_samples = int(round(self.sample_rate * self.frame_hop_seconds))
        n_frames_audio = wav.shape[0] // hop_samples + 1
        if f0.shape[0] > n_frames_audio:
            f0 = f0[:n_frames_audio]
        n_frames = int(f0.shape[0])
        return {
            "wav": torch.from_numpy(wav),
            "f0": torch.from_numpy(f0),
            "n_samples": int(wav.shape[0]),
            "n_frames": n_frames,
            "utt_id": item.utt_id,
        }


def collate_padded_f0(batch: list[dict]) -> dict:
    """Right-pad waveforms (zeros) and F0 (``F0_PAD`` = 0.0)."""
    max_samples = max(b["n_samples"] for b in batch)
    max_frames = max(b["n_frames"] for b in batch)

    wavs = torch.zeros(len(batch), max_samples, dtype=torch.float32)
    f0 = torch.full((len(batch), max_frames), F0_PAD, dtype=torch.float32)
    f0_valid = torch.zeros(len(batch), max_frames, dtype=torch.bool)

    for i, b in enumerate(batch):
        wavs[i, : b["n_samples"]] = b["wav"]
        f0[i, : b["n_frames"]] = b["f0"]
        f0_valid[i, : b["n_frames"]] = True

    return {
        "wav": wavs,
        "f0": f0,
        "f0_valid": f0_valid,  # True where we have an F0 estimate (incl. unvoiced=0)
        "utt_ids": [b["utt_id"] for b in batch],
    }
