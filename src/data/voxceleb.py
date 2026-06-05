"""VoxCeleb1 dataset for the SUPERB-style SID probe (speaker identification).

SUPERB SID uses the upstream ``iden_split.txt`` over VoxCeleb1's 1251
speakers: each line is ``<split_id> <relative_wav_path>`` where ``split_id``
is 1 (train), 2 (val), 3 (test). Speaker label is derived from the leading
``idNNNNN`` segment of the path.

After extraction the audio root mirrors the upstream tarball layout::

    wav/
    ├── id10001/
    │   ├── 1zcIwhmdeo4/
    │   │   ├── 00001.wav
    │   │   └── ...
    │   └── ...
    ├── id10003/
    └── ...

Audio is already 16 kHz mono PCM, so no resampling is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

SAMPLE_RATE = 16000

# iden_split.txt encoding.
SPLIT_TO_NAME = {1: "train", 2: "val", 3: "test"}
NAME_TO_SPLIT = {v: k for k, v in SPLIT_TO_NAME.items()}


@dataclass
class VoxItem:
    audio_path: Path
    speaker_id: int  # contiguous 0..N-1 over the 1251 VoxCeleb1 speakers


def load_iden_split(split_file: Path) -> list[tuple[int, str]]:
    """Parse iden_split.txt into a list of ``(split_id, relative_path)``."""
    out: list[tuple[int, str]] = []
    for line in split_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        sid_str, rel = line.split(maxsplit=1)
        out.append((int(sid_str), rel.strip()))
    return out


def build_speaker_map(entries: list[tuple[int, str]]) -> dict[str, int]:
    """Map ``id10001 -> 0``, ``id10002 -> 1``, ... in sorted order.

    iden_split.txt contains exactly 1251 distinct speakers; we sort the
    raw ids alphabetically so the mapping is deterministic across runs.
    """
    speakers = sorted({rel.split("/", 1)[0] for _, rel in entries})
    return {s: i for i, s in enumerate(speakers)}


class VoxCeleb1Dataset(Dataset):
    """VoxCeleb1, SUPERB iden_split-based speaker identification.

    Args:
        wav_dir: directory that holds the ``id*/`` speaker folders. After
            extracting the upstream zip this is the ``wav/`` subdirectory.
        split_file: path to ``iden_split.txt``.
        split: ``"train"`` | ``"val"`` | ``"test"``.
        sample_rate: required SR (errors on mismatch in the wav files).
        crop_seconds: training-time random crop length. For ``train`` we
            sample a random window and zero-pad if the file is shorter.
            For ``val`` / ``test`` we keep the full utterance up to
            ``max_seconds`` (collate pads to the batch max).
        max_seconds: hard cap for ``val`` / ``test`` files (long talks get
            truncated to keep batches reasonable).
        seed: deterministic seed for the per-item RNG used by train crops.
    """

    def __init__(
        self,
        wav_dir: Path,
        split_file: Path,
        split: str = "train",
        sample_rate: int = SAMPLE_RATE,
        crop_seconds: float = 3.0,
        max_seconds: float = 8.0,
        seed: int = 42,
    ) -> None:
        if split not in NAME_TO_SPLIT:
            raise ValueError(f"split must be train|val|test, got {split!r}")
        self.split = split
        self.sample_rate = sample_rate
        self.crop_seconds = crop_seconds
        self.max_seconds = max_seconds
        self.seed = seed

        wav_dir = Path(wav_dir)
        split_file = Path(split_file)
        if not wav_dir.exists():
            raise FileNotFoundError(f"VoxCeleb wav dir not found: {wav_dir}")
        if not split_file.exists():
            raise FileNotFoundError(f"iden_split.txt not found: {split_file}")

        entries = load_iden_split(split_file)
        self.speaker_to_id = build_speaker_map(entries)
        self.n_speakers = len(self.speaker_to_id)

        target_split = NAME_TO_SPLIT[split]
        items: list[VoxItem] = []
        missing = 0
        for sid, rel in entries:
            if sid != target_split:
                continue
            audio_path = wav_dir / rel
            if not audio_path.exists():
                missing += 1
                continue
            spk = rel.split("/", 1)[0]
            items.append(VoxItem(audio_path=audio_path, speaker_id=self.speaker_to_id[spk]))

        if not items:
            raise ValueError(
                f"No VoxCeleb1 items found for split={split} under {wav_dir} "
                f"using {split_file}. Did extraction finish?"
            )
        self.items = items
        print(
            f"VoxCeleb1Dataset[{split}]: {len(items)} items, "
            f"{self.n_speakers} speakers (missing files skipped: {missing})"
        )

    def __len__(self) -> int:
        return len(self.items)

    def _load_wav(self, idx: int) -> np.ndarray:
        item = self.items[idx]
        info = sf.info(str(item.audio_path))
        if info.samplerate != self.sample_rate:
            raise ValueError(
                f"sample rate {info.samplerate} != {self.sample_rate} "
                f"({item.audio_path})"
            )
        n_total = info.frames
        crop_samples = int(self.crop_seconds * self.sample_rate)
        max_samples = int(self.max_seconds * self.sample_rate)

        if self.split == "train":
            # Random crop with deterministic per-index RNG (seed + idx) so a
            # given (epoch, idx) is reproducible but still varies across idx.
            rng = np.random.default_rng(self.seed * 1_000_003 + idx)
            if n_total <= crop_samples:
                wav, _ = sf.read(str(item.audio_path), dtype="float32", always_2d=False)
                if wav.ndim > 1:
                    wav = wav.mean(axis=-1)
                if wav.shape[0] < crop_samples:
                    wav = np.pad(wav, (0, crop_samples - wav.shape[0]))
            else:
                start = int(rng.integers(0, n_total - crop_samples + 1))
                wav, _ = sf.read(
                    str(item.audio_path),
                    start=start,
                    frames=crop_samples,
                    dtype="float32",
                    always_2d=False,
                )
                if wav.ndim > 1:
                    wav = wav.mean(axis=-1)
            return wav

        # val / test — keep full utterance up to max_seconds
        n_read = min(n_total, max_samples)
        wav, _ = sf.read(
            str(item.audio_path),
            frames=n_read,
            dtype="float32",
            always_2d=False,
        )
        if wav.ndim > 1:
            wav = wav.mean(axis=-1)
        return wav

    def __getitem__(self, idx: int) -> dict:
        wav = self._load_wav(idx)
        item = self.items[idx]
        return {
            "wav": torch.from_numpy(wav),
            "label": int(item.speaker_id),
            "n_samples": int(wav.shape[0]),
        }


def collate_utt_padded(batch: list[dict]) -> dict:
    """Right-pad variable-length wavs and emit a ``wav_lens`` tensor.

    Used by the SID eval loader where utterances are kept at their natural
    length (up to ``max_seconds``); the train loader sees fixed-length
    crops and could also reuse this collate.
    """
    max_samples = max(b["n_samples"] for b in batch)
    wavs = torch.zeros(len(batch), max_samples, dtype=torch.float32)
    for i, b in enumerate(batch):
        wavs[i, : b["n_samples"]] = b["wav"]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    wav_lens = torch.tensor([b["n_samples"] for b in batch], dtype=torch.long)
    return {"wav": wavs, "labels": labels, "wav_lens": wav_lens}
