"""Google Speech Commands v0.01 dataset for the SUPERB-style KS probe.

SUPERB Keyword Spotting protocol: 10 target keywords + "_unknown_" (any of
the 20 non-target keyword folders) + "_silence_" (random 1 s crops sampled
from ``_background_noise_/*.wav``) → 12 classes.

Splits come from the upstream ``validation_list.txt`` / ``testing_list.txt``;
everything not listed there is train. All clips are normalised to exactly
1 second (16 kHz, ``SAMPLE_LENGTH`` = 16000 samples) — short clips are
zero-padded, long ones truncated — so the loader collates trivially.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

TARGET_KEYWORDS = (
    "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go",
)
UNKNOWN_LABEL = "_unknown_"
SILENCE_LABEL = "_silence_"
CLASS_NAMES: tuple[str, ...] = TARGET_KEYWORDS + (UNKNOWN_LABEL, SILENCE_LABEL)
CLASS_TO_ID: dict[str, int] = {c: i for i, c in enumerate(CLASS_NAMES)}
UNKNOWN_ID = CLASS_TO_ID[UNKNOWN_LABEL]
SILENCE_ID = CLASS_TO_ID[SILENCE_LABEL]
N_CLASSES = len(CLASS_NAMES)

SAMPLE_RATE = 16000
SAMPLE_LENGTH = SAMPLE_RATE  # 1 s


# Default number of synthetic silence items per split. Sized to roughly
# match a single target-keyword class so the probe head doesn't see a
# degenerate one-class blob. (s3prl uses a similar scaling.)
DEFAULT_SILENCE_PER_SPLIT = {"train": 2000, "val": 260, "test": 260}

# SUPERB / s3prl KS protocol: cap "unknown" by sampling roughly one
# target-keyword's worth across the 20 non-target folders, so the 12 classes
# are balanced. ~100 per folder × 20 folders ≈ 2000 unknown / split.
DEFAULT_UNKNOWN_PER_FOLDER = {"train": 100, "val": 14, "test": 14}


@dataclass
class SCItem:
    audio_path: Path
    label: int
    silence_crop: tuple[int, int] | None = None  # (start, end) samples for silence


def _read_list(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


class SpeechCommandsDataset(Dataset):
    """SUPERB-style 12-class Speech Commands v1.

    Args:
        root: directory holding the extracted v0.01 tarball
            (must contain the keyword folders, ``_background_noise_/``,
            ``validation_list.txt``, ``testing_list.txt``).
        split: ``"train"`` | ``"val"`` | ``"test"``.
        sample_rate: required SR (errors on mismatch in the wav files).
        silence_count: number of synthetic silence items to generate; pass
            ``None`` to use :data:`DEFAULT_SILENCE_PER_SPLIT`.
        seed: deterministic seed for silence-crop sampling so train/val/test
            silence sets stay reproducible across runs.
    """

    def __init__(
        self,
        root: Path,
        split: str = "train",
        sample_rate: int = SAMPLE_RATE,
        silence_count: int | None = None,
        unknown_per_folder: int | None = None,
        seed: int = 42,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be train|val|test, got {split!r}")
        self.sample_rate = sample_rate
        self.split = split

        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(f"Speech Commands root not found: {root}")

        val_set = _read_list(root / "validation_list.txt")
        test_set = _read_list(root / "testing_list.txt")

        # Per-split deterministic RNG so silence + unknown subsamples stay
        # disjoint by construction (train/val/test each use a different offset).
        split_seed_offset = {"train": 0, "val": 1, "test": 2}[split]
        rng = np.random.default_rng(seed + split_seed_offset)

        # Subsampling cap for non-target ("unknown") folders. SUPERB/s3prl-style:
        # sample ~100 per non-target folder to keep the 12 classes balanced.
        # Pass 0 / negative to disable subsampling (use all "unknown").
        unk_cap = (
            unknown_per_folder
            if unknown_per_folder is not None
            else DEFAULT_UNKNOWN_PER_FOLDER[split]
        )

        items: list[SCItem] = []
        for kw_dir in sorted(root.iterdir()):
            if not kw_dir.is_dir() or kw_dir.name.startswith("_"):
                continue
            label = CLASS_TO_ID.get(kw_dir.name, UNKNOWN_ID)
            folder_items: list[SCItem] = []
            for f in sorted(kw_dir.glob("*.wav")):
                rel = f"{kw_dir.name}/{f.name}"
                in_val = rel in val_set
                in_test = rel in test_set
                keep = (
                    (split == "val" and in_val)
                    or (split == "test" and in_test)
                    or (split == "train" and not in_val and not in_test)
                )
                if keep:
                    folder_items.append(SCItem(audio_path=f, label=label))

            # For non-target folders, subsample to ``unk_cap`` items to
            # balance the unknown class against the target keywords.
            if label == UNKNOWN_ID and unk_cap and unk_cap > 0 and len(folder_items) > unk_cap:
                pick = rng.choice(len(folder_items), size=unk_cap, replace=False)
                folder_items = [folder_items[i] for i in sorted(pick.tolist())]
            items.extend(folder_items)

        # Synthesize silence items by cropping background noise files.
        bg_dir = root / "_background_noise_"
        bg_files = sorted(p for p in bg_dir.glob("*.wav"))
        if not bg_files:
            raise FileNotFoundError(f"No background-noise wavs under {bg_dir}")

        n_silence = (
            silence_count
            if silence_count is not None
            else DEFAULT_SILENCE_PER_SPLIT[split]
        )
        bg_info = [(p, sf.info(str(p))) for p in bg_files]
        for _ in range(n_silence):
            bg_path, info = bg_info[int(rng.integers(len(bg_info)))]
            max_start = max(0, info.frames - SAMPLE_LENGTH)
            start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
            items.append(
                SCItem(
                    audio_path=bg_path,
                    label=SILENCE_ID,
                    silence_crop=(start, start + SAMPLE_LENGTH),
                )
            )

        self.items = items
        n_per_class = [0] * N_CLASSES
        for it in items:
            n_per_class[it.label] += 1
        print(
            f"SpeechCommandsDataset[{split}]: {len(items)} items "
            f"(per-class counts: {dict(zip(CLASS_NAMES, n_per_class))})"
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        if item.silence_crop is not None:
            start, end = item.silence_crop
            wav, sr = sf.read(
                str(item.audio_path), start=start, frames=end - start, dtype="float32"
            )
        else:
            wav, sr = sf.read(str(item.audio_path), dtype="float32")
        if sr != self.sample_rate:
            raise ValueError(f"sample rate {sr} != {self.sample_rate} ({item.audio_path})")
        if wav.ndim > 1:
            wav = wav.mean(axis=-1)
        # Pad or truncate to exactly 1 s.
        if wav.shape[0] < SAMPLE_LENGTH:
            wav = np.pad(wav, (0, SAMPLE_LENGTH - wav.shape[0]))
        elif wav.shape[0] > SAMPLE_LENGTH:
            wav = wav[:SAMPLE_LENGTH]
        return {
            "wav": torch.from_numpy(wav),
            "label": int(item.label),
            "n_samples": SAMPLE_LENGTH,
        }


def collate_utt(batch: list[dict]) -> dict:
    """Stack equally-shaped utterances and emit a length tensor.

    ``wav_lens`` is included so utterance-level probes can mask out any padded
    frames consistently with variable-length collation used elsewhere — even
    though KS items are all 1 s.
    """
    wavs = torch.stack([b["wav"] for b in batch], dim=0)
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    wav_lens = torch.tensor([b["n_samples"] for b in batch], dtype=torch.long)
    return {"wav": wavs, "labels": labels, "wav_lens": wav_lens}
