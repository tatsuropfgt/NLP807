"""Extract per-frame F0 from LibriSpeech audio using WORLD (pyworld).

Saves an ``.npy`` file per utterance, mirroring the audio layout::

    f0_root/
        {split}/<spk>/<chapter>/<utt_id>.npy

Each ``.npy`` is a 1-D ``float32`` array of F0 values in Hz at a 10 ms hop
(matching the encoder's mel-spec frame rate). Unvoiced frames are 0.0.

Pipeline: DIO (rough F0) -> StoneMask (refined F0). On a typical laptop
core this runs at roughly 5-10x real time per utterance; the script
parallelizes via ``multiprocessing`` so a full LibriSpeech split fits in
under an hour on a multi-core machine.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import warnings
from pathlib import Path

# Silence pyworld's pkg_resources DeprecationWarning (printed once per
# worker on import). Apply BEFORE importing pyworld so the filter is
# active when each spawned worker re-executes this module.
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pyworld")

import numpy as np
import pyworld
import soundfile as sf
from tqdm import tqdm

DEFAULT_LIBRI_ROOT = Path("/workspace/i_tatsuro/data/LibriSpeech/LibriSpeech")
DEFAULT_F0_ROOT = Path("/workspace/i_tatsuro/data/LibriSpeech/f0")
DEFAULT_FRAME_PERIOD_MS = 10.0  # 10 ms hop -> 100 Hz frame rate
DEFAULT_F0_FLOOR = 50.0
DEFAULT_F0_CEIL = 600.0


def utt_id_to_relpath(utt_id: str) -> Path:
    """``"1272-128104-0000"`` -> ``Path("1272/128104/1272-128104-0000")``."""
    spk, chap, _ = utt_id.split("-", 2)
    return Path(spk) / chap / utt_id


def extract_one(
    wav_path: Path,
    out_path: Path,
    sample_rate: int,
    frame_period_ms: float,
    f0_floor: float,
    f0_ceil: float,
    overwrite: bool,
) -> tuple[str, str]:
    if out_path.exists() and not overwrite:
        return wav_path.name, "skipped"
    try:
        wav, sr = sf.read(str(wav_path), dtype="float64", always_2d=False)
        if sr != sample_rate:
            return wav_path.name, f"error: sr {sr} != {sample_rate}"
        if wav.ndim > 1:
            wav = wav.mean(axis=-1)
        wav = np.ascontiguousarray(wav)

        f0_rough, t = pyworld.dio(
            wav, sample_rate,
            f0_floor=f0_floor, f0_ceil=f0_ceil,
            frame_period=frame_period_ms,
        )
        f0 = pyworld.stonemask(wav, f0_rough, t, sample_rate)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out_path), f0.astype(np.float32))
        return wav_path.name, "ok"
    except Exception as e:  # noqa: BLE001
        return wav_path.name, f"error: {type(e).__name__}: {e}"


def _worker(args: tuple) -> tuple[str, str]:
    return extract_one(*args)


def process_split(
    libri_root: Path,
    split: str,
    f0_root: Path,
    sample_rate: int,
    frame_period_ms: float,
    f0_floor: float,
    f0_ceil: float,
    workers: int,
    overwrite: bool,
    limit: int | None,
) -> None:
    audio_dir = libri_root / split
    out_dir = f0_root / split
    flac_files = sorted(audio_dir.rglob("*.flac"))
    if limit:
        flac_files = flac_files[:limit]
    if not flac_files:
        print(f"[warn] no .flac files in {audio_dir}")
        return

    print(
        f"\n=== Split {split} ({len(flac_files)} files) ===\n"
        f"  audio:      {audio_dir}\n"
        f"  output:     {out_dir}\n"
        f"  workers:    {workers}\n"
        f"  hop:        {frame_period_ms} ms (frame rate "
        f"{1000.0 / frame_period_ms:.0f} Hz)"
    )

    tasks = []
    for f in flac_files:
        utt_id = f.stem
        out = out_dir / utt_id_to_relpath(utt_id).with_suffix(".npy")
        tasks.append(
            (f, out, sample_rate, frame_period_ms, f0_floor, f0_ceil, overwrite)
        )

    n_ok = n_skip = n_err = 0
    errors: list[tuple[str, str]] = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        for name, status in tqdm(
            pool.imap_unordered(_worker, tasks), total=len(tasks), desc=f"f0 {split}"
        ):
            if status == "ok":
                n_ok += 1
            elif status == "skipped":
                n_skip += 1
            else:
                n_err += 1
                errors.append((name, status))

    print(f"{split}: ok={n_ok}, skipped={n_skip}, error={n_err}")
    if errors:
        print("Errors:")
        for name, status in errors[:10]:
            print(f"  {name}: {status}")
        if len(errors) > 10:
            print(f"  ... +{len(errors) - 10} more")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--librispeech-root", type=Path, default=DEFAULT_LIBRI_ROOT)
    ap.add_argument("--f0-root", type=Path, default=DEFAULT_F0_ROOT)
    ap.add_argument(
        "--splits",
        nargs="+",
        default=["dev-clean", "test-clean", "train-clean-100"],
    )
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--frame-period-ms", type=float, default=DEFAULT_FRAME_PERIOD_MS)
    ap.add_argument("--f0-floor", type=float, default=DEFAULT_F0_FLOOR)
    ap.add_argument("--f0-ceil", type=float, default=DEFAULT_F0_CEIL)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N files per split (smoke test).")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    for split in args.splits:
        process_split(
            libri_root=args.librispeech_root,
            split=split,
            f0_root=args.f0_root,
            sample_rate=args.sample_rate,
            frame_period_ms=args.frame_period_ms,
            f0_floor=args.f0_floor,
            f0_ceil=args.f0_ceil,
            workers=args.workers,
            overwrite=args.overwrite,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()
