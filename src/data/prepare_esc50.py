"""Resample ESC-50 to 16 kHz mono wav, matching the POP909-rendered layout.

ESC-50 ships as 2000 mono ``.wav`` files at 44.1 kHz / 5 s. Our pretraining
pipeline expects 16 kHz mono PCM_16; this script just resamples.

Output: ``--output-dir`` populated with the same filenames at 16 kHz.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as taF
from tqdm import tqdm

DEFAULT_INPUT = Path("/workspace/i_tatsuro/data/ESC-50/ESC-50-master/audio")
DEFAULT_OUTPUT = Path("/workspace/i_tatsuro/data/ESC-50-rendered")
DEFAULT_SR = 16000


def resample_one(
    in_path: Path, out_path: Path, target_sr: int, overwrite: bool
) -> tuple[str, str]:
    if out_path.exists() and not overwrite:
        return in_path.name, "skipped"
    try:
        wav, sr = sf.read(str(in_path), dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=-1)
        if sr != target_sr:
            wav_t = torch.from_numpy(wav)
            wav_t = taF.resample(wav_t, sr, target_sr)
            wav = wav_t.numpy()
        peak = float(np.abs(wav).max())
        if peak > 0:
            wav = wav / peak * 0.95
        pcm16 = np.clip(wav * 32767.0, -32768, 32767).astype(np.int16)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), pcm16, target_sr, subtype="PCM_16")
        return in_path.name, "ok"
    except Exception as e:  # noqa: BLE001
        return in_path.name, f"error: {type(e).__name__}: {e}"


def _worker(args: tuple) -> tuple[str, str]:
    return resample_one(*args)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--sample-rate", type=int, default=DEFAULT_SR)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    wavs = sorted(args.input_dir.glob("*.wav"))
    if args.limit:
        wavs = wavs[: args.limit]
    print(
        f"Resampling {len(wavs)} ESC-50 files\n"
        f"  input:  {args.input_dir}\n"
        f"  output: {args.output_dir}\n"
        f"  target_sr: {args.sample_rate}\n"
        f"  workers:   {args.workers}"
    )

    tasks = [
        (w, args.output_dir / w.name, args.sample_rate, args.overwrite)
        for w in wavs
    ]

    n_ok = n_skip = n_err = 0
    errors: list[tuple[str, str]] = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        for name, status in tqdm(
            pool.imap_unordered(_worker, tasks), total=len(tasks), desc="esc50"
        ):
            if status == "ok":
                n_ok += 1
            elif status == "skipped":
                n_skip += 1
            else:
                n_err += 1
                errors.append((name, status))

    print(f"\nDone: ok={n_ok}, skipped={n_skip}, error={n_err}")
    if errors:
        print("Errors:")
        for name, status in errors[:10]:
            print(f"  {name}: {status}")


if __name__ == "__main__":
    main()
