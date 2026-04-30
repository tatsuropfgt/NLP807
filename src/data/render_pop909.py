"""Render POP909 MIDI files to wav, optionally with an ablation transform.

Loads each .mid in --input-dir, applies the chosen --transform (one of
intact, pitch_strip, rhythm_strip, both_strip), synthesizes via FluidSynth +
a GM SoundFont, and writes mono 16-bit PCM wav files at --sample-rate to
--output-dir. The default --output-dir is derived from --transform.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf
from tqdm import tqdm

from src.data.midi_ops import TRANSFORMS

DEFAULT_INPUT = Path("/workspace/i_tatsuro/data/POP909-Dataset/align_mid")
DEFAULT_OUTPUT_BASE = Path("/workspace/i_tatsuro/data/POP909-rendered")
DEFAULT_SOUNDFONT = Path("/usr/share/sounds/sf2/FluidR3_GM.sf2")
DEFAULT_SR = 16000


def render_one(
    midi_path: Path,
    out_path: Path,
    soundfont: Path,
    sr: int,
    transform: str,
) -> None:
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    pm = TRANSFORMS[transform](pm)
    audio = pm.fluidsynth(fs=sr, sf2_path=str(soundfont))
    peak = float(np.abs(audio).max())
    if peak > 0:
        audio = audio / peak * 0.95
    pcm16 = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), pcm16, sr, subtype="PCM_16")


def _worker(args: tuple[Path, Path, Path, int, str, bool]) -> tuple[str, str]:
    midi_path, out_path, soundfont, sr, transform, overwrite = args
    try:
        if out_path.exists() and not overwrite:
            return midi_path.name, "skipped"
        render_one(midi_path, out_path, soundfont, sr, transform)
        return midi_path.name, "ok"
    except Exception as e:  # noqa: BLE001
        return midi_path.name, f"error: {type(e).__name__}: {e}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--transform",
        type=str,
        default="intact",
        choices=list(TRANSFORMS.keys()),
        help="MIDI transform to apply before synthesis.",
    )
    ap.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Defaults to {DEFAULT_OUTPUT_BASE}/<transform>.",
    )
    ap.add_argument("--soundfont", type=Path, default=DEFAULT_SOUNDFONT)
    ap.add_argument("--sample-rate", type=int, default=DEFAULT_SR)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument(
        "--limit", type=int, default=None, help="Render only first N files (for smoke test)."
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT_BASE / args.transform

    if not args.soundfont.exists():
        raise FileNotFoundError(f"SoundFont not found: {args.soundfont}")
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {args.input_dir}")

    midi_files = sorted(args.input_dir.glob("*.mid"))
    if args.limit:
        midi_files = midi_files[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Rendering {len(midi_files)} MIDI files\n"
        f"  transform: {args.transform}\n"
        f"  input:     {args.input_dir}\n"
        f"  output:    {args.output_dir}\n"
        f"  sf2:       {args.soundfont}\n"
        f"  sr:        {args.sample_rate} Hz\n"
        f"  workers:   {args.workers}"
    )

    tasks = [
        (
            m,
            args.output_dir / f"{m.stem}.wav",
            args.soundfont,
            args.sample_rate,
            args.transform,
            args.overwrite,
        )
        for m in midi_files
    ]

    n_ok = n_skip = n_err = 0
    errors: list[tuple[str, str]] = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        for name, status in tqdm(
            pool.imap_unordered(_worker, tasks), total=len(tasks), desc="render"
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
        for name, status in errors[:20]:
            print(f"  {name}: {status}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")


if __name__ == "__main__":
    main()
