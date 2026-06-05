"""Download and extract Google Speech Commands v0.01 for the KS probe.

SUPERB Keyword Spotting uses v0.01 (~1.4 GB tar.gz, 30 keyword folders +
``_background_noise_``), so this is the version pulled. v0.02 is *not*
backwards-compatible with the SUPERB protocol (extra keywords) and is not
used here.

Usage:
    uv run python -m src.data.prepare_speech_commands \
        --output-dir /workspace/i_tatsuro/data/SpeechCommands
"""

from __future__ import annotations

import argparse
import shutil
import tarfile
import urllib.request
from pathlib import Path

URL = "http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz"
DEFAULT_OUTPUT = Path("/workspace/i_tatsuro/data/SpeechCommands")


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[skip download] {dest} already exists ({dest.stat().st_size / 1e9:.2f} GB)")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading {url} -> {dest}")
    with urllib.request.urlopen(url) as r, tmp.open("wb") as f:
        shutil.copyfileobj(r, f)
    tmp.rename(dest)
    print(f"  done: {dest.stat().st_size / 1e9:.2f} GB")


def _extract(tar_path: Path, dest: Path) -> None:
    if (dest / "validation_list.txt").exists():
        print(f"[skip extract] looks already extracted under {dest}")
        return
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {tar_path} -> {dest}")
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(dest)  # noqa: S202 — trusted upstream
    print("  done")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                    help="Directory to download tarball and extract into")
    ap.add_argument("--keep-tar", action="store_true",
                    help="Keep the tar.gz after extraction (default: remove)")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tar_path = args.output_dir / "speech_commands_v0.01.tar.gz"

    _download(URL, tar_path)
    _extract(tar_path, args.output_dir)

    # Sanity check
    expected = ["validation_list.txt", "testing_list.txt", "_background_noise_", "yes", "no"]
    missing = [name for name in expected if not (args.output_dir / name).exists()]
    if missing:
        print(f"WARNING: expected entries missing after extraction: {missing}")
    else:
        print(f"OK. Speech Commands ready at {args.output_dir}")

    if not args.keep_tar and tar_path.exists():
        tar_path.unlink()
        print(f"Removed {tar_path}")


if __name__ == "__main__":
    main()
