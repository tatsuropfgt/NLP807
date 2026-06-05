"""Concatenate, extract and verify VoxCeleb1 for the SID probe.

VoxCeleb1 is distributed as four split parts of the dev set
(``vox1_dev_wav_part{aa,ab,ac,ad}``) plus a single test zip
(``vox1_test_wav.zip``). This script:

1. ``cat``s the four parts into ``vox1_dev_wav.zip``.
2. Deletes the four parts (their content is now in the cat'd zip).
3. Unzips dev into ``--output-dir/wav/``, then deletes the dev zip.
4. Unzips test into ``--output-dir/wav/``, then deletes the test zip.
5. Downloads the SUPERB-standard ``iden_split.txt`` if it's not already
   present alongside the parts.

Each large intermediate (parts, dev zip, test zip) is removed immediately
after the next step has consumed it, keeping peak disk close to the final
``wav/`` size (~28 GB) instead of ~92 GB. Use ``--keep-parts`` /
``--keep-zips`` to override.

The wav files inside are 16 kHz mono already so no resampling is performed.
Extraction is idempotent: files already present at the correct size are
skipped, so a re-run after a disk-space failure resumes cleanly.

Assumes the parts and test zip have been placed at ``--input-dir`` by hand
(they require registration on the VGG site to obtain).

Usage:
    uv run python -m src.data.prepare_voxceleb \
        --input-dir /workspace/i_tatsuro/data/VoxCeleb1 \
        --output-dir /workspace/i_tatsuro/data/VoxCeleb1
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

IDEN_SPLIT_URL = "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/iden_split.txt"
DEV_PARTS = ("vox1_dev_wav_partaa", "vox1_dev_wav_partab",
             "vox1_dev_wav_partac", "vox1_dev_wav_partad")
TEST_ZIP = "vox1_test_wav.zip"
DEV_ZIP = "vox1_dev_wav.zip"


def _concat_parts(input_dir: Path, dev_zip: Path, keep_parts: bool) -> None:
    """Concatenate the 4 dev parts into ``dev_zip``. Optionally delete parts.

    Deleting the parts frees ~32 GB; ``cat`` has already faithfully copied
    every byte into ``dev_zip``, so they're redundant from here on. Pass
    ``keep_parts=True`` to retain them.
    """
    parts = [input_dir / p for p in DEV_PARTS]
    if dev_zip.exists():
        print(f"[skip concat] {dev_zip} already exists ({dev_zip.stat().st_size / 1e9:.2f} GB)")
    else:
        missing = [p for p in parts if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"missing dev parts under {input_dir}: {[p.name for p in missing]}"
            )
        print(f"Concatenating 4 parts -> {dev_zip}")
        # Use ``cat`` so we don't materialise the parts in Python memory.
        with dev_zip.open("wb") as out:
            subprocess.run(["cat", *[str(p) for p in parts]], stdout=out, check=True)
        print(f"  done: {dev_zip.stat().st_size / 1e9:.2f} GB")

    if not keep_parts:
        for p in parts:
            if p.exists():
                p.unlink()
                print(f"Removed {p}")


def _extract_zip(zip_path: Path, output_dir: Path) -> None:
    """Extract ``zip_path`` into ``output_dir``, skipping files already present
    at the correct size. This makes the script resumable after a partial run
    (e.g. when the previous attempt died from a disk-space error)."""
    print(f"Extracting {zip_path} -> {output_dir}")
    extracted = 0
    skipped = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            target = output_dir / info.filename
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.exists() and target.stat().st_size == info.file_size:
                skipped += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted += 1
    print(f"  done: extracted={extracted}, skipped (already present)={skipped}")


def _download_iden_split(input_dir: Path) -> None:
    dest = input_dir / "iden_split.txt"
    if dest.exists():
        print(f"[skip iden_split] {dest} already exists")
        return
    print(f"Downloading {IDEN_SPLIT_URL} -> {dest}")
    with urllib.request.urlopen(IDEN_SPLIT_URL) as r, dest.open("wb") as f:
        shutil.copyfileobj(r, f)
    print("  done")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", type=Path, required=True,
                    help="Directory holding the uploaded dev parts + test zip")
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="Where to put the extracted wav/ tree")
    ap.add_argument("--keep-parts", action="store_true",
                    help="Keep the 4 dev parts after concatenation (default: delete; "
                         "they are redundant once cat'd)")
    ap.add_argument("--keep-zips", action="store_true",
                    help="Keep the concatenated dev zip and test zip after extraction "
                         "(default: remove to save disk)")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    dev_zip = args.input_dir / DEV_ZIP
    test_zip = args.input_dir / TEST_ZIP
    wav_root = args.output_dir / "wav"

    # 1) Concat dev parts -> dev zip (optionally delete parts)
    _concat_parts(args.input_dir, dev_zip, keep_parts=args.keep_parts)

    # 2) Extract dev, then delete dev zip immediately to keep peak disk low
    _extract_zip(dev_zip, args.output_dir)
    if not args.keep_zips and dev_zip.exists():
        dev_zip.unlink()
        print(f"Removed {dev_zip}")

    # 3) Extract test, then delete test zip
    if test_zip.exists():
        _extract_zip(test_zip, args.output_dir)
        if not args.keep_zips:
            test_zip.unlink()
            print(f"Removed {test_zip}")
    else:
        print(f"WARNING: {test_zip} not found; SUPERB test split will be missing")

    # 4) Download iden_split.txt if needed
    _download_iden_split(args.input_dir)

    # Sanity check
    n_speakers = 0
    if wav_root.exists():
        n_speakers = sum(1 for p in wav_root.iterdir() if p.is_dir() and p.name.startswith("id"))
    print(f"VoxCeleb1 extracted to {wav_root}: {n_speakers} speaker dirs")
    if n_speakers and n_speakers != 1251:
        print(f"WARNING: expected 1251 speakers in VoxCeleb1, found {n_speakers}")


if __name__ == "__main__":
    main()
