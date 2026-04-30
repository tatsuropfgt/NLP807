"""Phoneme alignments from MFA TextGrid files.

We use the **raw MFA output** (TextGrid format) from CorentinJ's release of
``librispeech-alignments``. One ``*.TextGrid`` file per utterance, mirroring
LibriSpeech's directory layout::

    alignments/LibriSpeech/{split}/{spk}/{chapter}/{spk}-{chapter}-{utt}.TextGrid

A TextGrid contains two ``IntervalTier``s: ``"words"`` and ``"phones"``. Each
phone interval has ``xmin``, ``xmax`` (seconds), and a ``text`` field with a
CMU phone with optional stress (``AH1``, ``ER0``, ``T``, ...). Silence is
``"sil"``; spoken noise is ``"spn"``. Anything we don't recognise (including
empty text) is mapped to ``SIL``.

This module provides:
- A flat 40-class CMU phoneme inventory (``ARPA_PHONES``) with ``SIL`` at id 0.
- ``parse_textgrid_phones`` -> list of ``PhoneSegment`` for one utterance.
- ``segments_to_frame_labels`` -> per-frame integer labels.
- ``alignment_path_for_utt`` -> derive the TextGrid path from an utt_id.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# 39 CMU phones, no stress, plus SIL at id 0.
ARPA_PHONES: tuple[str, ...] = (
    "SIL",
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH",
    "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K",
    "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH",
    "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH",
)
PHONE_TO_ID: dict[str, int] = {p: i for i, p in enumerate(ARPA_PHONES)}
N_PHONES: int = len(ARPA_PHONES)  # 40
SILENCE_ID: int = 0

_STRESS_RE = re.compile(r"\d+$")
# Match a single MFA interval entry inside the phones tier.
_INTERVAL_RE = re.compile(
    r'intervals\s*\[\d+\]\s*:\s*'
    r'xmin\s*=\s*([0-9.eE+\-]+)\s*'
    r'xmax\s*=\s*([0-9.eE+\-]+)\s*'
    r'text\s*=\s*"([^"]*)"',
    re.DOTALL,
)
_ITEM_SPLIT_RE = re.compile(r"item\s*\[\d+\]\s*:")
_NAME_RE = re.compile(r'name\s*=\s*"([^"]*)"')


@dataclass
class PhoneSegment:
    phone: str  # ARPA, no stress; ``SIL`` for silence / unknown.
    start: float  # seconds
    end: float    # seconds


def strip_stress(phone: str) -> str:
    """``'AH1'`` -> ``'AH'``; non-vowels pass through unchanged."""
    return _STRESS_RE.sub("", phone)


def _normalize_phone(text: str) -> str:
    text = text.strip()
    if not text:
        return "SIL"
    # MFA uses lowercase 'sil' / 'sp' / 'spn'.
    upper = text.upper()
    if upper in {"SIL", "SP", "SPN"}:
        return "SIL"
    cleaned = strip_stress(upper)
    return cleaned if cleaned in PHONE_TO_ID else "SIL"


def parse_textgrid_phones(path: Path) -> list[PhoneSegment]:
    """Parse the 'phones' tier of one MFA TextGrid file.

    Returns the list of ``PhoneSegment``s in time order. If the file is
    missing or contains no phones tier, returns ``[]``.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")

    # Split into items and find the one whose name == "phones".
    pieces = _ITEM_SPLIT_RE.split(text)
    phones_block: str | None = None
    for piece in pieces:
        m = _NAME_RE.search(piece)
        if m and m.group(1) == "phones":
            phones_block = piece
            break
    if phones_block is None:
        return []

    segments: list[PhoneSegment] = []
    for m in _INTERVAL_RE.finditer(phones_block):
        try:
            xmin = float(m.group(1))
            xmax = float(m.group(2))
        except ValueError:
            continue
        ph = _normalize_phone(m.group(3))
        segments.append(PhoneSegment(phone=ph, start=xmin, end=xmax))
    return segments


def alignment_path_for_utt(alignments_split_dir: Path, utt_id: str) -> Path:
    """Derive the TextGrid path for an utterance id like ``"1272-128104-0000"``.

    ``alignments_split_dir`` is a per-split directory, e.g.
    ``.../alignments/LibriSpeech/dev-clean``.
    """
    parts = utt_id.split("-")
    if len(parts) < 3:
        raise ValueError(f"unexpected utt_id format: {utt_id!r}")
    spk, chap = parts[0], parts[1]
    return alignments_split_dir / spk / chap / f"{utt_id}.TextGrid"


def segments_to_frame_labels(
    segments: list[PhoneSegment],
    n_frames: int,
    frame_hop_seconds: float,
    silence_id: int = SILENCE_ID,
) -> list[int]:
    """Quantize phone segments to frame-level integer labels.

    Frames not covered by any segment receive ``silence_id``. Later segments
    overwrite earlier ones in the rare case of overlap (shouldn't happen in
    well-formed MFA output).
    """
    labels = [silence_id] * n_frames
    for seg in segments:
        ph_id = PHONE_TO_ID.get(seg.phone, silence_id)
        start_f = max(0, int(seg.start / frame_hop_seconds))
        end_f = min(n_frames, int(round(seg.end / frame_hop_seconds)))
        for f in range(start_f, end_f):
            labels[f] = ph_id
    return labels
