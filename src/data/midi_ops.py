"""MIDI manipulation for ablation conditions.

Three transforms remove a specific information dimension from the source MIDI
while keeping the rest:

- ``strip_pitch``  : per-track fixed pitches (MELODY=C5, BRIDGE=C4, PIANO=C3).
                     Velocity, duration, onset preserved. Within-track pitch
                     contour and chord harmony are erased.
- ``strip_rhythm`` : every onset snapped to the nearest beat (via
                     :py:meth:`pretty_midi.PrettyMIDI.get_beats`); duration set
                     to one beat. Pitch / velocity preserved. Sub-beat rhythm,
                     syncopation, and durational variation are erased.
- ``strip_both``   : both transforms, in that order.

After every transform we deduplicate notes with the same ``(pitch, start)``
within a track (in particular: chord notes collapsed to one note in
``strip_pitch``; multiple sub-beat events collapsed to one in
``strip_rhythm``). Cross-track collisions stay distinct because each track
has either a different fixed pitch (``strip_pitch``) or, in ``strip_rhythm``,
the original pitches.

POP909 has the consistent layout: every song has 3 tracks named
``MELODY`` / ``BRIDGE`` / ``PIANO``, all using program 0 (Acoustic Grand
Piano). Without per-track pitch differentiation in ``strip_pitch``, all
tracks would acoustically merge — the per-track fixed pitches preserve track
separation as a coarse pitch cue (3 distinct pitches; the within-track pitch
dynamics that carry melody/harmony are still gone).
"""

from __future__ import annotations

import bisect
from copy import deepcopy
from typing import Iterable

import pretty_midi

# Per-track fixed pitches for `strip_pitch`. Chosen so that MELODY > BRIDGE >
# PIANO matches typical voicing in pop music.
TRACK_PITCH_MAP: dict[str, int] = {
    "MELODY": 72,  # C5
    "BRIDGE": 60,  # C4
    "PIANO": 48,   # C3
}
DEFAULT_PITCH: int = 60  # for tracks not in TRACK_PITCH_MAP


def _dedup_notes(
    notes: Iterable[pretty_midi.Note], eps: float = 1e-4
) -> list[pretty_midi.Note]:
    """Drop notes that share ``(pitch, start)`` within ``eps`` seconds.

    Among duplicates we keep the entry with the longest duration.
    """
    sorted_notes = sorted(notes, key=lambda n: (n.pitch, n.start, -(n.end - n.start)))
    out: list[pretty_midi.Note] = []
    for note in sorted_notes:
        if (
            out
            and out[-1].pitch == note.pitch
            and abs(out[-1].start - note.start) < eps
        ):
            # Duplicate; extend end if this one is longer.
            if note.end > out[-1].end:
                out[-1].end = note.end
            continue
        out.append(note)
    out.sort(key=lambda n: (n.start, n.pitch))
    return out


def strip_pitch(
    pm: pretty_midi.PrettyMIDI,
    pitch_map: dict[str, int] = TRACK_PITCH_MAP,
    dedup: bool = True,
) -> pretty_midi.PrettyMIDI:
    """Replace each track's note pitches with a single per-track fixed pitch.

    Velocity, onset, and duration are preserved. After substitution, chord
    notes (multiple notes at the same onset within a track) collapse to one.
    """
    pm = deepcopy(pm)
    for inst in pm.instruments:
        target = pitch_map.get(inst.name, DEFAULT_PITCH)
        for note in inst.notes:
            note.pitch = target
        if dedup:
            inst.notes = _dedup_notes(inst.notes)
    return pm


def _floor_beat_index(beats: list[float], t: float) -> int:
    """Return ``i`` such that ``beats[i] <= t < beats[i+1]`` (or 0 if ``t``
    is before the first beat, or ``len(beats) - 1`` if at/past the last)."""
    if not beats:
        raise ValueError("empty beats")
    idx = bisect.bisect_right(beats, t) - 1
    if idx < 0:
        return 0
    return idx


def strip_rhythm(
    pm: pretty_midi.PrettyMIDI,
    dedup: bool = True,
) -> pretty_midi.PrettyMIDI:
    """Snap every onset to the **start of its containing beat** (floor),
    then set duration to one beat.

    Beats come from ``pm.get_beats()`` which respects tempo changes. With
    floor quantization, a note starting at any time ``t`` in
    ``[beats[i], beats[i+1])`` moves to ``beats[i]``. This is semantically
    "all notes within beat i collapse onto beat i", and avoids notes
    crossing beat boundaries (which round-to-nearest would do for any onset
    after a beat midpoint). The last note's duration is extrapolated from
    the prior beat duration when it lands on the final beat.
    """
    pm = deepcopy(pm)
    beats = list(pm.get_beats())
    if len(beats) < 2:
        # Pathological MIDI without enough tempo info; leave as-is.
        return pm
    last_beat_dur = beats[-1] - beats[-2]

    for inst in pm.instruments:
        new_notes: list[pretty_midi.Note] = []
        for note in inst.notes:
            i = _floor_beat_index(beats, note.start)
            new_start = beats[i]
            new_end = beats[i + 1] if i + 1 < len(beats) else beats[-1] + last_beat_dur
            note.start = new_start
            note.end = new_end
            new_notes.append(note)
        if dedup:
            new_notes = _dedup_notes(new_notes)
        inst.notes = new_notes
    return pm


def strip_both(pm: pretty_midi.PrettyMIDI) -> pretty_midi.PrettyMIDI:
    """Apply ``strip_pitch`` then ``strip_rhythm``."""
    return strip_rhythm(strip_pitch(pm))


# Lookup used by render_pop909.py and example generators.
TRANSFORMS: dict[str, callable] = {
    "intact": lambda pm: pm,
    "pitch_strip": strip_pitch,
    "rhythm_strip": strip_rhythm,
    "both_strip": strip_both,
}
