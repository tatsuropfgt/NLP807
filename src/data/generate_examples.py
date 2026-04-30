"""Generate ~1-measure example clips (MIDI + WAV) for each MIDI transform.

Picks a small, fixed set of POP909 songs (one in 4/4, one in 3/4), trims the
first ``--measures`` measures of musical content, applies each transform
(intact / pitch_strip / rhythm_strip / both_strip), and writes both the
trimmed MIDI and a FluidSynth-rendered WAV to the output directory.

Output layout (default ``examples/``):

    examples/
        <song_id>/
            intact.mid       intact.wav
            pitch_strip.mid  pitch_strip.wav
            rhythm_strip.mid rhythm_strip.wav
            both_strip.mid   both_strip.wav
"""

from __future__ import annotations

import argparse
import bisect
from copy import deepcopy
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf

from src.data.midi_ops import TRANSFORMS

DEFAULT_SOUNDFONT = Path("/usr/share/sounds/sf2/FluidR3_GM.sf2")
DEFAULT_INPUT = Path("/workspace/i_tatsuro/data/POP909-Dataset/align_mid")
# A spread of POP909 songs: a few 4/4 of different density / tempo, plus
# two 3/4 songs from POP909's TIME_SIGNATURE_IN_THREE list. (POP909's MIDI
# time-signature metadata is unreliable, so we rely on the README's explicit
# list instead of pm.time_signature_changes.)
DEFAULT_SONGS = ("001", "042", "200", "500", "062", "107")

POP909_THREE_FOUR: frozenset[int] = frozenset({
    34, 62, 102, 107, 152, 173, 176, 203, 215, 231,
    254, 280, 307, 328, 369, 584, 592, 653, 654, 662,
    744, 749, 756, 770, 799, 843, 869, 872, 887,
})


def beats_per_measure(song_id: str) -> int:
    """3 if the song id is in POP909's known 3/4 list, else 4."""
    n = int(song_id.lstrip("0") or "0")
    return 3 if n in POP909_THREE_FOUR else 4


def first_content_beat_index(pm: pretty_midi.PrettyMIDI, beats: list[float]) -> int:
    """Index of the beat at or just before the earliest note onset."""
    earliest = float("inf")
    for inst in pm.instruments:
        for note in inst.notes:
            earliest = min(earliest, float(note.start))
    if earliest == float("inf"):
        return 0
    # bisect_right - 1 returns the last beat at or before `earliest`, so a
    # note that starts exactly on a beat resolves to that beat (not the one
    # before it).
    idx = bisect.bisect_right(beats, earliest) - 1
    return max(0, min(idx, len(beats) - 1))


def snap_to_downbeat(start_beat: int, numerator: int) -> int:
    """Round ``start_beat`` UP to the next measure boundary (downbeat).

    POP909 songs commonly start with an anacrusis on the last beat or two
    of an implicit "measure 0", which would make our trim start mid-measure
    if we used ``start_beat`` directly. Rounding up gives a clean example
    that begins on a downbeat (at the cost of dropping any anacrusis).
    """
    remainder = start_beat % numerator
    if remainder == 0:
        return start_beat
    return start_beat + (numerator - remainder)


def tempo_at(pm: pretty_midi.PrettyMIDI, t: float) -> float:
    """Tempo (BPM) in effect at time ``t``. Defaults to 120 if unknown."""
    times, tempi = pm.get_tempo_changes()
    if len(tempi) == 0:
        return 120.0
    # Latest tempo change at or before t.
    idx = max(0, int(np.searchsorted(times, t, side="right")) - 1)
    return float(tempi[idx])


def trim_window(
    pm: pretty_midi.PrettyMIDI, t_start: float, t_end: float
) -> pretty_midi.PrettyMIDI:
    """Keep only notes in ``[t_start, t_end)``, shifting the start to 0.

    The original tempo at ``t_start`` is preserved so that downstream beat
    quantization (``strip_rhythm``) still operates on the song's real beat
    grid rather than PrettyMIDI's default 120 BPM.
    """
    out = pretty_midi.PrettyMIDI(initial_tempo=tempo_at(pm, t_start))
    # Preserve the first applicable time signature.
    for ts in pm.time_signature_changes:
        if ts.time <= t_start:
            out.time_signature_changes.append(
                pretty_midi.TimeSignature(
                    numerator=ts.numerator,
                    denominator=ts.denominator,
                    time=0.0,
                )
            )
            break
    for inst in pm.instruments:
        new_inst = pretty_midi.Instrument(
            program=inst.program, is_drum=inst.is_drum, name=inst.name
        )
        for note in inst.notes:
            if t_start <= note.start < t_end:
                new_inst.notes.append(
                    pretty_midi.Note(
                        velocity=note.velocity,
                        pitch=note.pitch,
                        start=note.start - t_start,
                        end=min(note.end, t_end) - t_start,
                    )
                )
        out.instruments.append(new_inst)
    return out


def render_audio(pm: pretty_midi.PrettyMIDI, soundfont: Path, sr: int) -> np.ndarray:
    audio = pm.fluidsynth(fs=sr, sf2_path=str(soundfont))
    peak = float(np.abs(audio).max())
    if peak > 0:
        audio = audio / peak * 0.95
    return audio.astype(np.float32)


def write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    pcm16 = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), pcm16, sr, subtype="PCM_16")




def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT,
        help="Directory containing POP909 *.mid files.",
    )
    ap.add_argument(
        "--songs",
        type=str,
        nargs="+",
        default=list(DEFAULT_SONGS),
        help="Song IDs (3-digit strings) to use.",
    )
    ap.add_argument(
        "--measures",
        type=int,
        default=1,
        help="Number of measures to extract per song.",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples"),
        help="Where to write the {song_id}/{transform}.{mid,wav} files.",
    )
    ap.add_argument("--soundfont", type=Path, default=DEFAULT_SOUNDFONT)
    ap.add_argument("--sample-rate", type=int, default=16000)
    args = ap.parse_args()

    if not args.soundfont.exists():
        raise FileNotFoundError(f"SoundFont not found: {args.soundfont}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for song_id in args.songs:
        midi_path = args.input_dir / f"{song_id}.mid"
        if not midi_path.exists():
            print(f"[warn] missing: {midi_path}")
            continue
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        beats = list(pm.get_beats())
        numerator = beats_per_measure(song_id)
        if len(beats) < numerator + 1:
            print(f"[warn] {song_id}: not enough beats")
            continue

        content_beat = first_content_beat_index(pm, beats)
        start_beat = snap_to_downbeat(content_beat, numerator)
        end_beat = start_beat + args.measures * numerator
        if end_beat >= len(beats):
            print(f"[warn] {song_id}: not enough beats after content start")
            continue
        t_start = beats[start_beat]
        t_end = beats[end_beat]
        print(
            f"=== {song_id}: {numerator}/4 time, "
            f"{args.measures} measure(s) "
            f"[t={t_start:.3f}s, {t_end:.3f}s] ({t_end - t_start:.2f}s) "
            f"(content_beat={content_beat} -> downbeat={start_beat}) ==="
        )

        # Trim first, then apply each transform on the trimmed window. This
        # makes the four outputs exactly aligned in time.
        trimmed = trim_window(pm, t_start, t_end)

        song_out = args.output_dir / song_id
        song_out.mkdir(parents=True, exist_ok=True)

        for name, transform in TRANSFORMS.items():
            transformed = transform(deepcopy(trimmed))
            mid_path = song_out / f"{name}.mid"
            wav_path = song_out / f"{name}.wav"
            transformed.write(str(mid_path))
            audio = render_audio(transformed, args.soundfont, args.sample_rate)
            write_wav(wav_path, audio, args.sample_rate)
            n_notes = sum(len(i.notes) for i in transformed.instruments)
            print(f"  {name:>13}: {n_notes:>3} notes, {len(audio) / args.sample_rate:.2f}s wav")


if __name__ == "__main__":
    main()
