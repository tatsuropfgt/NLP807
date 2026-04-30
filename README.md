# music2speech

Investigate whether self-supervised pretraining on musical audio helps representation learning for speech tasks.

## Design

### Pretraining conditions

POP909 is manipulated at the MIDI stage and rendered through a shared synthesis pipeline to 16 kHz mono wav (so acoustic characteristics are matched).

| ID | Condition | Operation |
|---|---|---|
| A | intact | Original MIDI |
| B | pitch_strip | Per-track fixed pitch (MELODY=C5, BRIDGE=C4, PIANO=C3). velocity / onset / duration preserved |
| C | rhythm_strip | All onsets floor-quantized to the containing beat, duration uniformly set to 1 beat, pitch sequence preserved |
| D | both_strip | B + C |
| E | ESC-50 | Environmental sounds 2000×5s (~2.8h), control for music-specificity |
| F | random init | No pretraining |

### Downstream tasks

| Hypothesis | Task | Label | Metric |
|---|---|---|---|
| Pitch transfer | F0 tracking | LibriSpeech audio + pyworld-extracted F0 | RMSE [cents] (↓) |
| Rhythm transfer | Phoneme boundary | Binarized "frame where the phoneme changes" from TextGrid | F1 (↑) |
| Integrated capability | Phoneme classification | TextGrid + CMU 40 phonemes (incl. SIL) | frame accuracy (↑) |

## Data layout

| Data | Location |
|---|---|
| POP909 MIDI | `/workspace/i_tatsuro/data/POP909-Dataset/align_mid/` (`001.mid`..., 896 files with gaps) |
| ESC-50 raw audio | `/workspace/i_tatsuro/data/ESC-50/ESC-50-master/audio/` |
| LibriSpeech | `/workspace/i_tatsuro/data/LibriSpeech/LibriSpeech/{train-clean-100,dev-clean,test-clean}/` |
| MFA TextGrid | `/workspace/i_tatsuro/data/LibriSpeech/alignments/LibriSpeech/<split>/<spk>/<chap>/<utt>.TextGrid` |

The time-signature meta inside POP909 MIDI is unreliable. The list of triple-meter tracks is hardcoded in [src/data/midi_ops.py](src/data/midi_ops.py) and [src/data/generate_examples.py](src/data/generate_examples.py).
`align_mid` uses a version of the original [MIDI data](https://github.com/music-x-lab/POP909-Dataset) that we lightly preprocessed ourselves. A processed copy is committed under [align_mid/](align_mid/) for reproducibility.

## Setup

```bash
# dependencies
apt-get install -y fluidsynth fluid-soundfont-gm libfluidsynth3
uv sync
```

Python ≥3.11. SoundFont expected at `/usr/share/sounds/sf2/FluidR3_GM.sf2`.

## Replication

Every step has skip-existing built in, so a re-run picks up where it left off.

### 1. Data preparation

```bash
# Render POP909 to wav for each of the 4 transforms
for T in intact pitch_strip rhythm_strip both_strip; do
  uv run python -m src.data.render_pop909 --transform "$T"
done

# Resample ESC-50 to 16 kHz mono (assumes the zip is already extracted under /workspace/.../ESC-50/)
uv run python -m src.data.prepare_esc50 --workers 8

# LibriSpeech audio + MFA TextGrid alignments (TextGrids are downloaded manually from Google Drive)
# Adjust paths as needed
mkdir -p /workspace/i_tatsuro/data/LibriSpeech
cd /workspace/i_tatsuro/data/LibriSpeech
for s in train-clean-100 dev-clean test-clean; do
  wget -c https://www.openslr.org/resources/12/${s}.tar.gz
  tar xzf ${s}.tar.gz --no-same-owner
done

# Extract F0 with pyworld (10ms hop)
uv run python -m src.data.extract_f0 --workers 8
```

### 2. Full pipeline

3 seeds × (5 pretrain + 18 probe) in one go:

```bash
for SEED in 42 43 44; do
  bash scripts/run_seed_full.sh "$SEED"
done
# Output: runs/seed42/<cond>/{encoder.pt, probe_*/best_metrics.json}, ...
```

~5–6 hours per seed on a single GPU.

### Optional. Individual runs (for debugging)

#### Pretrain

```bash
uv run python -m src.pretrain.train \
  --data-dir /workspace/i_tatsuro/data/POP909-rendered/intact \
  --output-dir runs/intact \
  --max-steps 20000 --batch-size 32 \
  --val-every 500 --patience 5 \
  --split-seed 42 \
  --wandb --wandb-run-name pretrain_intact_v1
```

`encoder.pt` is automatically promoted to the best-val weights.

#### Probes (3 types)

```bash
LIBRI=/workspace/i_tatsuro/data/LibriSpeech/LibriSpeech
ALIGN=/workspace/i_tatsuro/data/LibriSpeech/alignments/LibriSpeech
F0=/workspace/i_tatsuro/data/LibriSpeech/f0
CKPT=runs/intact/encoder.pt   # for random init, omit --encoder-ckpt

uv run python -m src.eval.train_probe \
  --encoder-ckpt $CKPT --librispeech-root $LIBRI --alignments-root $ALIGN \
  --output-dir runs/intact/probe_phone --max-steps 10000 --batch-size 8

uv run python -m src.eval.train_boundary_probe \
  --encoder-ckpt $CKPT --librispeech-root $LIBRI --alignments-root $ALIGN \
  --output-dir runs/intact/probe_boundary --max-steps 10000 --batch-size 8
# Boundary frames are only ~10%, so --pos-weight=9.0 is the default

uv run python -m src.eval.train_f0_probe \
  --encoder-ckpt $CKPT --librispeech-root $LIBRI --f0-root $F0 \
  --output-dir runs/intact/probe_f0 --max-steps 10000 --batch-size 8
```

#### Bulk runner

```bash
# All conditions × all tasks (skip-existing built in)
bash scripts/run_all_probes.sh
# Subset
CONDITIONS="esc50 random_init" TASKS="boundary f0" bash scripts/run_all_probes.sh
```

### 4. Aggregation

```bash
# Re-execute the notebook to regenerate tables + figures from every best_metrics.json under runs/
uv run jupyter nbconvert --to notebook --execute notebooks/results.ipynb \
  --output results.ipynb --ExecutePreprocessor.kernel_name=music2speech
# → notebooks/figures/probe_comparison.png is updated
```

## Repository layout

```
src/
├── data/
│   ├── midi_ops.py              # pitch / rhythm / both strip transforms
│   ├── render_pop909.py         # POP909 MIDI -> wav (any transform)
│   ├── prepare_esc50.py         # Resample ESC-50 to 16 kHz mono
│   ├── extract_f0.py            # F0 extraction for all LibriSpeech utts via pyworld
│   ├── generate_examples.py     # Demo MIDI/WAV for the talk (1–2 bars)
│   ├── dataset.py               # WavFolderDataset (for pretrain)
│   ├── alignments.py            # MFA TextGrid phoneme-alignment parser
│   └── librispeech.py           # LibriSpeech + frame-level phoneme/F0
├── models/
│   ├── encoder.py               # mel + Transformer encoder
│   └── msm.py                   # Masked Spectrogram Modeling
├── pretrain/train.py            # SSL pretrain
└── eval/
    ├── probe.py                 # frozen encoder + linear head
    ├── train_probe.py           # phoneme classification
    ├── train_boundary_probe.py  # phoneme boundary detection
    └── train_f0_probe.py        # F0 tracking (regression)

scripts/
├── run_seed_full.sh             # Full pipeline for 1 seed (5 pretrain + 18 probe)
└── run_all_probes.sh            # All conditions × all probes (skip-existing built in)

examples/                        # 1–2 bar samples for the talk
notebooks/
├── results.ipynb                # cross-seed aggregation
└── figures/probe_comparison.png
```

## References

- POP909: https://github.com/music-x-lab/POP909-Dataset
- LibriSpeech: https://www.openslr.org/12/
- CorentinJ/librispeech-alignments: https://github.com/CorentinJ/librispeech-alignments (TextGrid version used)
- ESC-50: https://github.com/karolpiczak/ESC-50
- WORLD (pyworld): https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder
