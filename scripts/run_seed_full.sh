#!/usr/bin/env bash
# Run the entire pipeline (5 pretrains + 6 conditions x 5 probe tasks) for
# a single seed. Output goes to runs/seed<SEED>/<cond>/, leaving the
# existing default-seed runs at runs/<cond>/ untouched.
#
# Probe tasks:
#   phone, boundary, f0 — frame-level (LibriSpeech + MFA / F0)
#   ks, sid             — utterance-level (Speech Commands v1, VoxCeleb1)
#
# Skip-existing safe at every step (per-pretrain via summary.json, per-probe
# via best_metrics.json). Re-invoking will pick up where things stopped.
#
# Usage:
#   bash scripts/run_seed_full.sh <SEED>
#
# Example for two extra seeds split across two machines (overnight):
#   # On machine A
#   bash scripts/run_seed_full.sh 43
#   # On machine B
#   bash scripts/run_seed_full.sh 44

set -euo pipefail
cd "$(dirname "$0")/.."
unset VIRTUAL_ENV || true

SEED=${1:?Usage: $0 <SEED>}
SUFFIX="seed${SEED}"

POP909_RENDERED=/workspace/i_tatsuro/data/POP909-rendered
ESC50_RENDERED=/workspace/i_tatsuro/data/ESC-50-rendered
RUNS_BASE=/workspace/i_tatsuro/projects/music2speech/runs/${SUFFIX}

LIBRI_ROOT=/workspace/i_tatsuro/data/LibriSpeech/LibriSpeech
ALIGN_ROOT=/workspace/i_tatsuro/data/LibriSpeech/alignments/LibriSpeech
F0_ROOT=/workspace/i_tatsuro/data/LibriSpeech/f0
SC_ROOT=/workspace/i_tatsuro/data/SpeechCommands
VOX_WAV_DIR=/workspace/i_tatsuro/data/VoxCeleb1/wav
VOX_SPLIT=/workspace/i_tatsuro/data/VoxCeleb1/iden_split.txt

# (condition, render_dir) for the 5 pretrain conditions.
PRETRAIN_CONDS=(intact pitch_strip rhythm_strip both_strip esc50)
declare -A COND_DATA=(
    [intact]="${POP909_RENDERED}/intact"
    [pitch_strip]="${POP909_RENDERED}/pitch_strip"
    [rhythm_strip]="${POP909_RENDERED}/rhythm_strip"
    [both_strip]="${POP909_RENDERED}/both_strip"
    [esc50]="${ESC50_RENDERED}"
)

# All 6 conditions evaluated downstream (random_init = no encoder).
ALL_CONDS=(intact pitch_strip rhythm_strip both_strip esc50 random_init)
ALL_TASKS=(phone boundary f0 ks sid)

probe_module() {
    case "$1" in
        phone)    echo "src.eval.train_probe" ;;
        boundary) echo "src.eval.train_boundary_probe" ;;
        f0)       echo "src.eval.train_f0_probe" ;;
        ks)       echo "src.eval.train_ks_probe" ;;
        sid)      echo "src.eval.train_sid_probe" ;;
    esac
}

# Per-task data args. Frame-level probes share --librispeech-root + label root;
# utterance-level probes have their own dataset roots.
probe_data_args() {
    case "$1" in
        phone|boundary) echo "--librispeech-root ${LIBRI_ROOT} --alignments-root ${ALIGN_ROOT}" ;;
        f0)             echo "--librispeech-root ${LIBRI_ROOT} --f0-root ${F0_ROOT}" ;;
        ks)             echo "--data-root ${SC_ROOT}" ;;
        sid)            echo "--wav-dir ${VOX_WAV_DIR} --split-file ${VOX_SPLIT}" ;;
    esac
}

probe_batch_size() {
    # Frame-level probes use long utterances (~16 s capped) at bs=8;
    # utterance-level probes use 1–3 s crops and fit bs=32.
    case "$1" in
        phone|boundary|f0) echo 8 ;;
        ks|sid)            echo 32 ;;
    esac
}

mkdir -p "${RUNS_BASE}"

echo "########################################################"
echo "# Full pipeline for ${SUFFIX}"
echo "# Output base: ${RUNS_BASE}"
echo "########################################################"

# ---------------------------------------------------------------------------
# 1) Pretrain (5 conditions)
# ---------------------------------------------------------------------------
for COND in "${PRETRAIN_CONDS[@]}"; do
    DATA_DIR="${COND_DATA[$COND]}"
    PRETRAIN_OUT="${RUNS_BASE}/${COND}"

    if [ ! -d "${DATA_DIR}" ]; then
        echo "[error] missing render dir for ${COND}: ${DATA_DIR}"
        echo "        run src.data.render_pop909 / src.data.prepare_esc50 first"
        exit 1
    fi

    if [ -f "${PRETRAIN_OUT}/summary.json" ]; then
        echo "[skip pretrain ${COND}/${SUFFIX}] summary.json exists"
        continue
    fi
    if [ -d "${PRETRAIN_OUT}" ]; then
        echo "[clean pretrain ${COND}/${SUFFIX}] removing partial files"
        find "${PRETRAIN_OUT}" -maxdepth 1 -type f -delete
    fi

    echo
    echo "=== Pretrain ${COND} (${SUFFIX}) ==="
    uv run python -m src.pretrain.train \
        --data-dir "${DATA_DIR}" \
        --output-dir "${PRETRAIN_OUT}" \
        --max-steps 20000 \
        --batch-size 32 \
        --val-every 500 \
        --patience 5 \
        --split-seed "${SEED}" \
        --wandb --wandb-run-name "pretrain_${COND}_${SUFFIX}"
done

# ---------------------------------------------------------------------------
# 2) Probes (6 conditions x 3 tasks = 18 runs)
# ---------------------------------------------------------------------------
for COND in "${ALL_CONDS[@]}"; do
    if [ "${COND}" == "random_init" ]; then
        CKPT_ARG=""
    else
        CKPT="${RUNS_BASE}/${COND}/encoder.pt"
        if [ ! -f "${CKPT}" ]; then
            echo "[skip probes ${COND}/${SUFFIX}] no encoder.pt"
            continue
        fi
        CKPT_ARG="--encoder-ckpt ${CKPT}"
    fi

    for TASK in "${ALL_TASKS[@]}"; do
        OUT="${RUNS_BASE}/${COND}/probe_${TASK}"
        if [ -f "${OUT}/best_metrics.json" ]; then
            echo "[skip probe ${COND}/${TASK}/${SUFFIX}] already done"
            continue
        fi
        if [ -d "${OUT}" ]; then
            echo "[clean probe ${COND}/${TASK}/${SUFFIX}] removing partial files"
            find "${OUT}" -maxdepth 1 -type f -delete
        fi

        MODULE=$(probe_module "${TASK}")
        DATA_ARGS=$(probe_data_args "${TASK}")
        BS=$(probe_batch_size "${TASK}")

        echo
        echo "=== Probe ${COND}/${TASK} (${SUFFIX}) ==="
        # shellcheck disable=SC2086
        uv run python -m "${MODULE}" \
            ${CKPT_ARG} \
            ${DATA_ARGS} \
            --output-dir "${OUT}" \
            --max-steps 10000 \
            --batch-size "${BS}" \
            --wandb --wandb-run-name "probe_${TASK}_${COND}_${SUFFIX}"
    done
done

echo
echo "########################################################"
echo "# Done: ${SUFFIX}"
echo "# Pretrains in: ${RUNS_BASE}/{intact,pitch_strip,rhythm_strip,both_strip,esc50}/"
echo "# Probes in:    ${RUNS_BASE}/{...}/probe_{phone,boundary,f0,ks,sid}/"
echo "########################################################"
