#!/usr/bin/env bash
# Run downstream probes across multiple encoder conditions.
# Designed to be invokable on each GPU in parallel: set CUDA_VISIBLE_DEVICES
# and CONDITIONS to split work between GPUs. Skips probes whose
# best_metrics.json already exists, so re-invoking is safe.
#
# Frame-level tasks: phone, boundary, f0   (LibriSpeech + MFA / F0)
# Utterance-level tasks: ks, sid           (Speech Commands v1, VoxCeleb1)
#
# Usage examples:
#   # All conditions, all 5 tasks, on GPU 0:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_all_probes.sh
#
#   # Split across 2 GPUs:
#   CUDA_VISIBLE_DEVICES=0 CONDITIONS="intact pitch_strip rhythm_strip" \
#       bash scripts/run_all_probes.sh
#   CUDA_VISIBLE_DEVICES=1 CONDITIONS="both_strip esc50 random_init" \
#       bash scripts/run_all_probes.sh
#
#   # Only run F0 probes:
#   TASKS="f0" bash scripts/run_all_probes.sh
#
# Required env (have defaults):
#   CONDITIONS  - space-separated subset of:
#                 intact pitch_strip rhythm_strip both_strip esc50 random_init
#   TASKS       - space-separated subset of: phone boundary f0 ks sid
#   PROBE_STEPS - max-steps for each probe (default 10000)

set -euo pipefail
cd "$(dirname "$0")/.."
unset VIRTUAL_ENV || true

CONDITIONS=${CONDITIONS:-"intact pitch_strip rhythm_strip both_strip esc50 random_init"}
TASKS=${TASKS:-"phone boundary f0 ks sid"}
PROBE_STEPS=${PROBE_STEPS:-10000}

LIBRI_ROOT=/workspace/i_tatsuro/data/LibriSpeech/LibriSpeech
ALIGN_ROOT=/workspace/i_tatsuro/data/LibriSpeech/alignments/LibriSpeech
F0_ROOT=/workspace/i_tatsuro/data/LibriSpeech/f0
SC_ROOT=/workspace/i_tatsuro/data/SpeechCommands
VOX_WAV_DIR=/workspace/i_tatsuro/data/VoxCeleb1/wav
VOX_SPLIT=/workspace/i_tatsuro/data/VoxCeleb1/iden_split.txt
RUNS_BASE=/workspace/i_tatsuro/projects/music2speech/runs

# Map condition -> encoder path (empty string = random init).
encoder_path() {
    case "$1" in
        intact)        echo "${RUNS_BASE}/intact/encoder.pt" ;;
        pitch_strip)   echo "${RUNS_BASE}/pitch_strip/encoder.pt" ;;
        rhythm_strip)  echo "${RUNS_BASE}/rhythm_strip/encoder.pt" ;;
        both_strip)    echo "${RUNS_BASE}/both_strip/encoder.pt" ;;
        esc50)         echo "${RUNS_BASE}/esc50/encoder.pt" ;;
        random_init)   echo "" ;;
        *)             echo "_UNKNOWN_" ;;
    esac
}

# Where probe outputs live for this (condition, task) pair.
output_dir() {
    local cond=$1 task=$2
    case "$cond" in
        random_init) echo "${RUNS_BASE}/random_init/probe_${task}" ;;
        *)           echo "${RUNS_BASE}/${cond}/probe_${task}" ;;
    esac
}

run_probe() {
    local cond=$1 task=$2
    local ckpt
    ckpt=$(encoder_path "$cond")
    if [[ "$ckpt" == "_UNKNOWN_" ]]; then
        echo "[skip] unknown condition: $cond"
        return
    fi
    if [[ -n "$ckpt" && ! -f "$ckpt" ]]; then
        echo "[skip ${cond}/${task}] missing encoder: $ckpt"
        return
    fi

    local out
    out=$(output_dir "$cond" "$task")
    if [[ -f "${out}/best_metrics.json" ]]; then
        echo "[skip ${cond}/${task}] already done: ${out}/best_metrics.json"
        return
    fi
    if [[ -d "$out" ]]; then
        echo "[clean ${cond}/${task}] removing partial files in $out"
        find "$out" -maxdepth 1 -type f -delete
    fi

    local ckpt_arg=""
    if [[ -n "$ckpt" ]]; then
        ckpt_arg="--encoder-ckpt $ckpt"
    fi

    # Per-task module, data args, and batch size. Frame-level probes use
    # batch_size=8 (long utterances); utterance-level probes use 32+ since
    # 1 s / 3 s crops fit easily.
    local module data_args batch_size
    case "$task" in
        phone)
            module="src.eval.train_probe"
            data_args="--librispeech-root ${LIBRI_ROOT} --alignments-root ${ALIGN_ROOT}"
            batch_size=8
            ;;
        boundary)
            module="src.eval.train_boundary_probe"
            data_args="--librispeech-root ${LIBRI_ROOT} --alignments-root ${ALIGN_ROOT}"
            batch_size=8
            ;;
        f0)
            module="src.eval.train_f0_probe"
            data_args="--librispeech-root ${LIBRI_ROOT} --f0-root ${F0_ROOT}"
            batch_size=8
            ;;
        ks)
            module="src.eval.train_ks_probe"
            data_args="--data-root ${SC_ROOT}"
            batch_size=32
            ;;
        sid)
            module="src.eval.train_sid_probe"
            data_args="--wav-dir ${VOX_WAV_DIR} --split-file ${VOX_SPLIT}"
            batch_size=32
            ;;
        *)
            echo "[skip] unknown task: $task"
            return
            ;;
    esac

    echo
    echo "========================================================"
    echo "  ${cond} / ${task}  (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})"
    echo "========================================================"

    # shellcheck disable=SC2086
    uv run python -m "${module}" \
        ${ckpt_arg} \
        ${data_args} \
        --output-dir "${out}" \
        --max-steps "${PROBE_STEPS}" \
        --batch-size "${batch_size}" \
        --wandb --wandb-run-name "probe_${task}_${cond}"
}

echo "Conditions: ${CONDITIONS}"
echo "Tasks     : ${TASKS}"
echo "Probe steps: ${PROBE_STEPS}"

for COND in ${CONDITIONS}; do
    for TASK in ${TASKS}; do
        run_probe "${COND}" "${TASK}"
    done
done

echo
echo "All probe jobs complete (or skipped) for: ${CONDITIONS} x ${TASKS}"
