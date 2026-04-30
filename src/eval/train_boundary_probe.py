"""Train and evaluate a linear phoneme-boundary detection probe.

Frame-level binary classification: positive iff a phoneme boundary occurs
between this frame and the previous one (i.e. ``phone[t] != phone[t-1]``).
Boundary labels are derived **on the fly** from the same per-frame phoneme
labels used by ``train_probe.py``, so the data pipeline is reused.

Boundaries are rare (~10–20 % of frames), so we report F1 on the positive
class as the primary metric and select the best ckpt by it.

This task is the **direct probe for the rhythm hypothesis**: a model that
has internalized rhythmic / temporal structure should localize phoneme
boundaries better.

Usage:
    uv run python -m src.eval.train_boundary_probe \
        --encoder-ckpt runs/intact/encoder.pt \
        --librispeech-root /workspace/.../LibriSpeech/LibriSpeech \
        --alignments-root /workspace/.../LibriSpeech/alignments/LibriSpeech \
        --output-dir runs/intact/probe_boundary
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.librispeech import (
    LibriSpeechPhonemeDataset,
    PAD_LABEL,
    collate_padded,
)
from src.eval.probe import FrozenLinearProbe, load_encoder

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


N_CLASSES = 2  # 0 = not a boundary, 1 = boundary


def cosine_lr(step: int, warmup: int, max_steps: int, peak: float) -> float:
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    progress = min(max(progress, 0.0), 1.0)
    return peak * 0.5 * (1.0 + math.cos(math.pi * progress))


def align_lengths(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    T = min(logits.shape[1], labels.shape[1])
    return logits[:, :T, :], labels[:, :T]


def phone_to_boundary_labels(phone_labels: torch.Tensor) -> torch.Tensor:
    """``(B, T)`` int64 phoneme IDs (with PAD_LABEL = -100 for padding) ->
    ``(B, T)`` int64 binary boundary labels (with PAD_LABEL kept).

    Frame ``t`` is positive iff ``phone[t] != phone[t-1]`` and both are
    valid (not PAD). Frame 0 of every utterance is treated as non-boundary
    (no previous frame to compare with).
    """
    B, T = phone_labels.shape
    out = torch.zeros_like(phone_labels)
    if T < 2:
        # No transitions possible; preserve PAD positions.
        out[phone_labels == PAD_LABEL] = PAD_LABEL
        return out
    prev = phone_labels[:, :-1]
    cur = phone_labels[:, 1:]
    valid_pair = (prev != PAD_LABEL) & (cur != PAD_LABEL)
    boundary = (cur != prev) & valid_pair
    out[:, 1:] = boundary.long()
    # Ensure padded current frames stay PAD.
    out[phone_labels == PAD_LABEL] = PAD_LABEL
    return out


@torch.no_grad()
def evaluate(
    model: FrozenLinearProbe,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
) -> dict[str, float]:
    model.eval()
    tp = fp = fn = tn = 0
    loss_sum = 0.0
    loss_n = 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_LABEL, reduction="sum")

    for batch in loader:
        wav = batch["wav"].to(device, non_blocking=True)
        phone_labels = batch["labels"].to(device, non_blocking=True)
        boundary_labels = phone_to_boundary_labels(phone_labels)
        with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda", dtype=torch.bfloat16):
            logits = model(wav)
        logits, labels_t = align_lengths(logits, boundary_labels)
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_labels = labels_t.reshape(-1)
        valid = flat_labels != PAD_LABEL
        v_logits = flat_logits[valid]
        v_labels = flat_labels[valid]
        if v_labels.numel() == 0:
            continue
        loss = loss_fn(v_logits.float(), v_labels)
        loss_sum += float(loss.detach())
        loss_n += int(v_labels.numel())

        preds = v_logits.argmax(dim=-1)
        tp += int(((preds == 1) & (v_labels == 1)).sum())
        fp += int(((preds == 1) & (v_labels == 0)).sum())
        fn += int(((preds == 0) & (v_labels == 1)).sum())
        tn += int(((preds == 0) & (v_labels == 0)).sum())

    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / max(1, total)
    pos_rate = (tp + fn) / max(1, total)

    return {
        "loss": loss_sum / max(1, loss_n),
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "frame_acc": accuracy,
        "pos_rate": pos_rate,
        "n_frames": total,
    }


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    encoder = load_encoder(
        state_dict_path=str(args.encoder_ckpt) if args.encoder_ckpt else None,
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ffn_dim=args.ffn_dim,
        dropout=0.0,
    )
    model = FrozenLinearProbe(encoder, n_classes=N_CLASSES).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(
        f"Boundary probe: trainable={n_trainable / 1e3:.1f}K / total={n_total / 1e6:.2f}M "
        f"(encoder frozen)"
    )

    train_ds = LibriSpeechPhonemeDataset(
        librispeech_dir=args.librispeech_root / args.train_split,
        alignments_dir=args.alignments_root / args.train_split,
        sample_rate=args.sample_rate,
        max_seconds=args.max_seconds,
    )
    eval_ds = LibriSpeechPhonemeDataset(
        librispeech_dir=args.librispeech_root / args.eval_split,
        alignments_dir=args.alignments_root / args.eval_split,
        sample_rate=args.sample_rate,
        max_seconds=args.max_seconds,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=collate_padded,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=collate_padded,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Optional positive-class weighting to counter the ~85/15 imbalance.
    if args.pos_weight > 0:
        class_weights = torch.tensor([1.0, args.pos_weight], device=device)
        loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_LABEL, weight=class_weights)
        print(f"Using class weights: [1.0, {args.pos_weight}] for [non-boundary, boundary]")
    else:
        loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_LABEL)

    use_wandb = bool(args.wandb)
    if use_wandb:
        if wandb is None:
            print("wandb not installed; disabling wandb logging.")
            use_wandb = False
        else:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or args.output_dir.name,
                dir=str(args.output_dir),
                config={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                mode=args.wandb_mode,
            )

    with (args.output_dir / "args.json").open("w") as f:
        json.dump({k: str(v) for k, v in vars(args).items()}, f, indent=2)
    log_path = args.output_dir / "probe.log.jsonl"
    log_f = log_path.open("w")

    step = 0
    epoch = 0
    t_start = time.time()
    last_log = t_start
    running = {"loss": 0.0, "n": 0}
    best_f1 = -1.0
    model.train()

    while step < args.max_steps:
        epoch += 1
        for batch in train_loader:
            wav = batch["wav"].to(device, non_blocking=True)
            phone_labels = batch["labels"].to(device, non_blocking=True)
            boundary_labels = phone_to_boundary_labels(phone_labels)

            lr = cosine_lr(step, args.warmup_steps, args.max_steps, args.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                "cuda", enabled=args.amp and device.type == "cuda", dtype=torch.bfloat16
            ):
                logits = model(wav)
            logits, labels_t = align_lengths(logits, boundary_labels)
            loss = loss_fn(
                logits.float().reshape(-1, logits.shape[-1]),
                labels_t.reshape(-1),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()

            running["loss"] += float(loss.detach())
            running["n"] += 1
            step += 1

            now = time.time()
            if now - last_log >= args.log_every_seconds or step == args.max_steps:
                avg_loss = running["loss"] / max(1, running["n"])
                steps_per_sec = running["n"] / (now - last_log)
                rec = {
                    "step": step,
                    "epoch": epoch,
                    "train_loss": avg_loss,
                    "lr": lr,
                    "steps_per_sec": steps_per_sec,
                    "elapsed_sec": now - t_start,
                }
                print(
                    f"step {step:>6}/{args.max_steps} ep{epoch:>3} "
                    f"loss={avg_loss:.4f} lr={lr:.2e} "
                    f"{steps_per_sec:.2f} step/s elapsed={now - t_start:.0f}s"
                )
                log_f.write(json.dumps(rec) + "\n")
                log_f.flush()
                if use_wandb:
                    wandb.log(
                        {
                            "train/loss": avg_loss,
                            "train/lr": lr,
                            "train/steps_per_sec": steps_per_sec,
                            "train/epoch": epoch,
                        },
                        step=step,
                    )
                running = {"loss": 0.0, "n": 0}
                last_log = now

            if step % args.eval_every == 0 or step == args.max_steps:
                metrics = evaluate(model, eval_loader, device, args.amp)
                model.train()
                metrics_log = {
                    "step": step,
                    "epoch": epoch,
                    "eval_loss": metrics["loss"],
                    "eval_f1": metrics["f1"],
                    "eval_precision": metrics["precision"],
                    "eval_recall": metrics["recall"],
                    "eval_frame_acc": metrics["frame_acc"],
                    "eval_pos_rate": metrics["pos_rate"],
                    "eval_n_frames": metrics["n_frames"],
                }
                print(
                    f"  [eval] loss={metrics['loss']:.4f} "
                    f"F1={metrics['f1']:.4f} "
                    f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
                    f"acc={metrics['frame_acc']:.4f} pos_rate={metrics['pos_rate']:.4f}"
                )
                log_f.write(json.dumps(metrics_log) + "\n")
                log_f.flush()
                if use_wandb:
                    wandb.log(
                        {
                            "eval/loss": metrics["loss"],
                            "eval/f1": metrics["f1"],
                            "eval/precision": metrics["precision"],
                            "eval/recall": metrics["recall"],
                            "eval/frame_acc": metrics["frame_acc"],
                            "eval/pos_rate": metrics["pos_rate"],
                        },
                        step=step,
                    )
                if metrics["f1"] > best_f1:
                    best_f1 = metrics["f1"]
                    torch.save(model.head.state_dict(), args.output_dir / "probe_head_best.pt")
                    with (args.output_dir / "best_metrics.json").open("w") as f:
                        json.dump(metrics_log, f, indent=2)

            if step >= args.max_steps:
                break

    log_f.close()
    if use_wandb:
        try:
            wandb.finish()
        except Exception as e:  # noqa: BLE001
            print(f"warning: wandb teardown failed: {type(e).__name__}: {e}")
    print(f"Done. Best F1={best_f1:.4f}. Outputs in {args.output_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    # Data
    ap.add_argument("--librispeech-root", type=Path, required=True)
    ap.add_argument("--alignments-root", type=Path, required=True)
    ap.add_argument("--train-split", type=str, default="train-clean-100")
    ap.add_argument("--eval-split", type=str, default="dev-clean")
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--max-seconds", type=float, default=16.0)
    ap.add_argument("--num-workers", type=int, default=4)
    # Encoder
    ap.add_argument("--encoder-ckpt", type=Path, default=None,
                    help="Pretrained encoder; omit for random init.")
    ap.add_argument("--n-mels", type=int, default=80)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--ffn-dim", type=int, default=1024)
    # Optim
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--eval-batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--max-steps", type=int, default=10000)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument(
        "--pos-weight", type=float, default=9.0,
        help=(
            "Class weight for the positive (boundary) class in CE loss. "
            "Boundaries are ~10%% of frames, so a 9x weight roughly balances "
            "classes (inverse frequency). Set to 0 to disable weighting; "
            "with no weight the probe collapses to predicting all-negative."
        ),
    )
    # I/O & logging
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--log-every-seconds", type=float, default=10.0)
    ap.add_argument("--eval-every", type=int, default=1000)
    # wandb
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", type=str, default="music2speech")
    ap.add_argument("--wandb-run-name", type=str, default=None)
    ap.add_argument("--wandb-mode", type=str, default="online",
                    choices=["online", "offline", "disabled"])
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
