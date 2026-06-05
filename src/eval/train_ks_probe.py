"""Train and evaluate a linear KS (Keyword Spotting) probe on a frozen encoder.

SUPERB-style utterance-level probe: features come from the frozen encoder,
are mean-pooled over time with a length mask, and a single linear classifier
predicts one of 12 classes (10 target keywords + "_unknown_" + "_silence_").

This is the **timbre + integrated capability probe**: KS requires picking up
short, broadband phonetic content, less reliant on long-range structure.

Usage:
    uv run python -m src.eval.train_ks_probe \
        --encoder-ckpt runs/intact/encoder.pt \
        --data-root /workspace/i_tatsuro/data/SpeechCommands \
        --output-dir runs/intact/probe_ks
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

from src.data.speech_commands import (
    N_CLASSES,
    SpeechCommandsDataset,
    collate_utt,
)
from src.eval.probe import FrozenUtteranceProbe, load_encoder

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


def cosine_lr(step: int, warmup: int, max_steps: int, peak: float) -> float:
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    progress = min(max(progress, 0.0), 1.0)
    return peak * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(
    model: FrozenUtteranceProbe,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    n_classes: int,
) -> dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    per_class_correct = torch.zeros(n_classes, dtype=torch.long)
    per_class_total = torch.zeros(n_classes, dtype=torch.long)
    loss_sum = 0.0
    loss_n = 0
    loss_fn = nn.CrossEntropyLoss(reduction="sum")

    for batch in loader:
        wav = batch["wav"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        wav_lens = batch["wav_lens"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda", dtype=torch.bfloat16):
            logits = model(wav, wav_lens)
        loss = loss_fn(logits.float(), labels)
        loss_sum += float(loss.detach())
        loss_n += int(labels.numel())

        preds = logits.argmax(dim=-1)
        correct += int((preds == labels).sum())
        total += int(labels.numel())

        labels_cpu = labels.detach().cpu()
        preds_cpu = preds.detach().cpu()
        for c in range(n_classes):
            mask = labels_cpu == c
            n_c = int(mask.sum())
            if n_c == 0:
                continue
            per_class_total[c] += n_c
            per_class_correct[c] += int(((preds_cpu == c) & mask).sum())

    seen = per_class_total > 0
    macro_acc = (
        (per_class_correct[seen].float() / per_class_total[seen].float()).mean().item()
        if seen.any()
        else 0.0
    )
    return {
        "loss": loss_sum / max(1, loss_n),
        "acc": correct / max(1, total),
        "macro_acc": macro_acc,
        "n": total,
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
    model = FrozenUtteranceProbe(encoder, n_classes=N_CLASSES).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(
        f"KS probe: trainable={n_trainable / 1e3:.1f}K / total={n_total / 1e6:.2f}M "
        f"(encoder frozen, {N_CLASSES} classes)"
    )

    train_ds = SpeechCommandsDataset(args.data_root, split="train", sample_rate=args.sample_rate, seed=args.silence_seed)
    eval_ds = SpeechCommandsDataset(args.data_root, split="val", sample_rate=args.sample_rate, seed=args.silence_seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=collate_utt,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=collate_utt,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss()

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
    best_acc = -1.0
    model.train()

    while step < args.max_steps:
        epoch += 1
        for batch in train_loader:
            wav = batch["wav"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            wav_lens = batch["wav_lens"].to(device, non_blocking=True)

            lr = cosine_lr(step, args.warmup_steps, args.max_steps, args.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                "cuda", enabled=args.amp and device.type == "cuda", dtype=torch.bfloat16
            ):
                logits = model(wav, wav_lens)
            loss = loss_fn(logits.float(), labels)
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
                metrics = evaluate(model, eval_loader, device, args.amp, N_CLASSES)
                model.train()
                metrics_log = {
                    "step": step,
                    "epoch": epoch,
                    "eval_loss": metrics["loss"],
                    "eval_acc": metrics["acc"],
                    "eval_macro_acc": metrics["macro_acc"],
                    "eval_n": metrics["n"],
                }
                print(
                    f"  [eval] loss={metrics['loss']:.4f} "
                    f"acc={metrics['acc']:.4f} macro={metrics['macro_acc']:.4f}"
                )
                log_f.write(json.dumps(metrics_log) + "\n")
                log_f.flush()
                if use_wandb:
                    wandb.log(
                        {
                            "eval/loss": metrics["loss"],
                            "eval/acc": metrics["acc"],
                            "eval/macro_acc": metrics["macro_acc"],
                        },
                        step=step,
                    )
                if metrics["acc"] > best_acc:
                    best_acc = metrics["acc"]
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
    print(f"Done. Best acc={best_acc:.4f}. Outputs in {args.output_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    # Data
    ap.add_argument("--data-root", type=Path, required=True,
                    help="Directory holding extracted Speech Commands v0.01")
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--silence-seed", type=int, default=42,
                    help="Deterministic seed for silence-crop sampling")
    # Encoder
    ap.add_argument("--encoder-ckpt", type=Path, default=None)
    ap.add_argument("--n-mels", type=int, default=80)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--ffn-dim", type=int, default=1024)
    # Optim
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--max-steps", type=int, default=10000)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--amp", action="store_true", default=True)
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
