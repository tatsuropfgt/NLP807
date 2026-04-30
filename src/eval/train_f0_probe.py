"""Train and evaluate a linear F0 (pitch) regression probe.

Per-frame regression: predict ``log2(F0[Hz])`` for **voiced** frames; ignore
unvoiced (``F0 == 0``) and padded frames in both loss and eval. The probe
is a single linear layer on top of the frozen encoder.

This is the **direct probe for the pitch hypothesis**: a model that has
internalized pitch / spectral structure should track speech F0 better.

Eval metrics:
- ``rmse_log2``: RMSE in log2(Hz) units across voiced frames.
- ``rmse_cents``: same as above, scaled by 1200 (musical cents).
- ``gpe@50ct``: Gross Pitch Error rate at 50-cent tolerance — fraction of
  voiced frames whose prediction is off by more than 50 cents.

Usage:
    uv run python -m src.eval.train_f0_probe \
        --encoder-ckpt runs/intact/encoder.pt \
        --librispeech-root /workspace/.../LibriSpeech/LibriSpeech \
        --f0-root /workspace/.../LibriSpeech/f0 \
        --output-dir runs/intact/probe_f0
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
    LibriSpeechF0Dataset,
    collate_padded_f0,
)
from src.eval.probe import FrozenLinearProbe, load_encoder

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


N_OUT = 1  # one regression output: log2(F0)


def cosine_lr(step: int, warmup: int, max_steps: int, peak: float) -> float:
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    progress = min(max(progress, 0.0), 1.0)
    return peak * 0.5 * (1.0 + math.cos(math.pi * progress))


def align_lengths(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    T = min(pred.shape[1], target.shape[1])
    return pred[:, :T], target[:, :T], valid[:, :T]


def f0_to_log2(f0: torch.Tensor) -> torch.Tensor:
    """Convert F0 (Hz) to log2(F0). F0 must be > 0 (caller masks unvoiced)."""
    return torch.log2(f0.clamp(min=1.0))


@torch.no_grad()
def evaluate(
    model: FrozenLinearProbe,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    gpe_threshold_cents: float = 50.0,
) -> dict[str, float]:
    model.eval()
    sq_err_sum = 0.0
    abs_err_sum = 0.0
    n = 0
    n_gpe = 0  # frames with error > threshold

    for batch in loader:
        wav = batch["wav"].to(device, non_blocking=True)
        f0 = batch["f0"].to(device, non_blocking=True)
        f0_valid = batch["f0_valid"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda", dtype=torch.bfloat16):
            pred = model(wav).squeeze(-1)  # (B, T)
        pred, f0, f0_valid = align_lengths(pred, f0, f0_valid)

        voiced = f0_valid & (f0 > 0)  # True only for valid voiced frames
        if voiced.sum() == 0:
            continue
        log2_target = f0_to_log2(f0)
        diff = (pred.float() - log2_target)[voiced]
        sq_err_sum += float((diff ** 2).sum())
        abs_err_sum += float(diff.abs().sum())
        n += int(voiced.sum())
        n_gpe += int(((diff.abs() * 1200.0) > gpe_threshold_cents).sum())

    if n == 0:
        return {
            "rmse_log2": float("nan"),
            "rmse_cents": float("nan"),
            "mae_log2": float("nan"),
            "mae_cents": float("nan"),
            "gpe_50ct": float("nan"),
            "n_voiced": 0,
        }
    rmse_log2 = math.sqrt(sq_err_sum / n)
    mae_log2 = abs_err_sum / n
    return {
        "rmse_log2": rmse_log2,
        "rmse_cents": rmse_log2 * 1200.0,
        "mae_log2": mae_log2,
        "mae_cents": mae_log2 * 1200.0,
        "gpe_50ct": n_gpe / n,
        "n_voiced": n,
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
    model = FrozenLinearProbe(encoder, n_classes=N_OUT).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(
        f"F0 probe: trainable={n_trainable / 1e3:.1f}K / total={n_total / 1e6:.2f}M "
        f"(encoder frozen)"
    )

    train_ds = LibriSpeechF0Dataset(
        librispeech_dir=args.librispeech_root / args.train_split,
        f0_dir=args.f0_root / args.train_split,
        sample_rate=args.sample_rate,
        max_seconds=args.max_seconds,
    )
    eval_ds = LibriSpeechF0Dataset(
        librispeech_dir=args.librispeech_root / args.eval_split,
        f0_dir=args.f0_root / args.eval_split,
        sample_rate=args.sample_rate,
        max_seconds=args.max_seconds,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=collate_padded_f0,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=collate_padded_f0,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

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
    best_rmse = float("inf")
    model.train()

    while step < args.max_steps:
        epoch += 1
        for batch in train_loader:
            wav = batch["wav"].to(device, non_blocking=True)
            f0 = batch["f0"].to(device, non_blocking=True)
            f0_valid = batch["f0_valid"].to(device, non_blocking=True)

            lr = cosine_lr(step, args.warmup_steps, args.max_steps, args.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                "cuda", enabled=args.amp and device.type == "cuda", dtype=torch.bfloat16
            ):
                pred = model(wav).squeeze(-1)
            pred, f0_t, f0_valid_t = align_lengths(pred, f0, f0_valid)
            voiced = f0_valid_t & (f0_t > 0)
            if voiced.sum() == 0:
                continue
            log2_target = f0_to_log2(f0_t)
            loss = ((pred.float() - log2_target) ** 2)[voiced].mean()
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
                    "eval_rmse_log2": metrics["rmse_log2"],
                    "eval_rmse_cents": metrics["rmse_cents"],
                    "eval_mae_log2": metrics["mae_log2"],
                    "eval_mae_cents": metrics["mae_cents"],
                    "eval_gpe_50ct": metrics["gpe_50ct"],
                    "eval_n_voiced": metrics["n_voiced"],
                }
                print(
                    f"  [eval] RMSE={metrics['rmse_cents']:.1f}c "
                    f"MAE={metrics['mae_cents']:.1f}c "
                    f"GPE@50c={metrics['gpe_50ct']:.4f} "
                    f"n_voiced={metrics['n_voiced']}"
                )
                log_f.write(json.dumps(metrics_log) + "\n")
                log_f.flush()
                if use_wandb:
                    wandb.log(
                        {
                            "eval/rmse_log2": metrics["rmse_log2"],
                            "eval/rmse_cents": metrics["rmse_cents"],
                            "eval/mae_cents": metrics["mae_cents"],
                            "eval/gpe_50ct": metrics["gpe_50ct"],
                        },
                        step=step,
                    )
                if metrics["rmse_log2"] < best_rmse:
                    best_rmse = metrics["rmse_log2"]
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
    print(f"Done. Best RMSE={best_rmse * 1200:.1f} cents. Outputs in {args.output_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    # Data
    ap.add_argument("--librispeech-root", type=Path, required=True)
    ap.add_argument("--f0-root", type=Path, required=True)
    ap.add_argument("--train-split", type=str, default="train-clean-100")
    ap.add_argument("--eval-split", type=str, default="dev-clean")
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--max-seconds", type=float, default=16.0)
    ap.add_argument("--num-workers", type=int, default=4)
    # Encoder
    ap.add_argument("--encoder-ckpt", type=Path, default=None)
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
