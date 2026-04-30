"""Self-supervised pretraining (Masked Spectrogram Modeling).

Trains the MelTransformerEncoder on a folder of wav files. Splits the file
list into train/val by seeded shuffle, evaluates val MSM loss every
``--val-every`` steps, and stops early if val loss does not improve for
``--patience`` consecutive evaluations. The best-val encoder is saved as
``encoder.pt``.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import WavFolderDataset, collate_waveforms
from src.models.encoder import MelTransformerEncoder
from src.models.msm import MaskedSpecModel

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


def cosine_lr(step: int, warmup_steps: int, max_steps: int, peak_lr: float) -> float:
    if step < warmup_steps:
        return peak_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return peak_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def build_model(args: argparse.Namespace) -> tuple[MelTransformerEncoder, MaskedSpecModel]:
    encoder = MelTransformerEncoder(
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
    )
    model = MaskedSpecModel(
        encoder=encoder,
        mask_prob=args.mask_prob,
        mask_length=args.mask_length,
    )
    return encoder, model


def init_wandb(args: argparse.Namespace) -> bool:
    """Initialize wandb if --wandb is set. Returns True iff initialized."""
    if not args.wandb:
        return False
    if wandb is None:
        print("wandb not installed; skipping wandb logging. (uv add wandb)")
        return False
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or args.output_dir.name,
        dir=str(args.output_dir),
        config={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        mode=args.wandb_mode,
    )
    return True


@torch.no_grad()
def evaluate_val(
    model: MaskedSpecModel,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    mask_seed: int = 0,
) -> dict[str, float]:
    """Compute mean MSM loss over the val set.

    The encoder is set to ``eval()`` (no dropout) and we fix the global RNG so
    that mask sampling is reproducible across val calls — otherwise the
    val-loss curve is dominated by mask noise rather than model improvement.
    """
    was_training = model.training
    model.eval()
    rng_state_cpu = torch.get_rng_state()
    rng_state_cuda = (
        torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    )
    torch.manual_seed(mask_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(mask_seed)

    loss_sum = 0.0
    n_examples = 0
    try:
        for wav in loader:
            wav = wav.to(device, non_blocking=True)
            with torch.amp.autocast(
                "cuda", enabled=amp and device.type == "cuda", dtype=torch.bfloat16
            ):
                out = model(wav)
            loss_sum += float(out["loss"]) * wav.shape[0]
            n_examples += wav.shape[0]
    finally:
        torch.set_rng_state(rng_state_cpu)
        if rng_state_cuda is not None:
            torch.cuda.set_rng_state(rng_state_cuda, device)
        if was_training:
            model.train()

    return {
        "loss": loss_sum / max(1, n_examples),
        "n_examples": n_examples,
    }


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    use_wandb = init_wandb(args)

    train_dataset = WavFolderDataset(
        root=args.data_dir,
        sample_rate=args.sample_rate,
        window_seconds=args.window_seconds,
        split="train",
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
        deterministic_window=False,
    )
    val_dataset = WavFolderDataset(
        root=args.data_dir,
        sample_rate=args.sample_rate,
        window_seconds=args.window_seconds,
        split="val",
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
        deterministic_window=True,  # stable val loss across calls
    )
    print(f"Split: train={len(train_dataset)} val={len(val_dataset)} (seed={args.split_seed})")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=collate_waveforms,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(2, args.num_workers),
        drop_last=False,
        collate_fn=collate_waveforms,
        pin_memory=True,
        persistent_workers=False,
    )

    encoder, model = build_model(args)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_enc = sum(p.numel() for p in encoder.parameters())
    print(f"Model params: {n_params / 1e6:.2f}M (encoder={n_enc / 1e6:.2f}M)")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    with (args.output_dir / "args.json").open("w") as f:
        json.dump({k: str(v) for k, v in vars(args).items()}, f, indent=2)
    log_f = (args.output_dir / "train.log.jsonl").open("w")

    step = 0
    epoch = 0
    t_start = time.time()
    last_log = t_start
    running = {"loss": 0.0, "n": 0}

    best_val = float("inf")
    best_step = 0
    patience_counter = 0
    val_history: list[tuple[int, float]] = []
    best_path = args.output_dir / "encoder_best_val.pt"

    model.train()
    stop = False

    while step < args.max_steps and not stop:
        epoch += 1
        for wav in train_loader:
            wav = wav.to(device, non_blocking=True)
            lr = cosine_lr(step, args.warmup_steps, args.max_steps, args.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                "cuda", enabled=args.amp and device.type == "cuda", dtype=torch.bfloat16
            ):
                out = model(wav)
                loss = out["loss"]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running["loss"] += float(loss.detach())
            running["n"] += 1
            step += 1

            now = time.time()
            if now - last_log >= args.log_every_seconds or step == args.max_steps:
                avg_loss = running["loss"] / max(1, running["n"])
                steps_per_sec = running["n"] / (now - last_log)
                elapsed = now - t_start
                rec = {
                    "step": step,
                    "epoch": epoch,
                    "loss": avg_loss,
                    "lr": lr,
                    "mask_ratio": float(out["mask_ratio"]),
                    "steps_per_sec": steps_per_sec,
                    "elapsed_sec": elapsed,
                }
                print(
                    f"step {step:>6}/{args.max_steps} ep{epoch:>3} "
                    f"loss={avg_loss:.4f} lr={lr:.2e} "
                    f"mask={float(out['mask_ratio']):.2f} "
                    f"{steps_per_sec:.2f} step/s elapsed={elapsed:.0f}s"
                )
                log_f.write(json.dumps(rec) + "\n")
                log_f.flush()
                if use_wandb:
                    wandb.log(
                        {
                            "train/loss": avg_loss,
                            "train/lr": lr,
                            "train/mask_ratio": float(out["mask_ratio"]),
                            "train/steps_per_sec": steps_per_sec,
                            "train/elapsed_sec": elapsed,
                            "train/epoch": epoch,
                        },
                        step=step,
                    )
                running = {"loss": 0.0, "n": 0}
                last_log = now

            if step % args.save_every == 0 or step == args.max_steps:
                torch.save(encoder.state_dict(), args.output_dir / f"encoder_step{step}.pt")

            if step % args.val_every == 0 or step == args.max_steps:
                val_metrics = evaluate_val(
                    model, val_loader, device, args.amp, mask_seed=args.val_mask_seed
                )
                val_loss = val_metrics["loss"]
                val_history.append((step, val_loss))
                improved = val_loss < best_val - args.min_delta
                if improved:
                    best_val = val_loss
                    best_step = step
                    patience_counter = 0
                    torch.save(encoder.state_dict(), best_path)
                else:
                    patience_counter += 1

                rec = {
                    "step": step,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "best_val": best_val,
                    "best_step": best_step,
                    "patience_counter": patience_counter,
                    "improved": improved,
                }
                print(
                    f"  [val] step {step}: loss={val_loss:.4f} "
                    f"(best={best_val:.4f}@{best_step}, patience={patience_counter}/{args.patience})"
                )
                log_f.write(json.dumps(rec) + "\n")
                log_f.flush()
                if use_wandb:
                    wandb.log(
                        {
                            "val/loss": val_loss,
                            "val/best": best_val,
                            "val/best_step": best_step,
                            "val/patience_counter": patience_counter,
                        },
                        step=step,
                    )

                if patience_counter >= args.patience:
                    print(
                        f"Early stopping at step {step}: no val improvement for "
                        f"{args.patience} evaluations. Best val_loss={best_val:.4f} "
                        f"at step {best_step}."
                    )
                    stop = True
                    break

            if step >= args.max_steps:
                break

    # Promote best-val to encoder.pt for downstream use.
    if best_path.exists():
        shutil.copyfile(best_path, args.output_dir / "encoder.pt")
        promo_msg = f"best-val (step {best_step}, val_loss={best_val:.4f})"
    else:
        # Fallback: no val ever ran; use last weights.
        torch.save(encoder.state_dict(), args.output_dir / "encoder.pt")
        promo_msg = f"latest (step {step}; no val checkpoint available)"

    summary = {
        "total_steps": step,
        "epochs": epoch,
        "best_val_loss": best_val if best_val < float("inf") else None,
        "best_step": best_step,
        "stopped_early": stop,
        "val_history": val_history,
        "encoder_pt": promo_msg,
    }
    with (args.output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    log_f.close()
    if use_wandb:
        # Don't let wandb teardown failures (e.g. an OSError from a closed
        # log handler when run inside tmux) propagate and crash the script —
        # the encoder/summary/log files are already on disk by this point.
        try:
            wandb.summary["best_val_loss"] = best_val if best_val < float("inf") else None
            wandb.summary["best_step"] = best_step
            wandb.summary["stopped_early"] = stop
            wandb.finish()
        except Exception as e:  # noqa: BLE001
            print(f"warning: wandb teardown failed: {type(e).__name__}: {e}")
    print(f"Done. encoder.pt = {promo_msg}. Outputs in {args.output_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    # Data
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--window-seconds", type=float, default=5.0)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--val-ratio", type=float, default=0.05,
                    help="Fraction of files held out for validation.")
    ap.add_argument("--split-seed", type=int, default=42)
    # Model
    ap.add_argument("--n-mels", type=int, default=80)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--ffn-dim", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.1)
    # MSM
    ap.add_argument("--mask-prob", type=float, default=0.065)
    ap.add_argument("--mask-length", type=int, default=10)
    # Optim
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--warmup-steps", type=int, default=1000)
    ap.add_argument("--amp", action="store_true", default=True)
    # Validation / early stopping
    ap.add_argument("--val-every", type=int, default=500,
                    help="Steps between validation evaluations.")
    ap.add_argument("--patience", type=int, default=5,
                    help="Stop after this many val evaluations without improvement.")
    ap.add_argument("--min-delta", type=float, default=1e-4,
                    help="Minimum decrease in val loss to count as improvement.")
    ap.add_argument("--val-mask-seed", type=int, default=0,
                    help="Seed used for mask sampling during val (kept fixed across calls).")
    # Logging
    ap.add_argument("--log-every-seconds", type=float, default=10.0)
    ap.add_argument("--save-every", type=int, default=2000)
    # wandb
    ap.add_argument("--wandb", action="store_true", help="Log to Weights & Biases.")
    ap.add_argument("--wandb-project", type=str, default="music2speech")
    ap.add_argument("--wandb-run-name", type=str, default=None,
                    help="Defaults to output-dir basename.")
    ap.add_argument("--wandb-mode", type=str, default="online",
                    choices=["online", "offline", "disabled"])
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
