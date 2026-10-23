"""Training harness for the neural compressor."""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.blocks import CompressorNet
from model.losses import W_CURVE, W_PARAM, W_SMOOTH, W_STFT, CompressorLoss
from training.dataset import CompressorDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_scaler(device: torch.device, enabled: bool):
    try:
        return torch.amp.GradScaler(device.type, enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def amp_context(device: torch.device, enabled: bool):
    try:
        return torch.autocast(device_type=device.type, enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=enabled)


def safe_torch_load(path: Path, *, map_location="cpu"):
    try:
        obj = torch.load(path, map_location=map_location, weights_only=True)
        print(f"[load] {path.name}: weights_only=True")
        return obj
    except Exception as exc:
        print(f"[load] {path.name}: weights_only=True failed ({type(exc).__name__}); "
              f"retrying with weights_only=False")
        return torch.load(path, map_location=map_location, weights_only=False)


def unpack_forward(out):
    if isinstance(out, dict):
        return out["p_pred"], out["g_pred"], out["y"]
    if len(out) == 3:
        p_pred, g_pred, aux = out
        return p_pred, g_pred, aux["y"]
    if len(out) == 4:
        p_pred, g_pred, _g_bands, aux = out
        return p_pred, g_pred, aux["y"]
    raise TypeError(f"unexpected forward() return: {type(out)} len={len(out)}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the neural compressor (distillation).")
    p.add_argument("--out-dir", type=Path, default=ROOT / "checkpoints")
    p.add_argument("--n-samples", type=int, default=4096)
    p.add_argument("--n-val", type=int, default=256)
    p.add_argument("--duration", type=float, default=1.5)
    p.add_argument("--sample-rate", type=int, default=44100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--stft-every", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-amp", action="store_false", dest="amp")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    if args.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)

    device = torch.device(args.device)
    stft_every = max(1, args.stft_every)

    train_ds = CompressorDataset(
        n_samples=args.n_samples, duration_s=args.duration,
        sample_rate=args.sample_rate, seed=args.seed,
    )
    val_ds = CompressorDataset(
        n_samples=args.n_val, duration_s=args.duration,
        sample_rate=args.sample_rate, seed=args.seed + 7_919,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=len(train_ds) >= args.batch_size,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
    )

    if len(train_loader) == 0:
        raise SystemExit(
            f"train loader is empty: n_samples={args.n_samples} < batch_size={args.batch_size}."
        )

    model = CompressorNet().to(device)
    criterion = CompressorLoss(sample_rate=args.sample_rate).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(1, args.epochs))
    scaler = make_scaler(device, enabled=args.amp and device.type == "cuda")

    start_epoch = 0
    best_val = math.inf
    history: list[dict] = []

    if args.resume is not None:
        ckpt = safe_torch_load(args.resume)
        model.load_state_dict(ckpt["model"], strict=True)
        if "optim" in ckpt:
            optim.load_state_dict(ckpt["optim"])
        if "sched" in ckpt:
            sched.load_state_dict(ckpt["sched"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("best_val", math.inf))
        history = list(ckpt.get("history", []))

    if args.dry_run:
        model.train()
        batch = next(iter(train_loader))
        x = batch["x"].to(device)
        with torch.no_grad():
            with amp_context(device, enabled=False):
                p_pred, g_pred, y_pred = unpack_forward(model(x))
        print(f"[dry-run] x={tuple(x.shape)} p_pred={tuple(p_pred.shape)} "
              f"g_pred={tuple(g_pred.shape)} y_pred={tuple(y_pred.shape)} "
              f"finite={bool(torch.isfinite(y_pred).all())}")
        return 0

    def evaluate() -> dict[str, float]:
        model.eval()
        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        probe_err: list[float] = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                p_pred, g_pred, y_pred = unpack_forward(model(x))
                logs = criterion(
                    p_pred, g_pred, batch["params"].to(device), batch["g"].to(device),
                    y_pred=y_pred, y_tgt=batch["y"].to(device), use_stft=True,
                )
                for k, v in logs.items():
                    sums[k] = sums.get(k, 0.0) + float(v)
                    counts[k] = counts.get(k, 0) + 1
                probe_err.append(float((p_pred - batch["params"].to(device)).abs().mean()))
        model.train()
        out = {k: sums[k] / max(1, counts[k]) for k in sums}
        out["probe_mae"] = float(np.mean(probe_err)) if probe_err else float("nan")
        return out

    def save_ckpt(path: Path, epoch: int, val: float) -> None:
        torch.save(
            {
                "model": model.state_dict(),
                "optim": optim.state_dict(),
                "sched": sched.state_dict(),
                "epoch": epoch,
                "best_val": val,
                "history": history,
                "args": vars(args) | {"out_dir": str(args.out_dir), "resume": str(args.resume)},
            },
            path,
        )

    interrupted = {"flag": False}

    def _on_sigint(_sig, _frm):
        interrupted["flag"] = True

    signal.signal(signal.SIGINT, _on_sigint)

    t0 = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        n_stft = 0
        n_skipped = 0

        for step, batch in enumerate(train_loader):
            if interrupted["flag"]:
                break

            x = batch["x"].to(device)
            y_tgt = batch["y"].to(device)
            g_tgt = batch["g"].to(device)
            p_tgt = batch["params"].to(device)

            use_stft = (step % stft_every) == 0
            optim.zero_grad(set_to_none=True)

            with amp_context(device, enabled=scaler.is_enabled()):
                p_pred, g_pred, y_pred = unpack_forward(model(x))

            logs = criterion(
                p_pred, g_pred, p_tgt, g_tgt,
                y_pred=y_pred, y_tgt=y_tgt, use_stft=use_stft,
            )
            total = logs["total"]

            if not torch.isfinite(total):
                n_skipped += 1
                continue

            scale_before = scaler.get_scale() if scaler.is_enabled() else 1.0
            scaler.scale(total).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optim)
            scaler.update()

            stepped = True
            if scaler.is_enabled():
                stepped = scaler.get_scale() >= scale_before
            if stepped:
                sched.step()

            for k, v in logs.items():
                if k == "stft" and not use_stft:
                    continue
                sums[k] = sums.get(k, 0.0) + float(v)
                counts[k] = counts.get(k, 0) + 1
            if use_stft:
                n_stft += 1

        if interrupted["flag"]:
            save_ckpt(args.out_dir / "last.pt", epoch, best_val)
            return 130

        train_metrics = {k: sums[k] / max(1, counts[k]) for k in sums}
        val_metrics = evaluate()
        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "n_stft_steps": n_stft,
                "n_skipped": n_skipped,
                "lr": optim.param_groups[0]["lr"],
            }
        )

        if val_metrics["curve"] < best_val:
            best_val = val_metrics["curve"]
            save_ckpt(args.out_dir / "best.pt", epoch, best_val)

        save_ckpt(args.out_dir / "last.pt", epoch, best_val)

    elapsed = time.time() - t0
    report = {
        "argv": sys.argv,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "platform": platform.platform(),
        },
        "seed": args.seed,
        "best_val_curve": best_val,
        "elapsed_s": round(elapsed, 2),
        "history": history,
        "loss_weights": {
            "stft": W_STFT, "curve": W_CURVE, "param": W_PARAM, "smooth": W_SMOOTH,
        },
    }
    (args.out_dir / "train_report.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
