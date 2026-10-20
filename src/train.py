"""Config-driven training, one entry point for all five models.

    python -m src.train --config configs/unet_scratch.yaml
    python -m src.train --config configs/segformer.yaml --set train.epochs=5 data.root=data/fake-pet

Resuming after a crash, Ctrl+C, or a closed laptop:

    python -m src.train --config configs/unet_scratch.yaml --resume
    python -m src.train --resume runs/unet_scratch/last.pt --set train.epochs=40   # resume and extend

last.pt holds everything needed to continue exactly where training stopped: weights, optimizer,
LR schedule, AMP scaler, RNG state, best score so far, and the position inside the current epoch.
It is written at the end of every epoch, every `train.save_every_min` minutes inside an epoch,
and when you press Ctrl+C. Writes are atomic, so a crash during a save cannot corrupt it.

Writes runs/<run_name>/{best.pt, last.pt, history.jsonl, overlays/, summary.json, events.*}.
Watch live with:  tensorboard --logdir runs
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Sampler

if __package__ in (None, ""):
    # Run as a file (python src/train.py): put the repo root on the path and load the
    # src package, so the package-relative imports below resolve the same way they do
    # under python -m src.train.
    import importlib
    import sys

    here = str(Path(__file__).resolve().parent)
    sys.path[:] = [p for p in sys.path if p not in (here, "")]  # src/ itself must not shadow packages
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    importlib.import_module("src")
    __package__ = "src"

from .dataset import OxfordPetSeg, normalize_image
from .inference import pick_device
from .losses import DiceBCELoss
from .metrics import RunningSegMetrics
from .models import build_model, count_params
from .viz import TrainingLogger


# --------------------------------------------------------------------------- config
def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    for item in overrides or []:
        key, value = item.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(value)
    cfg.setdefault("run_name", cfg["model"]["name"])
    return cfg


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    return apply_overrides(yaml.safe_load(Path(path).read_text()), overrides)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def warmup_cosine(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * progress))


# --------------------------------------------------------------------------- resumable data order
class ResumableSampler(Sampler[int]):
    """Shuffles with a seed derived from the epoch number, so the order of any epoch can be
    recreated after a restart. set_epoch(epoch, skip) starts that epoch `skip` samples in, so
    batches already trained on are skipped without loading them."""

    def __init__(self, n: int, seed: int):
        self.n, self.seed = n, seed
        self.epoch, self.skip = 0, 0

    def set_epoch(self, epoch: int, skip: int = 0) -> None:
        self.epoch, self.skip = epoch, skip

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed * 100_003 + self.epoch)
        return iter(torch.randperm(self.n, generator=g)[self.skip:].tolist())

    def __len__(self) -> int:
        return self.n - self.skip


# --------------------------------------------------------------------------- checkpoints
def atomic_save(obj: dict, path: Path) -> None:
    """Write to a temp file, then rename. A crash mid-write leaves the previous file intact."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def rng_state() -> dict:
    s = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        s["cuda"] = torch.cuda.get_rng_state_all()
    return s


def set_rng_state(s: dict) -> None:
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if "cuda" in s and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def truncate_history(path: Path, last_epoch: int) -> None:
    """Drop history rows past the checkpoint (an epoch that was logged but never saved),
    so a resumed run does not log the same epoch twice."""
    if not path.exists():
        return
    rows = [r for r in path.read_text().splitlines() if r.strip() and json.loads(r)["step"] <= last_epoch]
    path.write_text("".join(r + "\n" for r in rows))


# --------------------------------------------------------------------------- evaluation helpers
@torch.no_grad()
def validate(model, loader, criterion, device, amp: bool) -> dict[str, float]:
    model.eval()
    metrics = RunningSegMetrics()
    total, n = 0.0, 0
    for batch in loader:
        x, y = batch["image"].to(device), batch["mask"].to(device)
        with torch.autocast(device.type, enabled=amp):
            logits = model(x)
        loss = criterion(logits.float(), y)
        total += loss.item() * x.shape[0]
        n += x.shape[0]
        metrics.update(torch.sigmoid(logits.float()), y)
    return {"val/loss": total / max(n, 1), **{f"val/{k}": v for k, v in metrics.summary().items()}}


@torch.no_grad()
def predict_fixed(model, images: list[np.ndarray], device) -> list[np.ndarray]:
    model.eval()
    x = torch.stack([normalize_image(im) for im in images]).to(device)
    return list(torch.sigmoid(model(x).float())[:, 0].cpu().numpy())


def fmt_time(s: float) -> str:
    s = int(s)
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


# --------------------------------------------------------------------------- training
def main(cfg: dict, resume: dict | None = None) -> dict:
    seed_everything(cfg.get("seed", 0))
    device = pick_device(cfg.get("device"))
    d, t, lg = cfg["data"], cfg["train"], cfg.get("log", {})
    run_dir = Path(lg.get("runs_dir", "runs")) / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    common = dict(root=d["root"], image_size=d.get("image_size", 256), boundary_mode=d.get("boundary_mode", "ignore"),
                  val_fraction=d.get("val_fraction", 0.1), seed=d.get("split_seed", 0))
    train_ds = OxfordPetSeg(split="train", grayscale_p=d.get("grayscale_p", 0.0), **common)
    val_ds = OxfordPetSeg(split="val", **common)
    nw = d.get("num_workers", 4)
    bs = t["batch_size"]
    sampler = ResumableSampler(len(train_ds), cfg.get("seed", 0))
    train_dl = DataLoader(train_ds, batch_size=bs, sampler=sampler, num_workers=nw, drop_last=True,
                          pin_memory=device.type == "cuda", persistent_workers=nw > 0)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw,
                        pin_memory=device.type == "cuda", persistent_workers=nw > 0)
    steps_per_epoch = len(train_ds) // bs

    model = build_model(cfg["model"]).to(device)
    n_params, n_train = count_params(model), count_params(model, trainable_only=True)
    print(f"{cfg['run_name']}: {n_params / 1e6:.2f}M params ({n_train / 1e6:.2f}M trainable) on {device}")
    print(f"train={len(train_ds)} val={len(val_ds)}  {steps_per_epoch} steps/epoch")

    criterion = DiceBCELoss(**cfg.get("loss", {}))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(t["lr"]), weight_decay=float(t.get("weight_decay", 1e-4)))
    total_steps = t["epochs"] * steps_per_epoch
    warmup = int(t.get("warmup_epochs", 1) * steps_per_epoch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: warmup_cosine(s, total_steps, max(1, warmup)))
    amp = bool(t.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=amp)

    # Training position. `epoch` counts completed epochs, `step` the batches done in the next one.
    start_epoch, start_step, running, seen, elapsed = 0, 0, 0.0, 0, 0.0
    best, best_epoch = -1.0, -1
    if resume is not None:
        model.load_state_dict(resume["model"])
        opt.load_state_dict(resume["optimizer"])
        sched.load_state_dict(resume["scheduler"])
        if amp and resume.get("scaler"):
            scaler.load_state_dict(resume["scaler"])
        start_epoch, start_step = resume["epoch"], resume.get("step", 0)
        running, seen, elapsed = resume.get("running", 0.0), resume.get("seen", 0), resume.get("elapsed", 0.0)
        best, best_epoch = resume.get("best", -1.0), resume.get("best_epoch", -1)
        set_rng_state(resume["rng"])
        truncate_history(run_dir / "history.jsonl", start_epoch)
        where = f"epoch {start_epoch + 1}, step {start_step}/{steps_per_epoch}" if start_step else f"epoch {start_epoch + 1}"
        print(f"resumed: continuing at {where} (best val Dice so far {best:.4f})")
        if start_epoch >= t["epochs"]:
            print(f"already finished {t['epochs']} epochs. Extend with --set train.epochs=<more>")

    def checkpoint(epoch_done: int, step: int, epoch_running: float, epoch_seen: int, epoch_elapsed: float):
        atomic_save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                     "scaler": scaler.state_dict() if amp else None, "config": cfg, "epoch": epoch_done,
                     "step": step, "running": epoch_running, "seen": epoch_seen, "elapsed": epoch_elapsed,
                     "best": best, "best_epoch": best_epoch, "val_dice": best, "rng": rng_state()},
                    run_dir / "last.pt")

    logger = TrainingLogger(run_dir, backend=lg.get("backend", "tensorboard"), run_name=cfg["run_name"], config=cfg)
    n_viz = min(lg.get("n_viz", 8), len(val_ds))
    viz_idx = np.linspace(0, len(val_ds) - 1, n_viz).astype(int).tolist()
    viz_pairs = [val_ds.load_pair(i) for i in viz_idx]
    viz_images, viz_gts = [p[0] for p in viz_pairs], [p[1] for p in viz_pairs]
    viz_every = lg.get("viz_every", 2)
    print_every = lg.get("print_every", 20)
    save_every_s = float(t.get("save_every_min", 5)) * 60

    epoch, step, t0 = start_epoch + 1, start_step, time.time() - elapsed
    try:
        for epoch in range(start_epoch + 1, t["epochs"] + 1):
            model.train()
            step = start_step if epoch == start_epoch + 1 else 0
            if step == 0:
                running, seen, elapsed = 0.0, 0, 0.0
            sampler.set_epoch(epoch, skip=step * bs)
            t0, last_save = time.time() - elapsed, time.time()
            for batch in train_dl:
                x, y = batch["image"].to(device, non_blocking=True), batch["mask"].to(device, non_blocking=True)
                with torch.autocast(device.type, enabled=amp):
                    logits = model(x)
                loss = criterion(logits.float(), y)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                if t.get("grad_clip"):
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(params, float(t["grad_clip"]))
                scaler.step(opt)
                scaler.update()
                sched.step()
                running += loss.item() * x.shape[0]
                seen += x.shape[0]
                step += 1
                now = time.time()
                if print_every and step % print_every == 0:
                    rate = (now - t0) / step
                    print(f"  epoch {epoch} step {step}/{steps_per_epoch}  loss {running / seen:.4f}  "
                          f"{rate:.2f}s/step  epoch ETA {fmt_time(rate * (steps_per_epoch - step))}", flush=True)
                if now - last_save > save_every_s and step < steps_per_epoch:
                    checkpoint(epoch - 1, step, running, seen, now - t0)
                    last_save = now

            val = validate(model, val_dl, criterion, device, amp)
            scalars = {"train/loss": running / max(seen, 1), **val, "lr": opt.param_groups[0]["lr"],
                       "time/epoch_s": time.time() - t0}
            logger.log_scalars(scalars, epoch)
            if epoch == 1 or epoch % viz_every == 0 or epoch == t["epochs"]:
                logger.log_val_overlays(viz_images, viz_gts, predict_fixed(model, viz_images, device), epoch)

            if val["val/dice"] > best:
                best, best_epoch = val["val/dice"], epoch
                atomic_save({"model": model.state_dict(), "config": cfg, "epoch": epoch, "val_dice": best},
                            run_dir / "best.pt")
            checkpoint(epoch, 0, 0.0, 0, 0.0)
            print(f"epoch {epoch:3d}  train {scalars['train/loss']:.4f}  val {val['val/loss']:.4f}  "
                  f"dice {val['val/dice']:.4f}  iou {val['val/iou']:.4f}  ({fmt_time(scalars['time/epoch_s'])})"
                  + ("  * best" if best_epoch == epoch else ""), flush=True)
    except KeyboardInterrupt:
        if step > 0:  # mid-epoch, or during validation (step == steps_per_epoch): keep the progress
            checkpoint(epoch - 1, step, running, seen, time.time() - t0)
        print(f"\nstopped. Resume with:  python src/train.py --resume {run_dir / 'last.pt'}")
        logger.close()
        raise SystemExit(130)

    logger.close()
    summary = {"run_name": cfg["run_name"], "epochs": t["epochs"], "best_val_dice": best, "best_epoch": best_epoch,
               "params": n_params, "trainable_params": n_train}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def resolve_resume(config: str | None, resume: str, overrides: list[str]) -> tuple[dict, dict]:
    """--resume with no path means runs/<run_name>/last.pt for the given --config."""
    if resume == "auto":
        if not config:
            raise SystemExit("--resume without a path needs --config to find the run")
        cfg = load_config(config, overrides)
        path = Path(cfg.get("log", {}).get("runs_dir", "runs")) / cfg["run_name"] / "last.pt"
    else:
        path = Path(resume)
    if not path.exists():
        raise SystemExit(f"nothing to resume: {path} does not exist")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if "optimizer" not in state:
        raise SystemExit(f"{path} was written by an older train.py without optimizer state, so it can't resume. "
                         f"Start that run again (it will save resumable checkpoints from now on).")
    cfg = apply_overrides(state["config"], overrides)  # the run's own config, plus any --set changes
    return cfg, state


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="YAML config (optional when --resume is given a path)")
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. train.epochs=2")
    ap.add_argument("--resume", nargs="?", const="auto", default=None,
                    help="continue from runs/<run_name>/last.pt, or from the checkpoint path given")
    a = ap.parse_args()
    if a.resume:
        main(*resolve_resume(a.config, a.resume, a.set))
    elif a.config:
        main(load_config(a.config, a.set))
    else:
        ap.error("--config is required unless you pass --resume <path>")
