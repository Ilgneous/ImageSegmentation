"""Config-driven training, one entry point for all five models.

    python -m src.train --config configs/unet_scratch.yaml
    python -m src.train --config configs/segformer.yaml --set train.epochs=5 data.root=data/fake-pet

Writes runs/<run_name>/{best.pt, last.pt, history.jsonl, overlays/, summary.json, events.*}.
Watch live with:  tensorboard --logdir runs
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

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


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    for item in overrides or []:
        key, value = item.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(value)
    cfg.setdefault("run_name", cfg["model"]["name"])
    return cfg


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


def main(cfg: dict) -> dict:
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
    train_dl = DataLoader(train_ds, batch_size=t["batch_size"], shuffle=True, num_workers=nw, drop_last=True,
                          pin_memory=device.type == "cuda", persistent_workers=nw > 0)
    val_dl = DataLoader(val_ds, batch_size=t["batch_size"], shuffle=False, num_workers=nw,
                        pin_memory=device.type == "cuda", persistent_workers=nw > 0)

    model = build_model(cfg["model"]).to(device)
    n_params, n_train = count_params(model), count_params(model, trainable_only=True)
    print(f"{cfg['run_name']}: {n_params / 1e6:.2f}M params ({n_train / 1e6:.2f}M trainable) on {device}")
    print(f"train={len(train_ds)} val={len(val_ds)}")

    criterion = DiceBCELoss(**cfg.get("loss", {}))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(t["lr"]), weight_decay=float(t.get("weight_decay", 1e-4)))
    total_steps = t["epochs"] * len(train_dl)
    warmup = int(t.get("warmup_epochs", 1) * len(train_dl))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: warmup_cosine(s, total_steps, max(1, warmup)))
    amp = bool(t.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=amp)

    logger = TrainingLogger(run_dir, backend=lg.get("backend", "tensorboard"), run_name=cfg["run_name"], config=cfg)
    n_viz = min(lg.get("n_viz", 8), len(val_ds))
    viz_idx = np.linspace(0, len(val_ds) - 1, n_viz).astype(int).tolist()
    viz_pairs = [val_ds.load_pair(i) for i in viz_idx]
    viz_images, viz_gts = [p[0] for p in viz_pairs], [p[1] for p in viz_pairs]
    viz_every = lg.get("viz_every", 2)

    best, best_epoch = -1.0, -1
    for epoch in range(1, t["epochs"] + 1):
        model.train()
        t0, running, seen = time.time(), 0.0, 0
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

        val = validate(model, val_dl, criterion, device, amp)
        scalars = {"train/loss": running / max(seen, 1), **val, "lr": opt.param_groups[0]["lr"],
                   "time/epoch_s": time.time() - t0}
        logger.log_scalars(scalars, epoch)
        if epoch == 1 or epoch % viz_every == 0 or epoch == t["epochs"]:
            logger.log_val_overlays(viz_images, viz_gts, predict_fixed(model, viz_images, device), epoch)

        state = {"model": model.state_dict(), "config": cfg, "epoch": epoch, "val_dice": val["val/dice"]}
        torch.save(state, run_dir / "last.pt")
        if val["val/dice"] > best:
            best, best_epoch = val["val/dice"], epoch
            torch.save(state, run_dir / "best.pt")
        print(f"epoch {epoch:3d}  train {scalars['train/loss']:.4f}  val {val['val/loss']:.4f}  "
              f"dice {val['val/dice']:.4f}  iou {val['val/iou']:.4f}  ({scalars['time/epoch_s']:.0f}s)")

    logger.close()
    summary = {"run_name": cfg["run_name"], "best_val_dice": best, "best_epoch": best_epoch,
               "params": n_params, "trainable_params": n_train}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. train.epochs=2")
    a = ap.parse_args()
    main(load_config(a.config, a.set))
