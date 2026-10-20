"""Evaluate one or more checkpoints and write every evaluation view.

    python -m src.evaluate --split val                        # every runs/*/best.pt
    python -m src.evaluate --checkpoints runs/segformer/best.pt --split test

results/<split>/<model>/   metrics.json, per_image.csv, worst_k.png, per_breed.png,
                           boundary.png, threshold_sweep.png
results/<split>/           summary.json, model_grid*.png, per_breed.png, boundary.png,
                           threshold_sweep.png, efficiency.png, training_curves.png

Scores are computed at each model's training resolution against the nearest-neighbor
resized ground truth, so every model is scored on identical masks.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import viz
from .dataset import OxfordPetSeg, convert_trimap, normalize_image
from .inference import Predictor, find_checkpoints
from .metrics import dice_per_image, iou_per_image, pixel_accuracy_per_image
from .models import count_params


def evaluate_model(pred: Predictor, ds: OxfordPetSeg, batch_size: int = 16, band: int = 3,
                   threshold: float = 0.5, max_images: int | None = None) -> dict:
    sweep, prof = viz.ThresholdSweep(), viz.BoundaryProfile(threshold=threshold)
    rows = []
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
    for batch in loader:
        probs = pred.predict_model_res(batch["image"])
        masks = batch["mask"]
        binp = (probs >= threshold).long()
        dice, iou = dice_per_image(binp, masks), iou_per_image(binp, masks)
        acc = pixel_accuracy_per_image(binp, masks)
        for i in range(probs.shape[0]):
            p, g = probs[i].numpy(), masks[i].numpy().astype(np.uint8)
            sweep.update(p, g)
            prof.update(p, g)
            rows.append({"name": batch["name"][i], "breed": batch["breed"][i], "species": batch["species"][i],
                         "dice": float(dice[i]), "iou": float(iou[i]), "pixel_acc": float(acc[i])})
        if max_images and len(rows) >= max_images:
            break
    d = np.array([r["dice"] for r in rows])
    bt, bd = sweep.best()
    return {"rows": rows, "sweep": sweep, "profile": prof,
            "metrics": {"dice": float(d.mean()), "dice_median": float(np.median(d)),
                        "iou": float(np.mean([r["iou"] for r in rows])),
                        "pixel_acc": float(np.mean([r["pixel_acc"] for r in rows])),
                        "best_threshold": bt, "dice_at_best_threshold": bd,
                        "n_images": len(rows), **{f"boundary_{k}": v for k, v in prof.split(band).items()}}}


def measure_latency(pred: Predictor, n: int = 20) -> float:
    x = torch.randn(1, 3, pred.image_size, pred.image_size)
    for _ in range(3):
        pred.predict_model_res(x)
    if pred.device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        pred.predict_model_res(x)
    if pred.device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


def main(a) -> None:
    ckpts = ({Path(c).parent.name: Path(c) for c in a.checkpoints} if a.checkpoints
             else find_checkpoints(a.runs_dir))
    if not ckpts:
        raise SystemExit(f"no checkpoints found (looked in {a.runs_dir}/*/best.pt)")
    out = Path(a.out) / a.split
    results, predictors = {}, {}
    for name, path in ckpts.items():
        pred = Predictor(path, a.device)
        predictors[name] = pred
        dcfg = pred.config["data"]
        ds = OxfordPetSeg(dcfg["root"] if not a.data_root else a.data_root, split=a.split,
                          image_size=pred.image_size, boundary_mode=dcfg.get("boundary_mode", "ignore"),
                          val_fraction=dcfg.get("val_fraction", 0.1), seed=dcfg.get("split_seed", 0))
        print(f"[{name}] evaluating {len(ds)} {a.split} images ...")
        r = evaluate_model(pred, ds, a.batch_size, a.band, a.threshold, a.max_images)
        r["metrics"].update(latency_ms=measure_latency(pred), params_m=count_params(pred.model) / 1e6)
        r["ds"] = ds
        results[name] = r
        write_model_outputs(name, pred, r, out / name, a)
        print(f"[{name}] " + "  ".join(f"{k}={v:.4f}" for k, v in r["metrics"].items()
                                     if k in ("dice", "iou", "pixel_acc", "best_threshold", "latency_ms")))

    summary = {n: r["metrics"] for n, r in results.items()}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    if len(results) > 1:
        write_comparison(results, predictors, out, a)
    print(f"wrote {out}")


def write_model_outputs(name: str, pred: Predictor, r: dict, out: Path, a) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(r["metrics"], indent=2))
    (out / "curves.json").write_text(json.dumps({"sweep": r["sweep"].to_dict(), "profile": r["profile"].to_dict()}))
    with (out / "per_image.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(r["rows"][0]))
        w.writeheader()
        w.writerows(r["rows"])
    ds, rows = r["ds"], r["rows"]
    dices = [x["dice"] for x in rows]
    eval_t = ds.transform

    def item(i: int) -> dict:  # re-predict only the selected images
        img, gt = ds.load_pair(i, eval_t)
        prob = pred.predict_model_res(normalize_image(img)[None])[0].numpy()
        return {"image": img, "gt": gt, "prob": prob, "name": f"{rows[i]['name']}"}

    viz.save_figure(viz.worst_k_gallery(dices, item, k=a.k, threshold=a.threshold,
                                        title=f"{name}: worst {a.split} images"), out / "worst_k.png")
    viz.save_figure(viz.plot_per_breed(dices, [x["breed"] for x in rows], [x["species"] for x in rows]),
                    out / "per_breed.png")
    viz.save_figure(viz.plot_boundary_interior(r["profile"], a.band), out / "boundary.png")
    viz.save_figure(viz.plot_threshold_sweep(r["sweep"]), out / "threshold_sweep.png")
    run_hist = Path(pred_history(pred, a))
    if run_hist.exists():
        viz.save_figure(viz.plot_training_curves(run_hist), out / "training_curves.png")


def pred_history(pred: Predictor, a) -> str:
    return str(Path(pred.config.get("log", {}).get("runs_dir", a.runs_dir)) / pred.config["run_name"] / "history.jsonl")


def write_comparison(results: dict, predictors: dict, out: Path, a) -> None:
    names = list(results)
    first = results[names[0]]
    breeds = [x["breed"] for x in first["rows"]]
    species = [x["species"] for x in first["rows"]]
    by_name = {n: {x["name"]: x["dice"] for x in r["rows"]} for n, r in results.items()}
    common = [x["name"] for x in first["rows"] if all(x["name"] in by_name[n] for n in names)]
    keep = [i for i, x in enumerate(first["rows"]) if x["name"] in set(common)]
    dices = {n: np.array([by_name[n][first["rows"][i]["name"]] for i in keep]) for n in names}
    viz.save_figure(viz.plot_per_breed(dices, [breeds[i] for i in keep], [species[i] for i in keep]),
                    out / "per_breed.png")
    viz.save_figure(viz.plot_boundary_interior({n: r["profile"] for n, r in results.items()}, a.band),
                    out / "boundary.png")
    viz.save_figure(viz.plot_threshold_sweep({n: r["sweep"] for n, r in results.items()}), out / "threshold_sweep.png")
    viz.save_figure(viz.plot_efficiency({n: r["metrics"] for n, r in results.items()}), out / "efficiency.png")
    hists = {n: pred_history(predictors[n], a) for n in names}
    hists = {n: h for n, h in hists.items() if Path(h).exists()}
    if hists:
        viz.save_figure(viz.plot_training_curves(hists), out / "training_curves.png")

    # Model grid: a few typical images plus the ones that are hardest on average across models.
    mean_d = np.mean([dices[n] for n in names], axis=0)
    n_rows = a.grid_n
    typical = np.linspace(0, len(keep) - 1, max(1, n_rows - n_rows // 2)).astype(int).tolist()
    hard = [int(i) for i in np.argsort(mean_d) if int(i) not in typical][: n_rows // 2]
    ds = first["ds"]
    images, gts, labels = [], [], []
    preds = {n: [] for n in names}
    for j in typical + hard:
        idx = keep[j]
        img, trimap = ds.load_raw(idx)
        images.append(img)
        gts.append(convert_trimap(trimap, ds.boundary_mode))
        labels.append(("hard: " if j in hard else "") + ds.records[idx].name)
        for n in names:
            preds[n].append(predictors[n].predict(img))
    for view in ("overlay", "error"):
        viz.save_figure(viz.model_grid(images, gts, preds, a.threshold, view=view, row_labels=labels),
                        out / f"model_grid_{view}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="*", help="default: every runs/*/best.pt")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--data-root", help="override the dataset root stored in the checkpoint config")
    ap.add_argument("--out", default="results")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--band", type=int, default=3, help="px from the boundary counted as 'near boundary'")
    ap.add_argument("--k", type=int, default=20, help="images in the worst-k gallery")
    ap.add_argument("--grid-n", type=int, default=6, help="rows in the model grid")
    ap.add_argument("--max-images", type=int)
    ap.add_argument("--device")
    main(ap.parse_args())
