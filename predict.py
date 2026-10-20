"""Run a trained model on new images or a video.

    python predict.py --model segformer --input img.jpg --out overlay.png
    python predict.py --model all --input img.jpg --out grid.png                    # every trained model
    python predict.py --model unet_resnet34 --input img.jpg --view panel --gt trimap.png --out panel.png
    python predict.py --model segformer --input photos/ --out preds/                # a folder
    python predict.py --model all --input clip.mp4 --out clip_seg.mp4               # a video

--model is a run name under runs/ (runs/<name>/best.pt), or 'all'. --checkpoint takes a path instead.
Views: overlay | binary | heatmap | error (needs --gt) | panel (all of them in one figure).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from src import viz
from src.dataset import convert_trimap
from src.inference import Predictor, find_checkpoints, load_image
from src.video import is_video, segment_video

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def load_gt(path: str | None) -> np.ndarray | None:
    """Accepts an Oxford trimap (values 1/2/3) or a binary mask (0/1 or 0/255)."""
    if not path:
        return None
    m = np.array(Image.open(path).convert("L"))
    vals = set(np.unique(m).tolist())
    if vals <= {1, 2, 3}:
        return convert_trimap(m)
    return (m > (127 if max(vals) > 1 else 0)).astype(np.uint8)


def load_predictors(model: str | None, checkpoint: str | None, runs_dir: str, device: str | None) -> dict:
    if checkpoint:
        return {Path(checkpoint).parent.name: Predictor(checkpoint, device)}
    available = find_checkpoints(runs_dir)
    if not available:
        raise SystemExit(f"no trained models in {runs_dir}/ (expected {runs_dir}/<name>/best.pt)")
    if model == "all":
        return {n: Predictor(p, device) for n, p in available.items()}
    if model not in available:
        raise SystemExit(f"unknown model {model!r}; trained: {', '.join(available)}")
    return {model: Predictor(available[model], device)}


def render(image, prob, gt, view: str, threshold: float) -> np.ndarray:
    if view == "overlay":
        return viz.overlay(image, prob, threshold=threshold)
    if view == "binary":
        return viz.binary_mask(prob, threshold)
    if view == "heatmap":
        return viz.prob_heatmap(prob, image)
    if view == "error":
        if gt is None:
            raise SystemExit("--view error needs --gt")
        return viz.error_map(prob, viz._resize_to(gt, image.shape[:2], True), threshold, image)
    raise ValueError(view)


def run_image(path: Path, out: Path, predictors: dict, gt, view: str, threshold: float) -> None:
    image = load_image(path)
    probs = {n: p.predict(image) for n, p in predictors.items()}
    out.parent.mkdir(parents=True, exist_ok=True)
    if view == "panel" or len(probs) > 1:
        if len(probs) == 1:
            (n, prob), = probs.items()
            fig = viz.show_prediction(image, prob, gt, threshold, title=f"{n}: {path.name}")
        else:
            v = "error" if (gt is not None and view in ("error", "panel")) else ("overlay" if view == "panel" else view)
            fig = viz.model_grid([image], [gt] if gt is not None else None,
                                 {n: [p] for n, p in probs.items()}, threshold, view=v)
        viz.save_figure(fig, out)
    else:
        (n, prob), = probs.items()
        Image.fromarray(render(image, prob, gt, view, threshold)).save(out)
    for n, prob in probs.items():
        msg = f"{n}: pet covers {np.mean(prob >= threshold):.1%} of {path.name}"
        if gt is not None:
            g = viz._resize_to(gt, prob.shape, True)
            msg += f", Dice {viz.dice_np(prob >= threshold, g):.4f}"
        print(msg)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="all", help="run name under --runs-dir, or 'all'")
    ap.add_argument("--checkpoint", help="explicit checkpoint path (overrides --model)")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--input", required=True, help="image, folder of images, or video")
    ap.add_argument("--out", required=True)
    ap.add_argument("--view", default="overlay", choices=["overlay", "binary", "heatmap", "error", "panel"])
    ap.add_argument("--gt", help="optional ground-truth trimap or binary mask (enables error map and Dice)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--max-frames", type=int, help="video: stop after N frames")
    ap.add_argument("--device")
    a = ap.parse_args()

    predictors = load_predictors(a.model, a.checkpoint, a.runs_dir, a.device)
    src, out = Path(a.input), Path(a.out)
    if is_video(src):
        out.parent.mkdir(parents=True, exist_ok=True)
        stab = segment_video(src, out, predictors, a.threshold, max_frames=a.max_frames)
        for n, s in stab.items():
            print(f"{n}: mean consecutive-frame IoU {s:.4f}")
        print(f"wrote {out}")
    elif src.is_dir():
        files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        for p in files:
            run_image(p, out / f"{p.stem}_{a.view}.png", predictors, None, a.view, a.threshold)
    else:
        run_image(src, out, predictors, load_gt(a.gt), a.view, a.threshold)


if __name__ == "__main__":
    main()
