"""Visualization tools for developing and debugging the pet segmentation models.

Every view is a plain function. Image renderers return HxWx3 uint8 arrays (usable in Gradio,
TensorBoard, OpenCV video writers). Figure builders return a matplotlib Figure (display in a
notebook, or write with save_figure).

Conventions
    image  uint8 HxWx3 RGB
    gt     uint8 HxW, 0 background / 1 pet / 255 ignored boundary
    prob   float HxW in [0, 1] (sigmoid output, before thresholding)
    pred   binary HxW, or a prob that gets thresholded

Checklist -> function
    Data sanity      show_sample, show_augmented
    Prediction       overlay, binary_mask, error_map, prob_heatmap, show_prediction, model_grid
    Evaluation       worst_k_gallery, plot_per_breed, BoundaryProfile + plot_boundary_interior,
                     ThresholdSweep + plot_threshold_sweep, plot_efficiency
    Training         TrainingLogger (TensorBoard or W&B, plus history.jsonl and overlay PNGs),
                     plot_training_curves
"""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from scipy.ndimage import distance_transform_edt

if __package__ in (None, ""):
    # Run as a file (python src/viz.py): put the repo root on the path and load the
    # src package, so the package-relative imports below resolve the same way they do
    # under python -m src.viz.
    import importlib
    import sys

    here = str(Path(__file__).resolve().parent)
    sys.path[:] = [p for p in sys.path if p not in (here, "")]  # src/ itself must not shadow packages
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    importlib.import_module("src")
    __package__ = "src"

from .dataset import IGNORE_INDEX, OxfordPetSeg, convert_trimap, get_transforms

# --------------------------------------------------------------------------- colors and style
PET_COLOR = (255, 64, 129)          # overlay tint, chosen to stand out against fur tones
TP_COLOR = (46, 184, 92)            # green
FP_COLOR = (230, 57, 70)            # red
FN_COLOR = (52, 120, 230)           # blue
IGNORE_COLOR = (160, 160, 160)      # gray
FP_COLOR_HEX = "#%02x%02x%02x" % FP_COLOR
FN_COLOR_HEX = "#%02x%02x%02x" % FN_COLOR
TRIMAP_COLORS = {1:(42, 120, 214), 2: (28, 28, 28), 3: (237, 161, 0)}
TRIMAP_LABELS = {1: "1 pet", 2: "2 background", 3: "3 boundary"}

# Categorical slots in fixed order. A model keeps its color in every chart.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MODEL_ORDER = ["unet_scratch", "unet_resnet34", "deeplabv3plus", "segformer", "dino_linear", "sam3"]
SPECIES_COLORS = {"cat": SERIES[0], "dog": SERIES[1]}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"

_RC = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK_2, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 10, "font.size": 9.5, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "lines.linewidth": 2.0, "lines.markersize": 6,
}


def _style():
    return plt.rc_context(_RC)


def model_color(name: str, index: int = 0) -> str:
    key = name.split("/")[-1].lower()
    for i, m in enumerate(MODEL_ORDER):
        if key.startswith(m):
            return SERIES[i]
    return SERIES[index % len(SERIES)]


def save_figure(fig: Figure, path: str | Path, dpi: int = 150) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- array helpers
def _np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    while x.ndim > 2 and x.shape[0] == 1:  # (1,H,W) or (1,1,H,W)
        x = x[0]
    return x


def to_binary(pred, threshold: float = 0.5) -> np.ndarray:
    """Probability, logits-free binary, or a 0/1/255 GT mask -> bool HxW (ignore counts as False)."""
    p = _np(pred)
    if p.dtype == bool:
        return p
    if np.issubdtype(p.dtype, np.floating):
        return p >= threshold
    return p == 1


def dice_np(pred_bin: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    valid = gt != IGNORE_INDEX
    p, g = pred_bin & valid, (gt == 1) & valid
    tp = np.count_nonzero(p & g)
    return float((2 * tp + eps) / (np.count_nonzero(p) + np.count_nonzero(g) + eps))


def _resize_to(arr: np.ndarray, hw: tuple[int, int], nearest: bool) -> np.ndarray:
    if arr.shape[:2] == hw:
        return arr
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.resize(arr, (hw[1], hw[0]), interpolation=interp)


# --------------------------------------------------------------------------- image renderers
def overlay(image: np.ndarray, mask, color: tuple[int, int, int] = PET_COLOR, alpha: float = 0.45,
            threshold: float = 0.5, contour: bool = True) -> np.ndarray:
    """Semi-transparent colored mask on top of the image, with an outline for crisp edges."""
    image = _np(image).astype(np.uint8)
    m = _resize_to(to_binary(mask, threshold).astype(np.uint8), image.shape[:2], nearest=True).astype(bool)
    out = image.astype(np.float32)
    out[m] = (1 - alpha) * out[m] + alpha * np.array(color, np.float32)
    out = out.clip(0, 255).astype(np.uint8)
    if contour and m.any():
        thickness = max(1, round(min(image.shape[:2]) / 200))
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cnts, -1, color, thickness)
    return out


def binary_mask(mask, threshold: float = 0.5) -> np.ndarray:
    """Black and white view. Ignored pixels (255 in a GT mask) render gray."""
    m = _np(mask)
    out = np.zeros((*m.shape, 3), np.uint8)
    out[to_binary(m, threshold)] = 255
    if not np.issubdtype(m.dtype, np.floating) and m.dtype != bool:
        out[m == IGNORE_INDEX] = IGNORE_COLOR
    return out


def trimap_to_rgb(trimap: np.ndarray) -> np.ndarray:
    """Raw Oxford trimap with each value in its own color. Unexpected values show magenta."""
    t = _np(trimap)
    out = np.full((*t.shape, 3), (255, 0, 255), np.uint8)
    for v, c in TRIMAP_COLORS.items():
        out[t == v] = c
    return out


def error_map(pred, gt: np.ndarray, threshold: float = 0.5, image: np.ndarray | None = None,
              dim: float = 0.35) -> np.ndarray:
    """Per-pixel outcome: TP green, FP red, FN blue, ignored boundary gray.
    True negatives show the dimmed grayscale image when `image` is given, otherwise black."""
    gt = _np(gt)
    p = _resize_to(to_binary(pred, threshold).astype(np.uint8), gt.shape, nearest=True).astype(bool)
    g = gt == 1
    ign = gt == IGNORE_INDEX
    if image is not None:
        img = _resize_to(_np(image).astype(np.uint8), gt.shape, nearest=False)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) * dim
        out = np.repeat(gray[..., None], 3, axis=2).astype(np.uint8)
    else:
        out = np.zeros((*gt.shape, 3), np.uint8)
    out[p & g & ~ign] = TP_COLOR
    out[p & ~g & ~ign] = FP_COLOR
    out[~p & g & ~ign] = FN_COLOR
    out[ign] = IGNORE_COLOR
    return out


def error_counts(pred, gt: np.ndarray, threshold: float = 0.5) -> dict[str, int]:
    gt = _np(gt)
    p = to_binary(pred, threshold)
    valid = gt != IGNORE_INDEX
    g = gt == 1
    return {"tp": int(np.count_nonzero(p & g & valid)), "fp": int(np.count_nonzero(p & ~g & valid)),
            "fn": int(np.count_nonzero(~p & g & valid)), "ignored": int(np.count_nonzero(~valid))}


def prob_heatmap(prob, image: np.ndarray | None = None, alpha: float = 0.6, cmap: str = "magma",
                 mode: str = "prob") -> np.ndarray:
    """Probability before thresholding. mode='uncertainty' shows 1 - |2p - 1| (bright = unsure)."""
    p = _np(prob).astype(np.float32).clip(0, 1)
    v = 1 - np.abs(2 * p - 1) if mode == "uncertainty" else p
    rgb = (plt.get_cmap(cmap)(v)[..., :3] * 255).astype(np.uint8)
    if image is None:
        return rgb
    img = _resize_to(_np(image).astype(np.uint8), p.shape, nearest=False)
    return cv2.addWeighted(img, 1 - alpha, rgb, alpha, 0)


def label_tile(img: np.ndarray, text: str, height: int = 22) -> np.ndarray:
    """Adds a dark caption banner on top of an image (for numpy grids, TensorBoard, video)."""
    banner = np.full((height, img.shape[1], 3), 24, np.uint8)
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, height / 30
    (tw, _), _ = cv2.getTextSize(text, font, scale, 1)
    scale *= min(1.0, (img.shape[1] - 8) / max(tw, 1))  # shrink to fit narrow tiles
    cv2.putText(banner, text, (4, int(height * 0.72)), font, scale, (240, 240, 240), 1, cv2.LINE_AA)
    return np.concatenate([banner, img], axis=0)


def tile(images: Sequence[np.ndarray], cols: int, pad: int = 2, fill: int = 255) -> np.ndarray:
    """Pack equally sized RGB images into a grid with a thin separator."""
    h, w = images[0].shape[:2]
    rows = int(np.ceil(len(images) / cols))
    grid = np.full((rows * (h + pad) - pad, cols * (w + pad) - pad, 3), fill, np.uint8)
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        grid[r * (h + pad): r * (h + pad) + h, c * (w + pad): c * (w + pad) + w] = _resize_to(im, (h, w), False)
    return grid


def prediction_strip(image: np.ndarray, prob, gt: np.ndarray | None = None, threshold: float = 0.5) -> np.ndarray:
    """One row: overlay | error map | heatmap, captioned with Dice when GT is available."""
    d = f"Dice {dice_np(to_binary(prob, threshold), _np(gt)):.3f}" if gt is not None else "no GT"
    panels = [label_tile(overlay(image, prob, threshold=threshold), d)]
    if gt is not None:
        panels.append(label_tile(error_map(prob, gt, threshold, image), "error map"))
    panels.append(label_tile(prob_heatmap(prob, image), "probability"))
    return tile(panels, cols=len(panels))


def _error_legend(ax, loc="lower center", y=-0.02, ncol=4):
    handles = [Patch(color=np.array(c) / 255, label=l) for c, l in
               ((TP_COLOR, "true positive"), (FP_COLOR, "false positive"),
                (FN_COLOR, "false negative"), (IGNORE_COLOR, "ignored boundary"))]
    ax.figure.legend(handles=handles, loc=loc, ncol=ncol, bbox_to_anchor=(0.5, y), fontsize=9)


def _imshow(ax, img, title: str | None = None, color: str = INK):
    ax.imshow(img)
    ax.set_xticks([]), ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=9.5, fontweight="normal", color=color)


# --------------------------------------------------------------------------- data sanity checks
def show_sample(dataset: OxfordPetSeg, idx: int) -> Figure:
    """Image, raw trimap, converted mask, and the mask overlaid on the image (alignment check).
    The title turns red if the image and trimap sizes disagree or the trimap has unexpected values."""
    image, trimap = dataset.load_raw(idx)
    mask = convert_trimap(trimap, dataset.boundary_mode)
    r = dataset.records[idx]
    values = sorted(np.unique(trimap).tolist())
    problems = []
    if image.shape[:2] != trimap.shape[:2]:
        problems.append(f"size mismatch {image.shape[:2]} vs {trimap.shape[:2]}")
    if set(values) - {1, 2, 3}:
        problems.append(f"unexpected trimap values {values}")

    with _style():
        fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
        _imshow(axes[0], image, f"image {image.shape[1]}x{image.shape[0]}")
        _imshow(axes[1], trimap_to_rgb(trimap), f"raw trimap, values {values}")
        axes[1].legend(handles=[Patch(color=np.array(TRIMAP_COLORS[v]) / 255, label=TRIMAP_LABELS[v])
                                for v in (1, 2, 3)], loc="lower center", bbox_to_anchor=(0.5, -0.2),
                       ncol=3, fontsize=8)
        _imshow(axes[2], binary_mask(mask), f"converted mask (boundary -> {dataset.boundary_mode})")
        _imshow(axes[3], overlay(image, mask), "mask on image")
        title = f"{r.name}   breed: {r.breed}   species: {r.species}   split: {dataset.split}"
        if problems:
            title += "   PROBLEM: " + "; ".join(problems)
        fig.suptitle(title, color="#c62828" if problems else INK, fontsize=11)
        fig.tight_layout()
    return fig


def show_augmented(dataset: OxfordPetSeg, idx: int, n: int = 8, transform=None,
                   image_size: int | None = None, seed: int | None = 0) -> Figure:
    """The same sample through n augmentation passes. Top row: augmented image with the
    augmented mask outlined on it. Bottom row: the augmented mask alone. If the outline
    drifts off the pet, image and mask transforms are out of sync."""
    if transform is None:
        size = image_size or _infer_size(dataset)
        transform = get_transforms(train=True, size=size)
    if seed is not None:
        import random
        random.seed(seed), np.random.seed(seed)
        if hasattr(transform, "set_random_seed"):
            transform.set_random_seed(seed)
    pairs = [dataset.load_pair(idx, transform) for _ in range(n)]
    with _style():
        fig, axes = plt.subplots(2, n, figsize=(2.1 * n, 5.0))
        axes = np.atleast_2d(axes).reshape(2, n)
        for j, (img, m) in enumerate(pairs):
            frac = np.mean(m == 1)
            _imshow(axes[0, j], overlay(img, m, alpha=0.25), f"pass {j + 1}")
            _imshow(axes[1, j], binary_mask(m), f"pet {frac:.0%}")
        fig.suptitle(f"{dataset.records[idx].name}: {n} augmentation passes "
                     f"(outline must stay on the pet)", fontsize=11)
        fig.tight_layout()
    return fig


def _infer_size(dataset: OxfordPetSeg) -> int:
    for t in getattr(dataset.transform, "transforms", []):
        for attr in ("height", "size"):
            v = getattr(t, attr, None)
            if v is not None:
                return int(v[0] if isinstance(v, (tuple, list)) else v)
    return 256


# --------------------------------------------------------------------------- prediction views
def show_prediction(image: np.ndarray, prob, gt: np.ndarray | None = None, threshold: float = 0.5,
                    title: str | None = None) -> Figure:
    """Image | overlay | binary mask | probability heatmap | error map (when GT is given)."""
    image = _np(image)
    prob = _resize_to(_np(prob).astype(np.float32), image.shape[:2], nearest=False)
    panels = [(image, "image"), (overlay(image, prob, threshold=threshold), "overlay"),
              (binary_mask(prob, threshold), f"binary mask (t={threshold:.2f})"),
              (prob_heatmap(prob), "probability")]
    if gt is not None:
        gt = _resize_to(_np(gt).astype(np.uint8), image.shape[:2], nearest=True)
        d = dice_np(to_binary(prob, threshold), gt)
        panels.append((error_map(prob, gt, threshold, image), f"error map, Dice {d:.3f}"))
    with _style():
        fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.9))
        for ax, (img, t) in zip(axes, panels):
            _imshow(ax, img, t)
        sm = plt.cm.ScalarMappable(cmap="magma", norm=plt.Normalize(0, 1))
        fig.colorbar(sm, ax=axes[3], fraction=0.046, pad=0.02).ax.tick_params(labelsize=8)
        if gt is not None:
            _error_legend(axes[0], y=-0.04)
        if title:
            fig.suptitle(title, fontsize=11)
        fig.tight_layout()
    return fig


def model_grid(images: Sequence[np.ndarray], gts: Sequence[np.ndarray] | None,
               preds: Mapping[str, Sequence], threshold: float = 0.5, view: str = "overlay",
               row_labels: Sequence[str] | None = None) -> Figure:
    """One row per image. Columns: image, ground truth, then one column per model.
    view: 'overlay' | 'error' | 'binary' | 'heatmap'. Each model cell is captioned with its Dice."""
    names = list(preds)
    ncols = 1 + (gts is not None) + len(names)
    nrows = len(images)
    render = {
        "overlay": lambda im, p, g: overlay(im, p, threshold=threshold),
        "error": lambda im, p, g: error_map(p, g, threshold, im) if g is not None else overlay(im, p, threshold=threshold),
        "binary": lambda im, p, g: binary_mask(p, threshold),
        "heatmap": lambda im, p, g: prob_heatmap(p, im),
    }[view]
    with _style():
        fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 2.55 * nrows + 0.4), squeeze=False)
        for i, img in enumerate(images):
            img = _np(img)
            g = None if gts is None else _resize_to(_np(gts[i]).astype(np.uint8), img.shape[:2], True)
            col = 0
            _imshow(axes[i, col], img)
            if row_labels:
                axes[i, col].set_ylabel(row_labels[i], fontsize=9, color=INK)
            col += 1
            if g is not None:
                _imshow(axes[i, col], binary_mask(g))
                col += 1
            for j, name in enumerate(names):
                p = _resize_to(_np(preds[name][i]).astype(np.float32), img.shape[:2], False)
                ax = axes[i, col + j]
                _imshow(ax, render(img, p, g))
                if g is not None:
                    d = dice_np(to_binary(p, threshold), g)
                    ax.text(0.03, 0.97, f"{d:.3f}", transform=ax.transAxes, va="top", ha="left",
                            fontsize=8.5, color="white",
                            bbox=dict(boxstyle="round,pad=0.2", fc="#000000aa", ec="none"))
        headers = ["image"] + (["ground truth"] if gts is not None else []) + names
        for c, h in enumerate(headers):
            axes[0, c].set_title(h, fontsize=10, fontweight="bold",
                                 color=INK if c < ncols - len(names) else model_color(h, c))
        if view == "error" and gts is not None:
            _error_legend(axes[0, 0], y=0.0)
        fig.tight_layout(rect=(0, 0.03 if view == "error" else 0, 1, 1))
    return fig


# --------------------------------------------------------------------------- evaluation views
def worst_k_gallery(dices: Sequence[float], items: Sequence[Mapping] | Callable[[int], Mapping],
                    k: int = 20, cols: int = 5, threshold: float = 0.5, view: str = "error",
                    title: str = "Worst validation images by Dice") -> Figure:
    """Lowest-Dice images first. `items[i]` (or items(i)) returns a dict with
    image, gt, prob and optionally name. Using a callable keeps memory flat: only the k
    selected images are loaded or re-predicted."""
    dices = np.asarray(dices, dtype=float)
    order = np.argsort(dices)[:k]
    get = items if callable(items) else (lambda i: items[i])
    rows = int(np.ceil(len(order) / cols))
    with _style():
        fig, axes = plt.subplots(rows, cols, figsize=(2.9 * cols, 3.5 * rows + 0.8), squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for rank, i in enumerate(order):
            it = get(int(i))
            img, gt, prob = _np(it["image"]), _np(it["gt"]), _np(it["prob"])
            ax = axes.flat[rank]
            ax.axis("on")
            pic = error_map(prob, gt, threshold, img) if view == "error" else overlay(img, prob, threshold=threshold)
            _imshow(ax, pic, f"#{rank + 1}  Dice {dices[i]:.3f}\n{it.get('name', i)}")
        if view == "error":
            _error_legend(axes[0, 0], y=0.0)
        fig.suptitle(f"{title} (bottom {len(order)} of {len(dices)}, mean {dices.mean():.3f})", fontsize=12)
        fig.tight_layout(rect=(0, 0.03, 1, 0.97), h_pad=1.5)
    return fig


def species_from_breed(breed: str) -> str:
    """Oxford naming convention: cat breeds are capitalized, dog breeds are lowercase."""
    return "cat" if breed[:1].isupper() else "dog"


def per_breed_table(dices: Sequence[float], breeds: Sequence[str]) -> dict[str, dict]:
    dices, breeds = np.asarray(dices, float), np.asarray(breeds)
    out = {}
    for b in sorted(set(breeds.tolist())):
        v = dices[breeds == b]
        out[b] = {"mean": float(v.mean()), "sem": float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0,
                  "n": int(len(v)), "species": species_from_breed(b)}
    return out


def plot_per_breed(dices: Sequence[float] | Mapping[str, Sequence[float]], breeds: Sequence[str],
                   species: Sequence[str] | None = None) -> Figure:
    """One model: horizontal bars per breed (sorted, colored by species, with standard-error
    whiskers) plus a cat vs dog panel. Several models: a breed x model heatmap plus grouped
    cat vs dog bars."""
    multi = isinstance(dices, Mapping)
    results = {k: np.asarray(v, float) for k, v in (dices.items() if multi else {"model": dices}.items())}
    breeds = np.asarray(breeds)
    species = np.asarray(species) if species is not None else np.array([species_from_breed(b) for b in breeds])
    spec_of = {b: s for b, s in zip(breeds, species)}

    with _style():
        if not multi:
            d = results["model"]
            table = per_breed_table(d, breeds)
            names = sorted(table, key=lambda b: table[b]["mean"])
            fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, max(4, 0.24 * len(names) + 1.2)),
                                          gridspec_kw={"width_ratios": [4, 1.2]})
            y = np.arange(len(names))
            means = [table[b]["mean"] for b in names]
            cols = [SPECIES_COLORS[spec_of[b]] for b in names]
            ax.barh(y, means, height=0.72, color=cols, xerr=[table[b]["sem"] for b in names],
                    error_kw=dict(ecolor=INK_2, elinewidth=0.8, capsize=0))
            ax.set_yticks(y, [f"{b} (n={table[b]['n']})" for b in names], fontsize=8)
            lo = max(0.0, min(means) - 0.08)
            ax.set_xlim(lo, 1.0)
            ax.set_xlabel("mean per-image Dice")
            ax.grid(axis="y", visible=False)
            for s, c in SPECIES_COLORS.items():
                m = d[species == s]
                if len(m):
                    ax.axvline(m.mean(), color=c, ls="--", lw=1.2)
            ax.legend(handles=[Patch(color=c, label=s) for s, c in SPECIES_COLORS.items()],
                      loc="lower right")
            ax.set_title("Dice by breed (dashed: species mean)", loc="left")
            sp = [s for s in ("cat", "dog") if np.any(species == s)]
            vals = [d[species == s].mean() for s in sp]
            ax2.bar(sp, vals, color=[SPECIES_COLORS[s] for s in sp], width=0.6)
            for i, v in enumerate(vals):
                ax2.text(i, v + 0.003, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
            ax2.set_ylim(max(0, min(vals) - 0.1), min(1.0, max(vals) + 0.04))
            ax2.set_title("cats vs dogs", loc="left")
            ax2.grid(axis="x", visible=False)
        else:
            models = list(results)
            ub = sorted(set(breeds.tolist()))
            mat = np.array([[results[m][breeds == b].mean() for m in models] for b in ub])
            order = np.argsort(mat.mean(1))
            mat, ub = mat[order], [ub[i] for i in order]
            fig, (ax, ax2) = plt.subplots(1, 2, figsize=(4 + 1.3 * len(models), max(5, 0.26 * len(ub) + 1.5)),
                                          gridspec_kw={"width_ratios": [len(models), 2.4]})
            im = ax.imshow(mat, cmap="Blues", aspect="auto", vmin=max(0, mat.min() - 0.02), vmax=1)
            ax.grid(False)
            ax.set_xticks(range(len(models)), models, rotation=30, ha="right")
            ax.set_yticks(range(len(ub)), ub, fontsize=8)
            for t, b in zip(ax.get_yticklabels(), ub):
                t.set_color(SPECIES_COLORS[spec_of[b]])
            thresh = mat.min() + 0.6 * (mat.max() - mat.min())
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if mat[i, j] > thresh else INK)
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="mean Dice")
            ax.set_title("Dice by breed (label color: cat / dog)", loc="left")
            sp = [s for s in ("cat", "dog") if np.any(species == s)]
            w = 0.8 / len(models)
            x = np.arange(len(sp))
            allv = []
            for j, m in enumerate(models):
                v = [results[m][species == s].mean() for s in sp]
                allv += v
                ax2.bar(x + (j - (len(models) - 1) / 2) * w, v, width=w * 0.92,
                        color=model_color(m, j), label=m)
            ax2.set_xticks(x, sp)
            ax2.set_ylim(max(0, min(allv) - 0.08), min(1.0, max(allv) + 0.03))
            ax2.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=1)
            ax2.set_title("cats vs dogs", loc="left")
            ax2.grid(axis="x", visible=False)
        fig.tight_layout()
    return fig


class BoundaryProfile:
    """Accumulates errors as a function of distance (pixels) from the ground-truth boundary.

    Distance is measured from each valid pixel to the nearest pixel of a different GT class
    (with boundary_mode='ignore', that is the edge of the ignored trimap band)."""

    def __init__(self, max_dist: int = 40, threshold: float = 0.5):
        self.max_dist, self.threshold = max_dist, threshold
        n = max_dist + 1
        self.pixels = np.zeros(n, np.int64)
        self.fp = np.zeros(n, np.int64)
        self.fn = np.zeros(n, np.int64)

    @staticmethod
    def distance_to_boundary(gt: np.ndarray) -> np.ndarray:
        fg, bg = gt == 1, gt == 0
        d_fg = distance_transform_edt(fg) if (~fg).any() else np.full(gt.shape, np.inf)
        d_bg = distance_transform_edt(bg) if (~bg).any() else np.full(gt.shape, np.inf)
        return np.where(fg, d_fg, d_bg)

    def update(self, pred, gt) -> None:
        gt = _np(gt)
        p = to_binary(pred, self.threshold)
        valid = gt != IGNORE_INDEX
        dist = self.distance_to_boundary(gt)
        b = np.minimum(np.round(np.nan_to_num(dist, posinf=self.max_dist)), self.max_dist).astype(np.int64)
        g = gt == 1
        n = self.max_dist + 1
        self.pixels += np.bincount(b[valid], minlength=n)
        self.fp += np.bincount(b[valid & p & ~g], minlength=n)
        self.fn += np.bincount(b[valid & ~p & g], minlength=n)

    @property
    def errors(self) -> np.ndarray:
        return self.fp + self.fn

    def split(self, band: int = 3) -> dict[str, float]:
        """Share of all errors (and of all pixels) within `band` px of the boundary."""
        e, px = self.errors, self.pixels
        near = slice(0, band + 1)
        tot_e, tot_px = max(e.sum(), 1), max(px.sum(), 1)
        return {"band_px": band,
                "error_share_near": float(e[near].sum() / tot_e),
                "pixel_share_near": float(px[near].sum() / tot_px),
                "error_rate_near": float(e[near].sum() / max(px[near].sum(), 1)),
                "error_rate_interior": float(e[band + 1:].sum() / max(px[band + 1:].sum(), 1))}

    def to_dict(self) -> dict:
        return {"max_dist": self.max_dist, "threshold": self.threshold, "pixels": self.pixels.tolist(),
                "fp": self.fp.tolist(), "fn": self.fn.tolist()}

    @classmethod
    def from_dict(cls, d: Mapping) -> "BoundaryProfile":
        bp = cls(d["max_dist"], d["threshold"])
        bp.pixels, bp.fp, bp.fn = (np.asarray(d[k], np.int64) for k in ("pixels", "fp", "fn"))
        return bp


def plot_boundary_interior(profiles: BoundaryProfile | Mapping[str, BoundaryProfile], band: int = 3) -> Figure:
    """Left: error rate vs distance from the boundary (log scale), with FP and FN split for a
    single model. Right: share of all errors within `band` px vs the share of pixels there."""
    multi = isinstance(profiles, Mapping)
    profs = dict(profiles) if multi else {"model": profiles}
    with _style():
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.3), gridspec_kw={"width_ratios": [1.6, 1]})
        log_y = any(bp.errors.sum() > 0 for bp in profs.values())

        def rate_of(counts, bp):
            r = counts / np.maximum(bp.pixels, 1)
            # A zero rate has no place on a log axis: leave a gap instead of a plunge to the floor.
            return np.where((bp.pixels > 0) & ((r > 0) | (not log_y)), r, np.nan)

        for j, (name, bp) in enumerate(profs.items()):
            d = np.arange(bp.max_dist + 1)
            c = model_color(name, j) if multi else INK
            ax.plot(d, rate_of(bp.errors, bp), color=c, label=name if multi else "all errors")
            if not multi:
                ax.plot(d, rate_of(bp.fp, bp), color=FP_COLOR_HEX, lw=1.5, ls="--", label="false positives (outside pet)")
                ax.plot(d, rate_of(bp.fn, bp), color=FN_COLOR_HEX, lw=1.5, ls="--", label="false negatives (inside pet)")
        ax.axvspan(0, band, color=GRID, alpha=0.6, lw=0)
        ax.text(band / 2, 0.02, f"≤{band}px", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=8, color=INK_2)
        if log_y:
            ax.set_yscale("log")  # error rate spans orders of magnitude from edge to interior
        ax.set_xlabel("distance from GT boundary (px)")
        ax.set_ylabel("pixel error rate")
        ax.set_title("Error rate vs distance from the boundary", loc="left")
        ax.legend(fontsize=8)

        names = list(profs)
        y = np.arange(len(names))
        splits = [profs[n].split(band) for n in names]
        err = [s["error_share_near"] for s in splits]
        pix = [s["pixel_share_near"] for s in splits]
        ax2.barh(y - 0.18, err, height=0.34, color="#2a78d6", label=f"share of errors ≤{band}px")
        ax2.barh(y + 0.18, pix, height=0.34, color="#9ec5f4", label=f"share of pixels ≤{band}px")
        for i, (e, p) in enumerate(zip(err, pix)):
            ax2.text(e + 0.01, i - 0.18, f"{e:.0%}", va="center", fontsize=8.5)
            ax2.text(p + 0.01, i + 0.18, f"{p:.0%}", va="center", fontsize=8.5, color=INK_2)
        ax2.set_yticks(y, names if multi else [""])
        # First model on top (same order as the legend); a single model keeps its bar pair slim.
        ax2.set_ylim((len(names) - 0.4, -0.6) if multi else (1.4, -1.4))
        ax2.set_xlim(0, 1.12)
        ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax2.grid(axis="y", visible=False)
        ax2.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
        ax2.set_title("Boundary vs interior", loc="left")
        fig.tight_layout()
    return fig


class ThresholdSweep:
    """Per-image Dice at every threshold in a single pass over the data.

    Each image's probabilities are histogrammed separately for pet and background pixels;
    a reverse cumulative sum then gives TP and FP for every threshold at once, so no
    prediction has to be kept in memory."""

    def __init__(self, steps: int = 100):
        self.steps = steps
        self.thresholds = np.arange(steps + 1) / steps
        self.dice_sum = np.zeros(steps + 1)
        self.n = 0
        self.tp = np.zeros(steps + 1)
        self.fp = np.zeros(steps + 1)
        self.fn = np.zeros(steps + 1)

    def update(self, prob, gt) -> None:
        p, gt = _np(prob).astype(np.float64).clip(0, 1), _np(gt)
        valid = gt != IGNORE_INDEX
        q = np.floor(p * self.steps + 1e-9).astype(np.int64)
        n = self.steps + 1
        g = gt == 1
        pos = np.bincount(q[valid & g], minlength=n)[::-1].cumsum()[::-1]   # count(p >= t | pet)
        neg = np.bincount(q[valid & ~g], minlength=n)[::-1].cumsum()[::-1]  # count(p >= t | bg)
        n_pet = np.count_nonzero(valid & g)
        tp, fp, fn = pos, neg, n_pet - pos
        self.dice_sum += (2 * tp + 1e-7) / (2 * tp + fp + fn + 1e-7)
        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.n += 1

    @property
    def mean_dice(self) -> np.ndarray:
        return self.dice_sum / max(self.n, 1)

    @property
    def global_dice(self) -> np.ndarray:
        return 2 * self.tp / np.maximum(2 * self.tp + self.fp + self.fn, 1)

    def best(self) -> tuple[float, float]:
        i = int(np.argmax(self.mean_dice))
        return float(self.thresholds[i]), float(self.mean_dice[i])

    def to_dict(self) -> dict:
        return {"steps": self.steps, "n": self.n, "dice_sum": self.dice_sum.tolist(),
                "tp": self.tp.tolist(), "fp": self.fp.tolist(), "fn": self.fn.tolist()}

    @classmethod
    def from_dict(cls, d: Mapping) -> "ThresholdSweep":
        s = cls(d["steps"])
        s.n = d["n"]
        s.dice_sum, s.tp, s.fp, s.fn = (np.asarray(d[k], float) for k in ("dice_sum", "tp", "fp", "fn"))
        return s


def threshold_sweep(probs: Sequence, gts: Sequence, steps: int = 100) -> ThresholdSweep:
    s = ThresholdSweep(steps)
    for p, g in zip(probs, gts):
        s.update(p, g)
    return s


def plot_threshold_sweep(sweeps: ThresholdSweep | Mapping[str, ThresholdSweep], zoom: bool = True) -> Figure:
    """Mean per-image Dice vs sigmoid threshold. Marks each model's best threshold and the
    default 0.5. A flat curve means a confident model; a peaked one means calibration matters."""
    multi = isinstance(sweeps, Mapping)
    sw = dict(sweeps) if multi else {"model": sweeps}
    with _style():
        fig, ax = plt.subplots(figsize=(7.5, 4.4 + 0.25 * len(sw)))
        ax.axvline(0.5, color=INK_2, lw=1, ls=":")
        ax.text(0.505, 0.98, "t = 0.5", transform=ax.get_xaxis_transform(), fontsize=8, color=INK_2, va="top")
        lows, highs = [], []
        for j, (name, s) in enumerate(sw.items()):
            c = model_color(name, j) if multi else SERIES[0]
            t, md = s.thresholds, s.mean_dice
            core = (t > 0.02) & (t < 0.98)
            lows.append(md[core].min()), highs.append(md.max())
            bt, bd = s.best()
            d05 = md[np.argmin(np.abs(t - 0.5))]
            ax.plot(t, md, color=c, label=f"{name}: best {bd:.3f} @ {bt:.2f}, {d05:.3f} @ 0.50" if multi
                    else f"best {bd:.3f} @ t={bt:.2f}  ({d05:.3f} @ 0.50)")
            ax.plot([bt], [bd], "o", color=c, ms=7, mec=SURFACE, mew=2)
        ax.set_xlim(0, 1)
        if zoom:
            ax.set_ylim(max(0, min(lows) - 0.02), min(1, max(highs) + 0.01))
        ax.set_xlabel("sigmoid threshold")
        ax.set_ylabel("mean per-image Dice")
        ax.set_title("Threshold sweep", loc="left")
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1)
        fig.tight_layout()
    return fig


def plot_efficiency(summary: Mapping[str, Mapping[str, float]], metric: str = "dice",
                    cost: str = "latency_ms") -> Figure:
    """Dice vs inference cost per model, every point direct-labeled. summary[name] has
    metric, cost and optionally 'params_m' (marker area scales with parameter count)."""
    with _style():
        fig, ax = plt.subplots(figsize=(7, 4.4))
        for j, (name, s) in enumerate(summary.items()):
            size = 40 + 6 * float(s.get("params_m", 10))
            ax.scatter(s[cost], s[metric], s=size, color=model_color(name, j), edgecolor=SURFACE,
                       linewidth=2, zorder=3)
            label = name + (f"  ({s['params_m']:.1f}M)" if "params_m" in s else "")
            ax.annotate(label, (s[cost], s[metric]), xytext=(8, 4), textcoords="offset points", fontsize=8.5)
        ax.set_xlabel({"latency_ms": "inference latency (ms / image)"}.get(cost, cost))
        ax.set_ylabel(f"test {metric}")
        ax.set_title("Quality vs cost (marker area: parameters)", loc="left")
        ax.margins(x=0.25, y=0.15)
        fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- training curves
class TrainingLogger:
    """Logs scalars and a fixed set of validation overlays during training.

    backend: 'tensorboard' (default), 'wandb', or 'none'. Regardless of backend it also writes
    <log_dir>/history.jsonl (for plot_training_curves) and <log_dir>/overlays/step_XXXX.png
    (so you can watch the masks sharpen without opening TensorBoard)."""

    def __init__(self, log_dir: str | Path, backend: str = "tensorboard", project: str = "pet-seg",
                 run_name: str | None = None, config: Mapping | None = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        (self.log_dir / "overlays").mkdir(exist_ok=True)
        self.history_path = self.log_dir / "history.jsonl"
        self.backend = backend
        self.tb = self.wandb = None
        if backend == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter
            self.tb = SummaryWriter(str(self.log_dir))
            if config:
                self.tb.add_text("config", "```\n" + json.dumps(dict(config), indent=2, default=str) + "\n```")
        elif backend == "wandb":
            import wandb
            self.wandb = wandb.init(project=project, name=run_name, config=dict(config or {}),
                                    dir=str(self.log_dir))

    def log_scalars(self, scalars: Mapping[str, float], step: int) -> None:
        """Use keys like 'train/loss', 'val/loss', 'val/dice', 'lr'."""
        with self.history_path.open("a") as f:
            f.write(json.dumps({"step": step, **{k: float(v) for k, v in scalars.items()}}) + "\n")
        if self.tb:
            for k, v in scalars.items():
                self.tb.add_scalar(k, float(v), step)
        if self.wandb:
            self.wandb.log(dict(scalars), step=step)

    def log_val_overlays(self, images: Sequence[np.ndarray], gts: Sequence[np.ndarray], probs: Sequence,
                         step: int, threshold: float = 0.5, tag: str = "val/overlays") -> np.ndarray:
        """One row per fixed validation image: overlay (captioned with Dice) | error map | probability."""
        rows = [prediction_strip(_np(im), _np(p), _np(g), threshold) for im, g, p in zip(images, gts, probs)]
        grid = label_tile(tile(rows, cols=1, pad=4), f"step {step}", height=26)
        cv2.imwrite(str(self.log_dir / "overlays" / f"step_{step:04d}.png"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        if self.tb:
            self.tb.add_image(tag, grid, step, dataformats="HWC")
        if self.wandb:
            import wandb
            self.wandb.log({tag: wandb.Image(grid)}, step=step)
        return grid

    def close(self) -> None:
        if self.tb:
            self.tb.close()
        if self.wandb:
            self.wandb.finish()


def read_history(path: str | Path) -> dict[str, list[tuple[int, float]]]:
    series: dict[str, list[tuple[int, float]]] = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        step = rec.pop("step")
        for k, v in rec.items():
            series.setdefault(k, []).append((step, v))
    return series


def plot_training_curves(histories: str | Path | Mapping[str, str | Path]) -> Figure:
    """Loss (train dashed, val solid) and val Dice per epoch, from history.jsonl files.
    Pass one path, or {model_name: path} to compare runs. Two panels, never a dual axis."""
    multi = isinstance(histories, Mapping)
    runs = {k: read_history(v) for k, v in (histories.items() if multi else {"run": histories}.items())}
    with _style():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
        for j, (name, h) in enumerate(runs.items()):
            c = model_color(name, j) if multi else SERIES[0]
            for key, ls, lab in (("train/loss", "--", "train"), ("val/loss", "-", "val")):
                if key in h:
                    s, v = zip(*h[key])
                    ax1.plot(s, v, ls=ls, color=c if multi else (SERIES[1] if lab == "train" else SERIES[0]),
                             label=(f"{name} {lab}" if multi else lab))
            if "val/dice" in h:
                s, v = zip(*h["val/dice"])
                ax2.plot(s, v, color=c, label=name if multi else "val Dice")
                b = int(np.argmax(v))
                ax2.plot([s[b]], [v[b]], "o", color=c, ms=7, mec=SURFACE, mew=2)
                ax2.annotate(f"{v[b]:.3f}", (s[b], v[b]), xytext=(4, 6), textcoords="offset points", fontsize=8)
        ax1.set_title("Loss (Dice + BCE)", loc="left")
        ax1.set_xlabel("epoch")
        ax1.legend(fontsize=8, ncol=2 if multi else 1)
        ax2.set_title("Validation Dice", loc="left")
        ax2.set_xlabel("epoch")
        ax2.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- gallery (python src/viz.py)
_GALLERY_CSS = """
:root { --bg:#f6f5f2; --card:#fff; --ink:#151514; --ink2:#5d5c58; --line:#e4e2dc; --accent:#ff4081; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#141413; --card:#1f1f1d; --ink:#f0efec; --ink2:#a9a8a1; --line:#2e2e2b; }
}
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
       font:15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, sans-serif; }
header { padding:40px 24px 16px; max-width:1200px; margin:auto; }
header h1 { margin:0; font-size:28px; letter-spacing:-0.02em; }
header h1 span { color:var(--accent); }
header p { margin:6px 0 0; color:var(--ink2); }
nav { position:sticky; top:0; z-index:2; background:var(--bg); border-bottom:1px solid var(--line); }
nav div { max-width:1200px; margin:auto; padding:10px 24px; display:flex; gap:18px; flex-wrap:wrap; }
nav a { color:var(--ink2); text-decoration:none; font-weight:600; font-size:14px; margin-right:6px; }
nav a:hover { color:var(--accent); }
main { max-width:1200px; margin:auto; padding:8px 24px 64px; }
section { margin-top:36px; }
section h2 { font-size:20px; margin:0 0 4px; }
section > p { margin:0 0 14px; color:var(--ink2); max-width:780px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:12px;
        margin-bottom:14px; }
.card h3 { margin:2px 4px 8px; font-size:14px; font-weight:600; color:var(--ink2); }
.card img { width:100%; display:block; border-radius:6px; background:#fcfcfb; cursor:zoom-in; }
.grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(520px, 1fr)); gap:14px; }
.grid .card { margin:0; }
.empty { color:var(--ink2); font-style:italic; }
dialog { border:none; padding:0; background:transparent; max-width:96vw; }
dialog img { max-width:96vw; max-height:92vh; border-radius:8px; background:#fcfcfb; }
dialog::backdrop { background:rgba(0,0,0,.78); }
@media (max-width:600px) { .grid { grid-template-columns:1fr; } header { padding-top:24px; } }
"""

_GALLERY_JS = """
const d = document.querySelector('dialog'), big = d.querySelector('img');
document.querySelectorAll('.card img').forEach(i => i.addEventListener('click', () => {
  big.src = i.src; d.showModal(); }));
d.addEventListener('click', () => d.close());
"""


def build_gallery(root: str | Path, out: str | Path, split: str = "train", n: int = 4,
                  runs_dir: str | Path = "runs", results_dir: str | Path = "results",
                  eval_split: str = "val", seed: int = 0) -> Path:
    """Renders every view that the available inputs allow into <out>/ and writes an index.html.

    Always: show_sample and show_augmented for n images.
    With trained runs (runs/*/best.pt): show_prediction per model and model grids.
    With evaluation output (results/<eval_split>/): the evaluation and training charts."""
    import html
    import random

    out = Path(out)
    img_dir = out / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    sections: list[tuple[str, str, str, list[tuple[str, str]], bool]] = []

    def emit(fig: Figure, name: str) -> str:
        save_figure(fig, img_dir / name)
        return f"img/{name}"

    ds = OxfordPetSeg(root, split=split)
    idxs = random.Random(seed).sample(range(len(ds)), min(n, len(ds)))
    print(f"[gallery] data sanity views for {len(idxs)} {split} images")
    cards = []
    for i in idxs:
        nm = ds.records[i].name
        cards.append((f"{nm}: image, raw trimap, converted mask", emit(show_sample(ds, i), f"sample_{nm}.png")))
        cards.append((f"{nm}: augmentation passes", emit(show_augmented(ds, i, n=8), f"augmented_{nm}.png")))
    sections.append(("data", "Data sanity checks",
                     "The converted mask must sit exactly on the pet, and the outline must stay on the pet "
                     "through every augmentation pass. Gray marks ignored boundary and padding pixels.", cards, False))

    from .inference import find_checkpoints  # local import: inference imports this module

    ckpts = find_checkpoints(runs_dir)
    cards = []
    if ckpts:
        from .inference import Predictor

        preds = {nm: Predictor(p) for nm, p in ckpts.items()}
        print(f"[gallery] predictions from {', '.join(preds)}")
        images, gts, labels = [], [], []
        for i in idxs:
            img, tri = ds.load_raw(i)
            images.append(img)
            gts.append(convert_trimap(tri, ds.boundary_mode))
            labels.append(ds.records[i].name)
        probs = {nm: [p.predict(im) for im in images] for nm, p in preds.items()}
        for view, title in (("overlay", "Model grid: overlays"), ("error", "Model grid: error maps")):
            cards.append((title, emit(model_grid(images, gts, probs, view=view, row_labels=labels),
                                      f"model_grid_{view}.png")))
        for nm in preds:
            cards.append((f"{nm}: {labels[0]}", emit(show_prediction(images[0], probs[nm][0], gts[0], title=nm),
                                                       f"prediction_{nm}.png")))
    sections.append(("predictions", "Predictions",
                     "Error map colors: green true positive, red false positive, blue false negative, "
                     "gray ignored boundary. Numbers in the grid are per-image Dice.", cards, False))

    res = Path(results_dir) / eval_split
    names = ("training_curves.png", "threshold_sweep.png", "boundary.png", "per_breed.png", "efficiency.png",
             "worst_k.png")
    cards = []
    if res.exists():
        import shutil

        for f in names:
            for src in [res / f, *sorted(res.glob(f"*/{f}"))]:
                if src.exists():
                    label = f"{src.parent.name if src.parent != res else 'all models'}: {src.stem.replace('_', ' ')}"
                    dst = img_dir / f"eval_{src.parent.name}_{f}"
                    shutil.copy(src, dst)
                    cards.append((label, f"img/{dst.name}"))
    sections.append(("evaluation", f"Evaluation ({eval_split})",
                     "Training curves, threshold sweep, boundary vs interior error, per-breed Dice, "
                     "quality vs cost, and each model's worst images.", cards, True))

    def section_html(sid, title, blurb, cards, grid):
        if not cards:
            hint = {"predictions": "No trained models yet. Train one with python -m src.train --config configs/unet_scratch.yaml.",
                    "evaluation": f"No evaluation output yet. Run python -m src.evaluate --split {eval_split}."}[sid]
            body = f'<p class="empty">{html.escape(hint)}</p>'
        else:
            items = "".join(f'<div class="card"><h3>{html.escape(t)}</h3><img src="{s}" loading="lazy" '
                            f'alt="{html.escape(t)}"></div>' for t, s in cards)
            body = f'<div class="grid">{items}</div>' if grid else items
        return f'<section id="{sid}"><h2>{title}</h2><p>{html.escape(blurb)}</p>{body}</section>'

    nav = "".join(f'<a href="#{s[0]}">{s[1]}</a>' for s in sections)
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1"><title>Pet segmentation: views</title>'
            f'<style>{_GALLERY_CSS}</style></head><body>'
            f'<header><h1>Pet segmentation <span>views</span></h1>'
            f'<p>{len(ds)} {split} images from {html.escape(str(root))} · {len(ckpts)} trained model(s)</p></header>'
            f'<nav><div>{nav}</div></nav><main>{"".join(section_html(*s) for s in sections)}</main>'
            f'<dialog><img alt=""></dialog><script>{_GALLERY_JS}</script></body></html>')
    index = out / "index.html"
    index.write_text(page)
    return index


def _main() -> None:
    import argparse

    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description="Render every visualization view into an HTML gallery.")
    ap.add_argument("--root", default=str(repo / "data" / "oxford-iiit-pet"), help="dataset folder")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--n", type=int, default=4, help="images to show")
    ap.add_argument("--runs-dir", default=str(repo / "runs"))
    ap.add_argument("--results-dir", default=str(repo / "results"))
    ap.add_argument("--eval-split", default="val")
    ap.add_argument("--out", default=str(repo / "debug" / "gallery"))
    ap.add_argument("--open", action="store_true", help="open the gallery in your browser")
    a = ap.parse_args()

    if not (Path(a.root) / "annotations" / "trimaps").is_dir():
        raise SystemExit(f"No dataset at {a.root}.\nDownload it with: python data/download.py\n"
                         f"or try the synthetic one: python scripts/make_fake_dataset.py --out data/fake-pet "
                         f"&& python src/viz.py --root data/fake-pet")
    plt.switch_backend("Agg")
    index = build_gallery(a.root, a.out, a.split, a.n, a.runs_dir, a.results_dir, a.eval_split)
    print(f"[gallery] open {index}")
    if a.open:
        import webbrowser

        webbrowser.open(index.resolve().as_uri())


if __name__ == "__main__":
    _main()
