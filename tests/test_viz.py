import matplotlib.pyplot as plt
import numpy as np
import pytest

from src import viz
from src.dataset import IGNORE_INDEX, OxfordPetSeg, get_transforms


def toy():
    gt = np.zeros((6, 6), np.uint8)
    gt[1:4, 1:4] = 1
    gt[0, :] = IGNORE_INDEX
    prob = np.zeros((6, 6), np.float32)
    prob[2:5, 2:5] = 0.9
    return gt, prob


def test_error_map_colors():
    gt, prob = toy()
    em = viz.error_map(prob, gt)
    assert tuple(em[2, 2]) == viz.TP_COLOR
    assert tuple(em[4, 4]) == viz.FP_COLOR
    assert tuple(em[1, 1]) == viz.FN_COLOR
    assert tuple(em[0, 3]) == viz.IGNORE_COLOR
    assert tuple(em[5, 0]) == (0, 0, 0)
    c = viz.error_counts(prob, gt)
    assert c == {"tp": 4, "fp": 5, "fn": 5, "ignored": 6}


def test_threshold_sweep_matches_brute_force():
    rng = np.random.default_rng(0)
    gts = [rng.integers(0, 2, (20, 20)).astype(np.uint8) for _ in range(4)]
    for g in gts:
        g[0] = IGNORE_INDEX
    probs = [np.clip(g * 0.6 + rng.random((20, 20)) * 0.5, 0, 1).astype(np.float32) for g in gts]
    s = viz.threshold_sweep(probs, gts, steps=20)
    for j in (3, 10, 15):
        t = s.thresholds[j]
        brute = np.mean([viz.dice_np(p >= t, g) for p, g in zip(probs, gts)])
        assert abs(s.mean_dice[j] - brute) < 1e-6
    rt = viz.ThresholdSweep.from_dict(s.to_dict())
    assert np.allclose(rt.mean_dice, s.mean_dice)


def test_boundary_profile_counts_every_valid_pixel_and_error():
    gt, prob = toy()
    bp = viz.BoundaryProfile(max_dist=10)
    bp.update(prob, gt)
    c = viz.error_counts(prob, gt)
    assert bp.pixels.sum() == np.count_nonzero(gt != IGNORE_INDEX)
    assert bp.fp.sum() == c["fp"] and bp.fn.sum() == c["fn"]
    s = bp.split(band=1)
    assert 0 <= s["error_share_near"] <= 1


def test_renderers_shapes_and_overlay_resizes():
    gt, prob = toy()
    img = np.full((12, 12, 3), 128, np.uint8)
    assert viz.overlay(img, prob).shape == (12, 12, 3)
    assert viz.binary_mask(gt)[0, 0].tolist() == list(viz.IGNORE_COLOR)
    assert viz.prob_heatmap(prob, mode="uncertainty").shape == (6, 6, 3)
    strip = viz.prediction_strip(np.full((6, 6, 3), 50, np.uint8), prob, gt)
    assert strip.ndim == 3 and strip.dtype == np.uint8


def test_figures_render(fake_root, tmp_path):
    ds = OxfordPetSeg(fake_root, split="train", image_size=64)
    figs = [viz.show_sample(ds, 0), viz.show_augmented(ds, 0, n=4, image_size=64)]
    img, gt = ds.load_pair(0, get_transforms(False, 64))
    prob = (gt == 1).astype(np.float32) * 0.8 + 0.1
    figs.append(viz.show_prediction(img, prob, gt))
    figs.append(viz.model_grid([img, img], [gt, gt], {"unet_scratch": [prob, prob], "segformer": [prob, 1 - prob]},
                               view="error"))
    items = [{"image": img, "gt": gt, "prob": prob, "name": f"x{i}"} for i in range(6)]
    figs.append(viz.worst_k_gallery(np.linspace(0.5, 1, 6), items, k=4, cols=2))
    breeds = [r.breed for r in ds.records]
    d = np.random.default_rng(0).uniform(0.7, 1, len(breeds))
    figs += [viz.plot_per_breed(d, breeds), viz.plot_per_breed({"a": d, "b": d[::-1]}, breeds)]
    bp = viz.BoundaryProfile()
    bp.update(prob, gt)
    figs += [viz.plot_boundary_interior(bp), viz.plot_boundary_interior({"a": bp, "b": bp})]
    sw = viz.threshold_sweep([prob], [gt])
    figs += [viz.plot_threshold_sweep(sw), viz.plot_threshold_sweep({"a": sw, "b": sw})]
    figs.append(viz.plot_efficiency({"a": {"dice": 0.9, "latency_ms": 5, "params_m": 7}}))
    for k, f in enumerate(figs):
        viz.save_figure(f, tmp_path / f"{k}.png")
    assert len(list(tmp_path.glob("*.png"))) == len(figs)
    plt.close("all")


def test_training_logger_writes_history_and_overlays(tmp_path):
    gt, prob = toy()
    img = np.full((6, 6, 3), 90, np.uint8)
    lg = viz.TrainingLogger(tmp_path, backend="none")
    for e in (1, 2):
        lg.log_scalars({"train/loss": 1 / e, "val/loss": 1.1 / e, "val/dice": 0.5 + e / 10}, e)
    lg.log_val_overlays([img, img], [gt, gt], [prob, prob], step=2)
    lg.close()
    assert (tmp_path / "overlays" / "step_0002.png").exists()
    h = viz.read_history(tmp_path / "history.jsonl")
    assert [v for _, v in h["val/dice"]] == pytest.approx([0.6, 0.7])
    viz.save_figure(viz.plot_training_curves(tmp_path / "history.jsonl"), tmp_path / "curves.png")
