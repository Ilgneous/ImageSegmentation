"""Gradio demo: upload an image or short video, pick a model or all of them, inspect the results.

    python app.py                    # serves http://127.0.0.1:7860
    python app.py --share            # public link for the interview

Images: overlay, error map (when ground truth is available) and probability heatmap per model.
Ground truth comes from picking a dataset image, or from uploading a trimap / binary mask.
Video: overlay video (models tiled when 'all' is picked) plus a frame-to-frame stability score.
"""
from __future__ import annotations

import argparse
import tempfile

import gradio as gr
import numpy as np

from predict import load_gt
from src import viz
from src.dataset import OxfordPetSeg, convert_trimap
from src.inference import Predictor, find_checkpoints
from src.video import segment_video

ALL = "all models"


def build_app(runs_dir: str = "runs", device: str | None = None, split: str = "test") -> gr.Blocks:
    predictors = {n: Predictor(p, device) for n, p in find_checkpoints(runs_dir).items()}
    if not predictors:
        raise SystemExit(f"no trained models in {runs_dir}/")
    try:  # dataset images with ground truth, for the error map
        cfg = next(iter(predictors.values())).config["data"]
        ds = OxfordPetSeg(cfg["root"], split=split, boundary_mode=cfg.get("boundary_mode", "ignore"),
                          val_fraction=cfg.get("val_fraction", 0.1), seed=cfg.get("split_seed", 0))
        sample_names = [r.name for r in ds.records]
    except Exception:
        ds, sample_names = None, []

    def pick(choice: str) -> dict:
        return predictors if choice == ALL else {choice: predictors[choice]}

    def load_sample(name):
        if not name or ds is None:
            return None, None
        img, trimap = ds.load_raw(ds.index_of(name))
        return img, convert_trimap(trimap, ds.boundary_mode)

    def run_image(image, gt_state, gt_file, choice, threshold):
        if image is None:
            raise gr.Error("Upload an image or pick a dataset sample first.")
        gt = load_gt(gt_file) if gt_file else gt_state
        if gt is not None and gt.shape != image.shape[:2]:
            gt = viz._resize_to(gt, image.shape[:2], nearest=True)
        gallery, lines = [], ["| model | Dice | pet area |", "|---|---|---|"]
        for name, p in pick(choice).items():
            prob = p.predict(image)
            gallery.append((viz.overlay(image, prob, threshold=threshold), f"{name}: overlay"))
            if gt is not None:
                gallery.append((viz.error_map(prob, gt, threshold, image), f"{name}: error map"))
            gallery.append((viz.prob_heatmap(prob, image), f"{name}: probability"))
            dice = f"{viz.dice_np(prob >= threshold, gt):.4f}" if gt is not None else "n/a (no GT)"
            lines.append(f"| {name} | {dice} | {np.mean(prob >= threshold):.1%} |")
        legend = ("\n\nError map: **green** true positive, **red** false positive, "
                  "**blue** false negative, **gray** ignored boundary." if gt is not None else "")
        return gallery, "\n".join(lines) + legend

    def run_video(video, choice, threshold, max_frames):
        if video is None:
            raise gr.Error("Upload a video first.")
        out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        stab = segment_video(video, out, pick(choice), threshold, max_frames=int(max_frames))
        lines = ["| model | mean consecutive-frame IoU |", "|---|---|"]
        lines += [f"| {n} | {s:.4f} |" for n, s in stab.items()]
        return out, "\n".join(lines) + "\n\n1.0 means a perfectly stable mask, lower means flicker."

    choices = [ALL, *predictors]
    with gr.Blocks(title="Pet segmentation") as demo:
        gr.Markdown(f"# Pet segmentation demo\nTrained models: {', '.join(predictors)}")
        with gr.Tab("Image"):
            gt_state = gr.State(None)
            with gr.Row():
                with gr.Column(scale=1):
                    image = gr.Image(type="numpy", label="Image")
                    sample = gr.Dropdown(sample_names, label=f"...or pick a {split} image (has ground truth)",
                                         filterable=True, visible=bool(sample_names))
                    gt_file = gr.File(label="Optional ground truth (trimap or binary mask)", type="filepath")
                    model = gr.Dropdown(choices, value=ALL, label="Model")
                    thr = gr.Slider(0.05, 0.95, 0.5, step=0.05, label="Threshold")
                    go = gr.Button("Segment", variant="primary")
                with gr.Column(scale=2):
                    gallery = gr.Gallery(label="Results", columns=3, height="auto", object_fit="contain")
                    table = gr.Markdown()
            sample.change(load_sample, sample, [image, gt_state])
            image.upload(lambda: None, None, gt_state)  # a new upload has no known GT
            go.click(run_image, [image, gt_state, gt_file, model, thr], [gallery, table])
        with gr.Tab("Video"):
            with gr.Row():
                with gr.Column():
                    video = gr.Video(label="Short video")
                    vmodel = gr.Dropdown(choices, value=next(iter(predictors)), label="Model")
                    vthr = gr.Slider(0.05, 0.95, 0.5, step=0.05, label="Threshold")
                    vmax = gr.Slider(10, 600, 150, step=10, label="Max frames")
                    vgo = gr.Button("Segment video", variant="primary")
                with gr.Column():
                    vout = gr.Video(label="Overlay", format="mp4")
                    vtable = gr.Markdown()
            vgo.click(run_video, [video, vmodel, vthr, vmax], [vout, vtable])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--split", default="test", help="dataset split offered in the sample picker")
    ap.add_argument("--device")
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    build_app(a.runs_dir, a.device, a.split).launch(share=a.share, server_port=a.port)
