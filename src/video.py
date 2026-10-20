"""Frame-by-frame video segmentation shared by the CLI and the Gradio app."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import cv2
import numpy as np

from . import viz
from .inference import Predictor

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def is_video(path: str | Path) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTS


def _to_h264(path: Path) -> None:
    """OpenCV can only write mp4v, which browsers will not play. Re-encode with ffmpeg when available."""
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None:
        return
    tmp = path.with_name(path.stem + ".h264.mp4")
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)])
    if r.returncode == 0:
        tmp.replace(path)
    elif tmp.exists():
        tmp.unlink()


def segment_video(src: str | Path, dst: str | Path, predictors: Mapping[str, Predictor], threshold: float = 0.5,
                  max_side: int = 640, max_frames: int | None = None, batch: int = 8) -> dict[str, float]:
    """Writes an overlay video (models tiled side by side when there are several) and returns the
    mean IoU between consecutive predicted masks per model. That temporal-consistency score is
    1.0 for a perfectly stable mask, and drops when masks flicker between frames."""
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise ValueError(f"cannot open video {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    names = list(predictors)
    writer, prev = None, {n: None for n in names}
    stability = {n: [] for n in names}
    frames, count = [], 0

    def flush(frames_rgb: list[np.ndarray]):
        nonlocal writer
        probs = {n: predictors[n].predict_batch(frames_rgb) for n in names}
        for k, f in enumerate(frames_rgb):
            tiles = []
            for n in names:
                m = probs[n][k] >= threshold
                if prev[n] is not None:
                    union = np.count_nonzero(m | prev[n])
                    stability[n].append(np.count_nonzero(m & prev[n]) / union if union else 1.0)
                prev[n] = m
                tiles.append(viz.label_tile(viz.overlay(f, m), n, height=max(18, f.shape[0] // 20)))
            out = viz.tile(tiles, cols=min(3, len(tiles))) if len(tiles) > 1 else tiles[0]
            if writer is None:
                h, w = out.shape[:2]
                writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

    while True:
        ok, bgr = cap.read()
        if not ok or (max_frames and count >= max_frames):
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        s = max_side / max(rgb.shape[:2])
        if s < 1:
            rgb = cv2.resize(rgb, (int(rgb.shape[1] * s) // 2 * 2, int(rgb.shape[0] * s) // 2 * 2))
        frames.append(rgb)
        count += 1
        if len(frames) == batch:
            flush(frames)
            frames = []
    if frames:
        flush(frames)
    cap.release()
    if writer is not None:
        writer.release()
        _to_h264(Path(dst))
    return {n: float(np.mean(v)) if v else float("nan") for n, v in stability.items()}
