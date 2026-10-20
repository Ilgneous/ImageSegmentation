"""Load trained checkpoints and predict on arbitrary images or video frames."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from .dataset import normalize_image
from .models import build_model


def pick_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_image(path: str | Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


class Predictor:
    """Wraps a checkpoint written by src.train. predict() takes any-size uint8 RGB and returns
    a float probability map at the ORIGINAL resolution (the model runs at its training size)."""

    def __init__(self, checkpoint: str | Path, device: str | None = None):
        self.device = pick_device(device)
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.config = ckpt["config"]
        self.name = self.config.get("run_name", self.config["model"]["name"])
        self.image_size = int(self.config["data"].get("image_size", 256))
        model_cfg = dict(self.config["model"], pretrained=False)  # weights come from the checkpoint
        self.model = build_model(model_cfg)
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device).eval()
        self.val_dice = ckpt.get("val_dice")

    @torch.no_grad()
    def predict_batch(self, images: list[np.ndarray]) -> list[np.ndarray]:
        s = self.image_size
        x = torch.stack([normalize_image(cv2.resize(im, (s, s), interpolation=cv2.INTER_LINEAR)) for im in images])
        probs = torch.sigmoid(self.model(x.to(self.device)))[:, 0].float().cpu().numpy()
        return [cv2.resize(p, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_LINEAR)
                for p, im in zip(probs, images)]

    def predict(self, image: np.ndarray) -> np.ndarray:
        return self.predict_batch([image])[0]

    def predict_model_res(self, x: torch.Tensor) -> torch.Tensor:
        """Already-normalized batch at model resolution -> probabilities (B,H,W)."""
        with torch.no_grad():
            return torch.sigmoid(self.model(x.to(self.device)))[:, 0].float().cpu()


def find_checkpoints(runs_dir: str | Path = "runs") -> dict[str, Path]:
    """{run_name: runs/<run_name>/best.pt} for every finished run, in the canonical model order
    (scratch U-Net first, DINOv3 probe last) so tables and grid columns read as the comparison ladder."""
    from .viz import MODEL_ORDER

    def rank(p: Path) -> tuple[int, str]:
        name = p.parent.name
        hits = [i for i, m in enumerate(MODEL_ORDER) if name.startswith(m)]
        return (hits[0] if hits else len(MODEL_ORDER), name)

    return {p.parent.name: p for p in sorted(Path(runs_dir).glob("*/best.pt"), key=rank)}
