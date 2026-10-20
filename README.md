# pet-seg

Five segmentation models for the Oxford-IIIT Pet dataset, trained and compared with one
pipeline, plus the visualization tools used to develop and debug them.

| # | Model | Comparison it supports |
|---|---|---|
| 1 | `unet_scratch`: U-Net written from scratch | baseline |
| 2 | `unet_resnet34`: same U-Net, ImageNet ResNet34 encoder | 1 vs 2 isolates pretraining |
| 3 | `deeplabv3plus`: ResNet50, ASPP | 2 vs 3 isolates decoder design |
| 4 | `segformer`: MiT-B0 | 3 vs 4 CNN vs transformer |
| 5 | `dino_linear`: frozen DINOv3 ViT-S/16, linear head | 4 vs 5 task-trained vs frozen features |

Metric: per-image Dice (IoU reported alongside). Loss: soft Dice + BCE. Trimap boundary
pixels (class 3) are ignored in both by default (`data.boundary_mode`).

## Layout

```
data/download.py           download + verify the dataset
scripts/make_fake_dataset.py   tiny synthetic dataset in the same layout (smoke tests)
src/
  dataset.py               splits, trimap conversion, augmentation, normalization
  models/                  unet.py, smp_models.py, segformer.py, dino_probe.py, registry
  losses.py, metrics.py
  train.py                 config-driven training for all five models
  evaluate.py              metrics + every evaluation chart
  viz.py                   all visualization views (see below)
  inference.py, video.py   checkpoint loading, image and video prediction
configs/*.yaml             one per model
app.py                     Gradio demo
predict.py                 CLI
tests/                     metrics, viz and dataset tests (run on the synthetic data)
REPORT.md  AI_USAGE.md  REFERENCES.md
```

## Quick start

```bash
pip install -r requirements.txt
python data/download.py                      # data/oxford-iiit-pet/
python scripts/sanity_check.py --scan       # sample figures to debug/, checks every image
pytest -q                                    # ~10 s, no GPU or dataset needed

python -m src.train --config configs/unet_scratch.yaml
tensorboard --logdir runs                    # loss, Dice, and 8 fixed val overlays over time
python -m src.evaluate --split val           # every runs/*/best.pt -> results/val/
python predict.py --model segformer --input img.jpg --out overlay.png
python app.py
```

DINOv3 weights on Hugging Face are gated: accept the license on the model page and run
`huggingface-cli login` before training model 5. Install `ffmpeg` so output videos are
re-encoded to H.264 (OpenCV alone writes mp4v, which browsers will not play).

Smoke-test the whole pipeline on CPU in about a minute:

```bash
python scripts/make_fake_dataset.py --out data/fake-pet --n 120
python -m src.train --config configs/unet_scratch.yaml \
    --set data.root=data/fake-pet data.image_size=96 train.epochs=3 train.batch_size=8 data.num_workers=0
python -m src.evaluate --split val
```

## Visualization tools (`src/viz.py`)

| Checklist item | Function |
|---|---|
| Image, raw trimap, converted mask side by side | `show_sample(ds, idx)` |
| Same sample through n augmentation passes | `show_augmented(ds, idx, n=8)` |
| Semi-transparent overlay | `overlay(image, prob)` |
| Black and white mask | `binary_mask(prob)` |
| Error map (TP green, FP red, FN blue, ignored gray) | `error_map(prob, gt, image=image)` |
| Probability heatmap before thresholding | `prob_heatmap(prob, image)` (`mode="uncertainty"` too) |
| All of the above for one prediction | `show_prediction(image, prob, gt)` |
| Model grid: rows = images, cols = image, GT, each model | `model_grid(images, gts, {name: probs})` |
| Worst-k gallery by per-image Dice | `worst_k_gallery(dices, items, k=20)` |
| Per-breed Dice, cats vs dogs | `plot_per_breed(dices, breeds)` (dict of models for a heatmap) |
| Boundary vs interior error split | `BoundaryProfile` + `plot_boundary_interior` |
| Threshold sweep | `ThresholdSweep` + `plot_threshold_sweep` |
| Dice vs latency | `plot_efficiency(summary)` |
| TensorBoard / W&B scalars + 8 fixed val overlays | `TrainingLogger` (used by `train.py`) |
| Training curves for the report | `plot_training_curves({name: history.jsonl})` |

Render every view the available inputs allow (data checks always, predictions once a model is
trained, evaluation charts once `src.evaluate` has run) into one HTML page:

```bash
python src/viz.py --open                     # writes debug/gallery/index.html
python src/viz.py --root data/fake-pet --n 6
```

Image renderers return uint8 RGB arrays, figure builders return matplotlib figures:

```python
from src.dataset import OxfordPetSeg
from src import viz

ds = OxfordPetSeg("data/oxford-iiit-pet", split="train")
viz.show_sample(ds, 0)
viz.show_augmented(ds, 0, n=8)                 # outline must stay on the pet
viz.save_figure(viz.show_sample(ds, 42), "debug/sample_42.png")
```

`ThresholdSweep` and `BoundaryProfile` are accumulators: call `.update(prob, gt)` per image
inside an evaluation loop, so no predictions need to be kept in memory.

## Demo

`python app.py` serves the Gradio app. Image tab: upload an image or pick a test image (which
brings its ground truth, enabling the error map and Dice), choose one model or all, adjust the
threshold. Video tab: overlay video for one or all models plus mean consecutive-frame IoU as a
flicker score.

CLI equivalents:

```bash
python predict.py --model segformer --input img.jpg --out overlay.png
python predict.py --model all --input img.jpg --gt trimap.png --view error --out grid.png
python predict.py --model unet_resnet34 --input img.jpg --view panel --out panel.png
python predict.py --model all --input clip.mp4 --out clip_seg.mp4
```
