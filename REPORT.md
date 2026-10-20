# Results

> Figures referenced below are produced by `python -m src.evaluate --split test` into `results/test/`.

## Setup
- Data: Oxford-IIIT Pet, official trainval split into train / val (90/10, stratified by breed, seed 0), official test split held out until the end.
- Input 256x256, ImageNet normalization, boundary pixels ignored.
- Loss soft Dice + BCE, AdamW, 1 epoch warmup then cosine, 30 epochs, best checkpoint by val Dice.

## Summary table
| Model | Params | Test Dice | Test IoU | Best threshold | Latency (ms) |
|---|---|---|---|---|---|
| | | | | | |

## Training
![training curves](results/test/training_curves.png)

## Comparisons
### 1 vs 2: pretraining
### 2 vs 3: decoder design
### 3 vs 4: CNN vs transformer
### 4 vs 5: task-trained vs frozen features

![model grid](results/test/model_grid_error.png)
![efficiency](results/test/efficiency.png)

## Error analysis
![boundary](results/test/boundary.png)
![per breed](results/test/per_breed.png)
![threshold](results/test/threshold_sweep.png)

Worst-k galleries: `results/test/<model>/worst_k.png`

## Generalization
- Grayscale
- Resolution
- Other animals
- Video

## What I would change with more time
