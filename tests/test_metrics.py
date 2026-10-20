import numpy as np
import torch

from src.dataset import IGNORE_INDEX, convert_trimap
from src.losses import DiceBCELoss, soft_dice_loss
from src.metrics import dice_per_image, iou_per_image, pixel_accuracy_per_image


def test_trimap_conversion_modes():
    t = np.array([[1, 2, 3]], np.uint8)
    assert convert_trimap(t, "ignore").tolist() == [[1, 0, IGNORE_INDEX]]
    assert convert_trimap(t, "fg").tolist() == [[1, 0, 1]]
    assert convert_trimap(t, "bg").tolist() == [[1, 0, 0]]


def test_dice_known_values():
    gt = torch.zeros(1, 4, 4, dtype=torch.long)
    gt[0, :2] = 1                                   # 8 pet pixels
    pred = torch.zeros(1, 4, 4, dtype=torch.long)
    pred[0, :1] = 1                                 # 4 predicted, all correct
    assert torch.isclose(dice_per_image(pred, gt), torch.tensor([2 * 4 / (4 + 8)]), atol=1e-6)
    assert torch.isclose(iou_per_image(pred, gt), torch.tensor([4 / 8]), atol=1e-6)
    assert dice_per_image(gt, gt).item() > 0.9999


def test_ignored_pixels_do_not_count():
    gt = torch.tensor([[[1, 1, IGNORE_INDEX, 0]]])
    good = torch.tensor([[[1, 1, 0, 0]]])
    also_good = torch.tensor([[[1, 1, 1, 0]]])     # differs only on the ignored pixel
    assert dice_per_image(good, gt).item() == dice_per_image(also_good, gt).item() > 0.9999


def test_iou_dice_relation():
    rng = np.random.default_rng(0)
    gt = torch.from_numpy(rng.integers(0, 2, (5, 16, 16)))
    pred = torch.from_numpy(rng.integers(0, 2, (5, 16, 16)))
    d, i = dice_per_image(pred, gt), iou_per_image(pred, gt)
    assert torch.allclose(i, d / (2 - d), atol=1e-5)


def test_pixel_accuracy_is_misleading():
    gt = torch.zeros(1, 10, 10, dtype=torch.long)
    gt[0, :3] = 1                                    # 30% pet
    all_bg = torch.zeros_like(gt)
    assert abs(pixel_accuracy_per_image(all_bg, gt).item() - 0.7) < 1e-6
    assert dice_per_image(all_bg, gt).item() < 1e-6


def test_soft_dice_matches_hard_dice_at_confident_logits_and_has_gradient():
    gt = torch.zeros(2, 8, 8, dtype=torch.long)
    gt[:, 2:6, 2:6] = 1
    logits = (gt.float() * 2 - 1).unsqueeze(1) * 20
    assert soft_dice_loss(logits, gt).item() < 1e-3
    logits = torch.zeros(2, 1, 8, 8, requires_grad=True)
    DiceBCELoss()(logits, gt).backward()
    assert logits.grad.abs().sum() > 0
    gt_ign = gt.clone()
    gt_ign[:, 0, 0] = IGNORE_INDEX
    logits2 = torch.zeros(2, 1, 8, 8, requires_grad=True)
    DiceBCELoss()(logits2, gt_ign).backward()
    assert logits2.grad[:, 0, 0, 0].abs().sum() == 0   # no gradient from ignored pixels
