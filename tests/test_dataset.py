import numpy as np

from src.dataset import IGNORE_INDEX, OxfordPetSeg, get_transforms, make_splits


def test_splits_are_disjoint_deterministic_and_stratified(fake_root):
    a, b = make_splits(fake_root, seed=0), make_splits(fake_root, seed=0)
    assert [r.name for r in a["val"]] == [r.name for r in b["val"]]
    tr, va = {r.name for r in a["train"]}, {r.name for r in a["val"]}
    assert not tr & va
    assert {r.breed for r in a["val"]} == {r.breed for r in a["train"] + a["val"]}


def test_item_shapes_and_mask_values(fake_root):
    ds = OxfordPetSeg(fake_root, split="train", image_size=64)
    item = ds[0]
    assert item["image"].shape == (3, 64, 64)
    assert item["mask"].shape == (64, 64)
    assert set(np.unique(item["mask"].numpy())) <= {0, 1, IGNORE_INDEX}


def test_augmentation_keeps_image_and_mask_in_sync(fake_root):
    """Automated version of show_augmented: paint the pet a known color, augment many times
    (without color jitter), and check that the mask still lands on that color."""
    ds = OxfordPetSeg(fake_root, split="train", image_size=64)
    image, trimap = ds.load_raw(0)
    image = image.copy()
    image[trimap == 1] = (255, 0, 0)
    image[trimap != 1] = (0, 0, 255)
    t = get_transforms(train=True, size=64)
    t.transforms = [op for op in t.transforms if type(op).__name__ != "ColorJitter"]
    import cv2

    from src.dataset import convert_trimap

    def red_fraction(img, m):
        # Erode by one pixel: bilinear image resampling blends colors along the edge, the mask does not.
        inner = cv2.erode((m == 1).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        return (img[inner][:, 0] > img[inner][:, 2]).mean() if inner.sum() > 20 else None

    for _ in range(20):
        out = t(image=image, mask=convert_trimap(trimap))
        f = red_fraction(out["image"], out["mask"])
        if f is not None:
            assert f > 0.99
    # The check must catch a real desync: flip the image but not the mask.
    out = t(image=image, mask=convert_trimap(trimap))
    assert red_fraction(out["image"][:, ::-1], out["mask"]) < 0.9
