import io

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from mlops.tasks.vision import (
    HAZE_LEVELS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    VisionModel,
    apply_haze,
    build_model,
    build_slices,
    preprocess,
)
from mlops.tasks.vision import TASK as VISION


def textured_image(seed=0, size=64):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (size, size, 3), dtype=np.uint8), "RGB")


def png_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_zero_haze_keeps_pixels():
    image = textured_image()

    assert np.array_equal(np.asarray(apply_haze(image, 0.0)), np.asarray(image))


def test_stronger_haze_is_brighter_and_flatter():
    image = textured_image()

    hazed = [np.asarray(apply_haze(image, s), dtype=float) for s in (0.0, 0.25, 0.45, 0.65)]

    means = [h.mean() for h in hazed]
    spreads = [h.std() for h in hazed]
    assert means == sorted(means)
    assert spreads == sorted(spreads, reverse=True)


def test_haze_is_deterministic():
    image = textured_image()

    assert np.array_equal(np.asarray(apply_haze(image, 0.45)), np.asarray(apply_haze(image, 0.45)))


@pytest.mark.parametrize("strength", [-0.1, 1.0])
def test_haze_strength_must_be_in_range(strength):
    with pytest.raises(ValueError, match="haze"):
        apply_haze(textured_image(), strength)


def test_preprocess_returns_normalized_nchw_float32():
    image = textured_image()

    batch = preprocess([png_bytes(image)], image_size=64)

    assert batch.shape == (1, 3, 64, 64)
    assert batch.dtype == np.float32
    pixel = np.asarray(image)[5, 7] / 255.0
    expected = (pixel - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    assert np.allclose(batch[0, :, 5, 7], expected, atol=1e-5)


def test_preprocess_resizes_to_the_requested_size():
    assert preprocess([png_bytes(textured_image())], image_size=32).shape == (1, 3, 32, 32)


def make_split(prefix, n):
    images = [png_bytes(textured_image(seed=i, size=8)) for i in range(n)]
    return pd.DataFrame(
        {
            "image_id": [f"{prefix}{i}" for i in range(n)],
            "image": images,
            "label": [i % 10 for i in range(n)],
        }
    )


def test_test_split_becomes_disjoint_slices_with_increasing_haze():
    test = make_split("test", 40)

    slices = build_slices(make_split("train", 4), make_split("valid", 4), test)

    assert tuple(slices) == VISION.slices
    period_ids = [set(slices[name]["image_id"]) for name in HAZE_LEVELS]
    assert set().union(*period_ids) == set(test["image_id"])
    assert sum(len(ids) for ids in period_ids) == len(test)
    assert [slices[name]["haze"].iloc[0] for name in HAZE_LEVELS] == list(HAZE_LEVELS.values())


def test_hazed_slices_store_hazed_images():
    slices = build_slices(make_split("train", 4), make_split("valid", 4), make_split("test", 40))

    clean = Image.open(io.BytesIO(slices["t0"]["image"].iloc[0]))
    hazed = Image.open(io.BytesIO(slices["t3"]["image"].iloc[0]))
    assert np.asarray(hazed, dtype=float).std() < np.asarray(clean, dtype=float).std()


def test_model_head_predicts_every_class():
    import torch

    net = build_model(num_classes=10, pretrained=False).eval()

    with torch.inference_mode():
        logits = net(torch.zeros(2, 3, 32, 32))

    assert logits.shape == (2, 10)


def test_exported_onnx_model_matches_torch(tmp_path):
    model = VisionModel(net=build_model(num_classes=10, pretrained=False).eval(), image_size=32)
    df = make_split("x", 6)

    VISION.export(model, tmp_path)

    assert (tmp_path / "model.onnx").exists()
    assert np.array_equal(VISION.predict_exported(tmp_path, df), VISION.predict(model, df))
