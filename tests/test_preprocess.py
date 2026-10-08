"""Testes do pré-processamento da triagem (meter_reader/preprocess.py)."""

import numpy as np
import pytest
import torch
from PIL import Image

from meter_reader import preprocess as pp


def low_contrast_photo(tmp_path):
    """Foto sintética 360 x 480 (retrato), com valores só entre 100 e 140."""
    rng = np.random.default_rng(0)
    pixels = rng.integers(100, 141, size=(480, 360, 3), dtype=np.uint8)
    path = tmp_path / "sintetica.jpg"
    Image.fromarray(pixels).save(path, quality=95)
    return path


def test_enhance_stretches_the_histogram(tmp_path):
    with Image.open(low_contrast_photo(tmp_path)) as image:
        before = np.asarray(image.convert("RGB"))
        after = np.asarray(pp.enhance(image))
    assert after.max() - after.min() > before.max() - before.min()
    assert after.min() <= 5 and after.max() >= 250


@pytest.mark.parametrize(("short_side", "shape"), [
    (None, (3, 480, 360)),  # nativa: altura 480, largura 360
    (224, (3, 299, 224)),   # lado menor 224, proporção 3:4 mantida
])
def test_output_shape_keeps_aspect_ratio(tmp_path, short_side, shape):
    tensor = pp.preprocess(low_contrast_photo(tmp_path), short_side)
    assert tuple(tensor.shape) == shape
    assert tensor.dtype == torch.float32


def test_normalization_uses_imagenet_statistics():
    gray = Image.new("RGB", (4, 4), (128, 128, 128))
    tensor = pp.to_tensor(gray)
    expected = [(128 / 255 - m) / s for m, s in zip(pp.IMAGENET_MEAN, pp.IMAGENET_STD)]
    assert tensor[:, 0, 0].tolist() == pytest.approx(expected, abs=1e-6)
