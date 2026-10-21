"""Testes do pré-processamento e das vistas do DINOv2, com imagens sintéticas."""

import numpy as np
import pytest
import torch
from PIL import Image

from training.dino_features import (
    PATCH,
    augment,
    crop_to_multiple,
    model_input,
    photo_inputs,
    view_seed,
)


def _photo(width: int = 360, height: int = 480, seed: int = 0) -> Image.Image:
    """Foto sintética com ruído, no tamanho das fotos da base."""
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))


@pytest.mark.parametrize(("size", "expected"), [
    ((224, 299), (224, 294)),   # retrato: só a altura é cortada
    ((299, 224), (294, 224)),   # paisagem: só a largura
    ((224, 280), (224, 280)),   # já múltiplo de 14: não muda
])
def test_crop_to_multiple(size, expected) -> None:
    assert crop_to_multiple(Image.new("RGB", size)).size == expected


def test_model_input_shape() -> None:
    # 360 x 480 -> 224 x 299 -> 224 x 294; tensor (canais, altura, largura)
    x = model_input(_photo())
    assert x.shape == (3, 294, 224)
    assert x.shape[1] % PATCH == 0 and x.shape[2] % PATCH == 0


def test_view_seed_is_stable_and_distinct() -> None:
    assert view_seed("F-abc", 1) == view_seed("F-abc", 1)
    assert len({view_seed("F-abc", v) for v in range(1, 6)}) == 5
    assert view_seed("F-abc", 1) != view_seed("F-abd", 1)


def test_augment_is_deterministic_by_seed() -> None:
    photo = _photo()
    same_a, same_b = np.asarray(augment(photo, 7)), np.asarray(augment(photo, 7))
    other = np.asarray(augment(photo, 8))
    assert np.array_equal(same_a, same_b)
    assert not np.array_equal(same_a, other)


def test_augment_keeps_area_and_proportion() -> None:
    for seed in range(20):
        width, height = augment(_photo(), seed).size
        assert 0.80 * 360 * 480 - 2000 <= width * height <= 360 * 480
        assert abs(width / height - 360 / 480) < 0.01


def test_photo_inputs_original_first_and_same_shape() -> None:
    photo = _photo()
    x = photo_inputs(photo, "F-abc", n_views=5)
    assert x.shape == (6, 3, 294, 224)          # todas as vistas com o mesmo tamanho
    assert torch.equal(x[0], photo_inputs(photo, "F-abc", n_views=0)[0])
    assert torch.equal(x, photo_inputs(photo, "F-abc", n_views=5))  # determinístico
    assert not torch.equal(x[0], x[1])          # a vista difere da original
