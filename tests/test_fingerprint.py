"""Testes da assinatura visual (imagens sintéticas geradas no teste)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from training.fingerprint import dhash, hamming

RNG = np.random.default_rng(0)


def _scene(seed: int) -> Image.Image:
    """Imagem com estrutura suave: blocos aleatórios ampliados e suavizados."""
    small = np.random.default_rng(seed).integers(0, 256, size=(8, 9), dtype=np.uint8)
    return Image.fromarray(small).resize((640, 480), Image.Resampling.BICUBIC).convert("RGB")


def _save(image: Image.Image, path: Path, quality: int = 90) -> Path:
    image.save(path, quality=quality)
    return path


def test_hash_has_64_bits_in_hex(tmp_path: Path) -> None:
    value = dhash(_save(_scene(1), tmp_path / "a.jpg"))
    assert len(value) == 16
    int(value, 16)  # é hexadecimal válido


def test_same_image_has_distance_zero(tmp_path: Path) -> None:
    a = dhash(_save(_scene(1), tmp_path / "a.jpg"))
    b = dhash(_save(_scene(1), tmp_path / "b.jpg"))
    assert hamming(a, b) == 0


def test_recompressed_copy_is_near(tmp_path: Path) -> None:
    original = dhash(_save(_scene(1), tmp_path / "a.jpg", quality=95))
    recompressed = dhash(_save(_scene(1), tmp_path / "b.jpg", quality=40))
    assert hamming(original, recompressed) <= 4


def test_resized_copy_is_near(tmp_path: Path) -> None:
    original = dhash(_save(_scene(1), tmp_path / "a.jpg"))
    smaller = dhash(_save(_scene(1).resize((480, 360)), tmp_path / "b.jpg"))
    assert hamming(original, smaller) <= 4


def test_different_images_are_far(tmp_path: Path) -> None:
    a = dhash(_save(_scene(1), tmp_path / "a.jpg"))
    b = dhash(_save(_scene(2), tmp_path / "b.jpg"))
    assert hamming(a, b) >= 16


def test_hamming_counts_differing_bits() -> None:
    assert hamming("0000000000000000", "0000000000000000") == 0
    assert hamming("0000000000000000", "000000000000000f") == 4
    assert hamming("0000000000000000", "ffffffffffffffff") == 64
