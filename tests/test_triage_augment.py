"""Testes da aumentação única da triagem (revisão R2)."""

import numpy as np
from PIL import Image

from training.triage_augment import augment, photo_views

W, H = 360, 480  # resolução nativa das fotos (retrato)


def two_meters() -> Image.Image:
    """Fundo claro com dois "medidores" escuros perto das bordas laterais."""
    pixels = np.full((H, W, 3), 200, dtype=np.uint8)
    for x0 in (int(0.10 * W), int(0.85 * W)):  # 10% da borda esquerda e 15% da direita
        pixels[H // 2 - 20:H // 2 + 20, x0:x0 + 18] = 20
    return Image.fromarray(pixels)


def test_same_seed_same_view_and_size_kept():
    image = two_meters()
    a, b = augment(image, 7), augment(image, 7)
    assert np.array_equal(np.asarray(a), np.asarray(b))
    assert a.size == image.size
    assert not np.array_equal(np.asarray(a), np.asarray(augment(image, 8)))


def test_no_view_cuts_a_meter_near_the_edge():
    # Sem recorte: em 50 sementes, os dois medidores continuam visíveis
    image = two_meters()
    for seed in range(50):
        gray = np.asarray(augment(image, seed).convert("L"))
        assert (gray[:, : W // 3] < 100).sum() > 200      # medidor da esquerda
        assert (gray[:, 2 * W // 3:] < 100).sum() > 200   # medidor da direita


def test_photo_views_starts_with_original_enhanced():
    views = photo_views(two_meters(), "F-00000000000a", 5)
    assert len(views) == 6
    assert np.array_equal(np.asarray(views[0]), np.asarray(photo_views(two_meters(), "x", 0)[0]))
    # Vistas fixas por pseudônimo (R1): outra foto gera outras vistas
    other = photo_views(two_meters(), "F-00000000000b", 5)
    assert not np.array_equal(np.asarray(views[1]), np.asarray(other[1]))

    
