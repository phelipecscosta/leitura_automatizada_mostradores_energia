"""Testes do extrator das vistas da triagem de cinco classes (T4f.3)."""

import numpy as np
import pytest
import torch
from PIL import Image

from training import triage_features as tf

DIMS = {"mobilenet_v3_large_224": 960, "dinov2_224": 384}


def same(a, b) -> bool:
    """Diferença relativa do vetor inteiro (como na E33): vale para qualquer escala.

    Elemento a elemento, um valor quase zero reprovaria uma diferença desprezível
    de arredondamento (o tamanho do lote muda a ordem das somas na convolução).
    """
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return np.linalg.norm(a - b) / np.linalg.norm(a) < 1e-4


def extract(*args, **kwargs):
    # Pesos aleatórios iguais a cada chamada; sem isso, cada modelo seria outro
    torch.manual_seed(0)
    return tf.extract_views(*args, pretrained=False, **kwargs)


def cached(*args):
    torch.manual_seed(0)
    return tf.load_or_extract(*args, pretrained=False)


@pytest.fixture
def photos(tmp_path):
    rng = np.random.default_rng(0)
    paths = []
    for i in range(2):
        path = tmp_path / f"foto{i}.jpg"
        Image.fromarray(rng.integers(0, 255, (480, 360, 3), dtype=np.uint8)).save(path)
        paths.append(path)
    return paths, ["F-00000000000a", "F-00000000000b"]


@pytest.mark.parametrize("name", list(tf.EXTRACTORS))
def test_shape_and_original_first(photos, name):
    paths, codes = photos
    feats = extract(paths, codes, name, n_views=2)
    assert feats.shape == (2, 3, DIMS[name])
    # A posição 0 é a foto original: igual à extração sem vistas
    assert same(feats[:, 0], extract(paths, codes, name, n_views=0)[:, 0])
    # O extrator usa as vistas, e não repete a original: no mesmo lote, a mesma
    # entrada daria saída idêntica. (Com pesos aleatórios, o ViT é quase
    # insensível à entrada; a diferença real das vistas é testada em test_triage_augment.)
    assert not np.array_equal(feats[:, 0], feats[:, 1])


def test_cache_is_reused_and_checked_by_codes(photos, tmp_path):
    paths, codes = photos
    name = "mobilenet_v3_large_224"
    first = cached(paths, codes, name, 1, tmp_path)
    stamp = tf.cache_path(tmp_path, name, 1).stat().st_mtime_ns
    assert np.array_equal(first, cached(paths, codes, name, 1, tmp_path))
    assert tf.cache_path(tmp_path, name, 1).stat().st_mtime_ns == stamp  # não reextraiu
    # Outra lista de fotos nunca usa o cache antigo: reextrai na nova ordem
    other = cached(paths[::-1], codes[::-1], name, 1, tmp_path)
    assert same(other[0], first[1]) and same(other[1], first[0])
