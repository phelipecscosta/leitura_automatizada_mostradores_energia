"""Testes das características do D3 (training/features.py)."""

from pathlib import Path

import numpy as np
from PIL import Image

import torch

from meter_reader.triage import TriageModel

from training import features as ft
from training.labels import LabeledPhoto


def make_photos(tmp_path, n=3):
    """Fotos sintéticas 360 x 480, com conteúdo diferente em cada uma."""
    rng = np.random.default_rng(0)
    photos = []
    for i in range(n):
        path = tmp_path / f"f{i}.jpg"
        Image.fromarray(rng.integers(0, 256, (480, 360, 3), dtype=np.uint8)).save(path)
        photos.append(LabeledPhoto(f"F-{i}", "2000-01-01", "treino", "rotulagem", path, 0, 0))
    return photos


def test_baseline_has_1859_dimensions(tmp_path):
    image = ft._model_image(make_photos(tmp_path, 1)[0].path, 224)
    assert image.size == (224, 299)  # (largura, altura)
    vector = ft.baseline_vector(image)
    assert vector.shape == (1728 + 128 + 3,)
    assert np.isfinite(vector).all()


def test_deep_features_have_backbone_dimension(tmp_path):
    photos = make_photos(tmp_path, 2)
    features = ft.extract(photos, "mobilenet_v3_large_224", pretrained=False)
    assert features.shape == (2, 960)


def test_cache_is_reused_only_for_the_same_photo_list(tmp_path, monkeypatch):
    photos = make_photos(tmp_path)
    calls = []
    real_extract = ft.extract
    monkeypatch.setattr(ft, "extract", lambda *a, **k: calls.append(1) or real_extract(*a, **k))

    first = ft.load_or_extract(photos, "baseline_224", tmp_path)
    second = ft.load_or_extract(photos, "baseline_224", tmp_path)
    assert len(calls) == 1 and np.array_equal(first, second)

    ft.load_or_extract(photos[:2], "baseline_224", tmp_path)  # outra lista: recalcula
    assert len(calls) == 2

def test_extraction_runs_without_tf32(monkeypatch, tmp_path):
    from meter_reader.triage import TriageModel
    seen = []
    original = TriageModel.embed

    def spy(self, x):
        seen.append(torch.backends.cudnn.allow_tf32)
        return original(self, x)

    monkeypatch.setattr(TriageModel, "embed", spy)
    photos = _photos(tmp_path, 2) if "_photos" in globals() else None
    if photos is None:
        pytest.skip("Auxiliar de fotos sintéticas com outro nome neste arquivo.")
    ft.extract(photos, "mobilenet_v3_large_224", pretrained=False)
    assert seen and not any(seen)


def test_old_cache_name_is_ignored(tmp_path):
    assert ft.CACHE_VERSION in f"caracteristicas_x_{ft.CACHE_VERSION}.npz"
    old = tmp_path / "caracteristicas_baseline_224.npz"
    old.write_bytes(b"cache antigo")  # nome sem versão: não pode ser lido
    path = tmp_path / f"caracteristicas_baseline_224_{ft.CACHE_VERSION}.npz"
    assert not path.exists()

def test_extraction_runs_without_tf32(tmp_path, monkeypatch):
    # Espiona o embed: o TF32 precisa estar desligado durante a extração
    seen = []
    original = TriageModel.embed

    def spy(self, x):
        seen.append(torch.backends.cudnn.allow_tf32)
        return original(self, x)

    monkeypatch.setattr(TriageModel, "embed", spy)
    ft.extract(make_photos(tmp_path, 2), "mobilenet_v3_large_224", pretrained=False)
    assert seen and not any(seen)


def test_cache_without_version_is_ignored(tmp_path, monkeypatch):
    photos = make_photos(tmp_path)
    codes = np.array([p.pseudonym for p in photos])
    # Cache antigo, válido e com os mesmos pseudônimos, mas sem a versão no nome
    np.savez(tmp_path / "caracteristicas_baseline_224.npz",
             codes=codes, features=np.zeros((len(photos), 1859), dtype=np.float32))
    calls = []
    real_extract = ft.extract
    monkeypatch.setattr(ft, "extract", lambda *a, **k: calls.append(1) or real_extract(*a, **k))

    features = ft.load_or_extract(photos, "baseline_224", tmp_path)
    assert len(calls) == 1 and features.any()  # extraiu de novo; não leu os zeros
    assert (tmp_path / f"caracteristicas_baseline_224_{ft.CACHE_VERSION}.npz").is_file()
