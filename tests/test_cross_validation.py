"""Testes da validação por lote deixado de fora (training/cross_validation.py)."""

from pathlib import Path

import numpy as np
import pytest

from training import cross_validation as cv
from training.labels import MASKED, SCENE_METER_ONLY, LabeledPhoto

LOTS = ("2000-01-01", "2000-01-02", "2000-01-03")


def synthetic(n_per_lot=12, subset="treino"):
    """Características que separam as classes; uma foto de cada tipo por bloco."""
    rng = np.random.default_rng(0)
    cycle = [(0, 0), (1, 1), (2, MASKED), (SCENE_METER_ONLY, 1)]
    photos, rows = [], []
    for lot in LOTS:
        for i in range(n_per_lot):
            scene, illegible = cycle[i % len(cycle)]
            photos.append(LabeledPhoto(f"F-{lot}-{i}", lot, subset, "rotulagem",
                                       Path("x.jpg"), scene, illegible))
            center = np.eye(4)[scene if scene >= 0 else 3] * 3
            rows.append(center + rng.normal(0, 0.1, 4))
    return np.array(rows, dtype=np.float32), photos


def test_every_photo_gets_an_out_of_fold_prediction():
    features, photos = synthetic()
    p_scene, p_ill, folds = cv.cross_validate(features, photos, seed=0)
    assert not np.isnan(p_scene).any() and not np.isnan(p_ill).any()
    assert np.allclose(p_scene.sum(axis=1), 1.0, atol=1e-5)
    assert [f.held_out_lot for f in folds] == list(LOTS)
    assert all(f.n_train == 24 for f in folds)


def test_separable_data_is_learned_out_of_fold():
    features, photos = synthetic()
    p_scene, _, _ = cv.cross_validate(features, photos, seed=0)
    exact = np.array([p.scene >= 0 for p in photos])
    truth = np.array([p.scene for p in photos])
    assert (p_scene[exact].argmax(axis=1) == truth[exact]).mean() > 0.95


def test_same_seed_is_deterministic_and_fraction_shrinks_training():
    features, photos = synthetic()
    a, _, _ = cv.cross_validate(features, photos, seed=1, train_fraction=0.5)
    b, _, folds = cv.cross_validate(features, photos, seed=1, train_fraction=0.5)
    assert np.array_equal(a, b)
    assert all(f.n_train == 12 for f in folds)


def test_test_subset_is_refused():
    features, photos = synthetic(subset="teste")
    with pytest.raises(ValueError):
        cv.cross_validate(features, photos, seed=0)

def test_probabilities_are_the_activation_of_the_logits():
    features, photos = synthetic()
    s_logits, l_logits, _ = cv.cross_validate_logits(features, photos, seed=0)
    p_scene, p_ill, _ = cv.cross_validate(features, photos, seed=0)
    assert np.isfinite(s_logits).all() and np.isfinite(l_logits).all()
    # Softmax e sigmoide calculadas à parte, em NumPy
    exp = np.exp(s_logits - s_logits.max(axis=1, keepdims=True))
    assert np.allclose(p_scene, exp / exp.sum(axis=1, keepdims=True), atol=1e-6)
    assert np.allclose(p_ill, 1 / (1 + np.exp(-l_logits)), atol=1e-6)
