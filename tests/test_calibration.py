"""Testes da calibração por temperatura (training/calibration.py)."""

import numpy as np

from training import calibration as cal
from training.labels import MASKED, SCENE_METER_ONLY


def test_recovers_known_temperature_in_both_heads():
    # Rótulos sorteados de logits verdadeiros; o modelo recebe os logits x3
    rng = np.random.default_rng(0)
    z = rng.normal(0, 1.5, (5000, 3))
    p = np.exp(cal._log_softmax(z))
    scene = np.array([rng.choice(3, p=pi) for pi in p])
    assert abs(cal.fit_temperature(cal.scene_nll, z * 3, scene) - 3) < 0.3

    zl = rng.normal(0, 2, 5000)
    y = (rng.random(5000) < 1 / (1 + np.exp(-zl))).astype(int)
    assert abs(cal.fit_temperature(cal.legibility_nll, zl * 3, y) - 3) < 0.3


def test_partial_label_uses_meter_probability():
    logits = np.array([[1.0, 2.0, 0.5]])
    p = np.exp(cal._log_softmax(logits))[0]
    nll = cal.scene_nll(logits, np.array([SCENE_METER_ONLY]))
    assert np.isclose(nll, -np.log(p[0] + p[1]))


def test_masked_photos_do_not_count_in_legibility():
    logits = np.array([2.0, -50.0])
    with_masked = cal.legibility_nll(logits, np.array([1, MASKED]))
    alone = cal.legibility_nll(logits[:1], np.array([1]))
    assert np.isclose(with_masked, alone)


def test_temperature_one_for_already_calibrated_logits():
    rng = np.random.default_rng(1)
    zl = rng.normal(0, 2, 20000)
    y = (rng.random(20000) < 1 / (1 + np.exp(-zl))).astype(int)
    assert abs(cal.fit_temperature(cal.legibility_nll, zl, y) - 1) < 0.1
