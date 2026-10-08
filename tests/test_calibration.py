"""Testes da calibração por temperatura (training/calibration.py)."""

import numpy as np
import pytest
import torch

from training import triage_loss as tl
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

def test_ece_hand_example():
    # Faixa 9: confiança 0,9, frequência 0,5. Faixa 1: confiança 0,1, frequência 0
    prob = np.array([0.9, 0.9, 0.1, 0.1])
    label = np.array([1, 0, 0, 0])
    assert np.isclose(cal.ece(prob, label), 0.5 * 0.4 + 0.5 * 0.1)


def test_ece_small_when_calibrated_and_large_when_overconfident():
    rng = np.random.default_rng(2)
    z = rng.normal(0, 2, 20000)
    y = (rng.random(20000) < 1 / (1 + np.exp(-z))).astype(int)
    assert cal.ece(cal.legibility_probabilities(z), y) < 0.02
    assert cal.ece(cal.legibility_probabilities(z * 4), y) > 0.08


def test_probabilities_are_valid_and_temperature_softens():
    logits = np.array([[4.0, 0.0, -4.0]])
    hot = cal.scene_probabilities(logits, temperature=3.0)
    assert np.isclose(hot.sum(), 1.0)
    assert hot.max() < cal.scene_probabilities(logits).max()

def _synthetic_lots(scales, n=3000, seed=3):
    """Um bloco por lote; logits verdadeiros multiplicados pela escala do lote."""
    rng = np.random.default_rng(seed)
    s_logits, l_logits, scene, ill, lots = [], [], [], [], []
    for k, scale in enumerate(scales):
        z = rng.normal(0, 1.5, (n, 3))
        p = np.exp(cal._log_softmax(z))
        scene.append([rng.choice(3, p=pi) for pi in p])
        zl = rng.normal(0, 2, n)
        ill.append((rng.random(n) < 1 / (1 + np.exp(-zl))).astype(int))
        s_logits.append(z * scale)
        l_logits.append(zl * scale)
        lots.append([f"lote{k}"] * n)
    return (np.vstack(s_logits), np.concatenate(l_logits), np.concatenate(scene),
            np.concatenate(ill), np.concatenate(lots))


def test_held_out_lot_uses_temperature_from_other_lots():
    # lote0 sem exagero; lote1 e lote2 exagerados x3
    s, l, scene, ill, lots = _synthetic_lots(scales=(1, 3, 3))
    result = cal.calibrate_nested(s, l, scene, ill, lots)
    t_scene, t_leg = result.per_lot["lote0"]
    assert abs(t_scene - 3) < 0.4 and abs(t_leg - 3) < 0.4
    assert not np.isnan(result.p_scene).any() and not np.isnan(result.p_illegible).any()


def test_final_temperature_uses_all_lots_and_fixes_overconfidence():
    s, l, scene, ill, lots = _synthetic_lots(scales=(3, 3, 3))
    result = cal.calibrate_nested(s, l, scene, ill, lots)
    assert all(abs(t - 3) < 0.3 for t in result.final)
    before = cal.ece(cal.legibility_probabilities(l), ill)
    after = cal.ece(result.p_illegible, ill)
    assert after < before / 3

def test_log_weights_match_training_weights():
    scene = np.array([0, 0, 0, 1, 2, SCENE_METER_ONLY])
    ill = np.array([0, 1, 0, 0, MASKED, 1])
    log_w, log_pw = cal.log_weights(scene, ill)
    expected = tl.scene_weights(torch.as_tensor(scene))[:3].double().numpy()
    assert np.allclose(log_w, np.log(expected))
    assert np.isclose(log_pw, np.log(tl.legibility_pos_weight(torch.as_tensor(ill)).item()))


def test_fold_offsets_ignore_the_held_out_lot():
    scene = np.array([2, 2, 2, 0, 1, 0, 1, 0, 2])
    ill = np.array([MASKED, MASKED, MASKED, 0, 1, 1, 0, 0, MASKED])
    lots = np.array(["A", "A", "A", "B", "B", "B", "C", "C", "C"])
    scene_off, ill_off = cal.fold_offsets(scene, ill, lots)
    log_w, log_pw = cal.log_weights(scene[lots != "A"], ill[lots != "A"])
    assert np.allclose(scene_off[lots == "A"], log_w)
    assert np.allclose(ill_off[lots == "A"], log_pw)


def test_correction_recovers_unweighted_probability():
    # Com peso w, o mínimo da perda ponderada fica em q = w·p / (w·p + 1 − p)
    p, w = 0.2, 3.0
    q = w * p / (w * p + 1 - p)
    corrected = np.log(q / (1 - q)) - np.log(w)
    assert np.isclose(corrected, np.log(p / (1 - p)))


def test_missing_class_is_refused():
    with pytest.raises(ValueError):
        cal.log_weights(np.array([0, 0, 1]), np.array([0, 1, 0]))
