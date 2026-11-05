"""Testes da avaliação comum da triagem de cinco classes (T4f.2)."""

import numpy as np
import pytest

from meter_reader.contracts import PRODUCT_CLASSES
from training import triage_eval as te
from training.group_labels import TRAINING_GROUP_CLASS, TRAINING_GROUPS

RNG = np.random.default_rng(0)


def test_class_probs_are_sums_of_group_probs():
    logits = RNG.normal(size=(6, len(TRAINING_GROUPS)))
    p_group = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    expected = np.stack([p_group[:, np.asarray(TRAINING_GROUP_CLASS) == c].sum(axis=1)
                         for c in range(len(PRODUCT_CLASSES))], axis=1)
    got = np.exp(te.class_log_probs(logits))
    assert np.allclose(got, expected) and np.allclose(got.sum(axis=1), 1)


def test_extreme_logits_stay_finite():
    logits = np.zeros((1, len(TRAINING_GROUPS)))
    logits[0, 0] = 1000.0  # sem logsumexp, exp(1000) estouraria
    assert np.isfinite(te.class_log_probs(logits)).all()


def test_broad_mapping_is_plain_softmax():
    logits = RNG.normal(size=(4, 5))
    p = np.exp(te.class_log_probs(logits, output_class=te.BROAD_CLASS))
    assert np.allclose(p, np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True))


def test_nested_temperature_recovers_known_scale():
    # Logits "verdadeiros" multiplicados por 3: a temperatura certa é perto de 3
    n = 3000
    true = RNG.normal(scale=2.0, size=(n, len(TRAINING_GROUPS)))
    p = np.exp(te.class_log_probs(true))
    labels = np.array([RNG.choice(5, p=row) for row in p])
    lots = np.repeat(["a", "b", "c"], n // 3)
    result = te.calibrate_nested(true * 3.0, labels, lots)
    assert result.final == pytest.approx(3.0, rel=0.15)
    assert set(result.per_lot) == {"a", "b", "c"} and not np.isnan(result.probs).any()


def test_score_and_destination():
    probs = np.array([[0.90, 0.05, 0.02, 0.01, 0.02],
                      [0.10, 0.00, 0.10, 0.60, 0.20]])
    assert np.allclose(te.rejection_score(probs), [0.05, 0.90])
    assert PRODUCT_CLASSES[te.rejection_destination(probs)[1]] == "multiplos"


from training.risk_coverage import clopper_pearson_upper  # noqa: E402

# Rótulos: 0 e 1 determinados; 2, 3 e 4 rejeitáveis
LABELS = np.array([0, 0, 1, 0, 2, 4, 3, 4])
SCORE = np.array([0.1, 0.2, 0.9, 0.2, 0.95, 0.7, 0.9, 0.3])


def test_cp_upper_matches_bisection_version():
    for k, n in [(0, 10), (3, 50), (39, 7800), (10, 10)]:
        assert float(te.cp_upper(np.array([k]), n)[0]) == pytest.approx(
            clopper_pearson_upper(k, n), abs=1e-9)


def test_curve_counts_losses_coverage_and_ties():
    curve = te.loss_coverage_curve(SCORE, LABELS)
    # Limiares distintos, do maior ao menor; o empate 0,9 rejeita duas fotos de uma vez
    assert curve["threshold"].tolist() == [0.95, 0.9, 0.7, 0.3, 0.2, 0.1]
    row = curve[curve["threshold"] == 0.9].iloc[0]
    assert row["losses"] == 1 and row["coverage"] == pytest.approx(2 / 4)


def test_choose_threshold_respects_target_or_returns_none():
    curve = te.loss_coverage_curve(SCORE, LABELS)
    best = te.choose_threshold(curve, max_loss=1.0)      # meta folgada: aceita tudo
    assert best["coverage"] == 1.0 and best["threshold"] == 0.3  # maior limiar empatado
    assert te.choose_threshold(curve, max_loss=0.0) is None  # nada cumpre perda zero


def test_paired_diff_is_zero_for_same_model_and_sign_is_right():
    point, low, high = te.paired_coverage_diff(SCORE, 0.9, SCORE, 0.9, LABELS, n_boot=200)
    assert point == low == high == 0.0
    point, low, high = te.paired_coverage_diff(SCORE, 0.3, SCORE, 0.95, LABELS, n_boot=200)
    assert point == pytest.approx(3 / 4) and low <= point <= high


import json  # noqa: E402

CODES = ["F-00000000000a", "F-00000000000b"]
LOTS = ["2026-05-20", "2026-05-21"]


def test_save_and_load_average_seeds(tmp_path):
    logits = np.stack([np.zeros((2, 12)), np.full((2, 12), 2.0)])  # duas sementes
    path = te.save_oof(tmp_path, "c0_teste", CODES, LOTS, logits, {"lr": 1}, "a" * 64)
    record = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert record["sementes"] == 2 and record["sha256_rotulos"] == "a" * 64
    assert np.allclose(te.load_oof(tmp_path, "c0_teste", CODES), 1.0)


def test_save_refuses_overwrite_nan_and_wrong_shape(tmp_path):
    logits = np.zeros((1, 2, 12))
    te.save_oof(tmp_path, "c0", CODES, LOTS, logits, {}, "a" * 64)
    with pytest.raises(FileExistsError):
        te.save_oof(tmp_path, "c0", CODES, LOTS, logits, {}, "a" * 64)
    with pytest.raises(ValueError, match="sem logit"):
        te.save_oof(tmp_path, "c1", CODES, LOTS, np.full((1, 2, 12), np.nan), {}, "a" * 64)
    with pytest.raises(ValueError, match="forma"):
        te.save_oof(tmp_path, "c2", CODES, LOTS, np.zeros((2, 12)), {}, "a" * 64)


def test_load_refuses_other_photo_list(tmp_path):
    te.save_oof(tmp_path, "c0", CODES, LOTS, np.zeros((1, 2, 12)), {}, "a" * 64)
    with pytest.raises(ValueError, match="pseudônimos"):
        te.load_oof(tmp_path, "c0", CODES[::-1])
