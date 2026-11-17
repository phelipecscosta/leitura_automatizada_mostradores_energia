"""Testes da localização e da segunda porta (training/detector_eval.py)."""
import numpy as np
import pytest

from meter_reader.detector import meter_family
from training import detector_eval as de
from training.detector_dataset import DetectorPhoto


def _photo(code, boxes, labels, weight=1.0):
    return DetectorPhoto(code, "desenvolvimento", "2026-05-20", weight, "g",
                         tuple(map(tuple, boxes)), tuple(labels))


def _pred(boxes, scores, labels):
    return {"boxes": np.array(boxes, float).reshape(-1, 4), "scores": np.array(scores, float),
            "labels": np.array(labels, int)}


def test_iou_matrix_known_cases():
    iou = de.iou_matrix([[0, 0, 10, 10]], [[0, 0, 10, 10], [0, 0, 5, 10], [20, 20, 30, 30]])
    assert np.allclose(iou, [[1.0, 0.5, 0.0]])


def test_match_is_greedy_by_score_and_one_to_one():
    gt = np.array([[0, 0, 10, 10]], float)
    preds = np.array([[0, 0, 10, 10], [0, 0, 9, 10]], float)
    tp, matched = de.match_photo(preds, np.array([0.4, 0.9]), gt)
    assert tp.tolist() == [False, True] and matched.tolist() == [1]  # a de maior confiança casa
    tp, _ = de.match_photo(np.array([[0, 0, 4, 10]], float), np.array([0.9]), gt)
    assert tp.tolist() == [False]  # IoU 0,4: não acerta


def test_perfect_detector_has_ap_one_and_full_coverage():
    photos = [_photo("F-000000000001", [[0, 0, 10, 10]], [1]),
              _photo("F-000000000002", [[5, 5, 25, 15]], [4])]
    preds = [_pred([[0, 0, 10, 10]], [0.9], [1]), _pred([[5, 5, 25, 15]], [0.8], [4])]
    ap, matches = de.ap_table(photos, preds)
    assert ap["display_digital"] == pytest.approx(1.0) and ap["numero_serie"] == pytest.approx(1.0)
    assert np.isnan(ap["display_kvarh"])  # classe sem anotadas
    cov = de.coverage_summary(matches)
    assert cov.loc["display_digital", "cobertura_p50"] == 1.0


def test_weighted_ap_counts_photo_weights():
    # Duas fotos com um campo cada; o detector só acha o da foto 1
    photos = [_photo("F-000000000001", [[0, 0, 10, 10]], [1], weight=1.0),
              _photo("F-000000000002", [[0, 0, 10, 10]], [1], weight=3.0)]
    preds = [_pred([[0, 0, 10, 10]], [0.9], [1]), _pred([], [], [])]
    m = de.match_class(photos, preds, 1)
    # Revocação máxima: 1 ÷ 4 do peso; precisão 1 até ela: AP = 26 pontos de 101 com precisão 1
    assert de.weighted_ap(m, np.array([1.0, 3.0])) == pytest.approx(26 / 101)
    assert de.weighted_ap(m, np.array([3.0, 1.0])) == pytest.approx(76 / 101)


def test_wrong_class_is_a_miss_and_coverage_reports_a_short_box():
    photos = [_photo("F-000000000001", [[0, 0, 100, 10]], [4])]
    preds = [_pred([[0, 0, 100, 10]], [0.9], [1]), ]  # classe errada
    assert de.ap_table(photos, preds)[0]["numero_serie"] == 0.0
    preds = [_pred([[0, 0, 80, 10]], [0.9], [4])]   # cobre 80% do campo, IoU 0,8
    _, matches = de.ap_table(photos, preds)
    assert de.coverage_summary(matches).loc["numero_serie", "cobertura_p50"] == 0.8


def test_bootstrap_of_identical_detectors_is_zero():
    photos = [_photo(f"F-00000000000{i}", [[0, 0, 10, 10]], [4]) for i in range(5)]
    preds = [_pred([[0, 0, 10, 10]], [0.5 + i / 10], [4]) for i in range(5)]
    _, m = de.ap_table(photos, preds)
    diffs = de.bootstrap_selection_diff(photos, m, m, n_boot=20)
    assert np.allclose(np.nan_to_num(diffs), 0.0)


def test_gate_curve_and_threshold():
    # 4 fotos com campo e 2 sem campo; escores das sem campo são os menores
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.2, 0.1])
    has = np.array([True, True, True, True, False, False])
    curve = de.gate_curve(scores, has, np.ones(6))
    first = curve.iloc[0]
    assert first["limiar"] == 0 and first["perda"] == 0 and first["cobertura"] == 0
    row = curve[curve["limiar"] == 0.6].iloc[0]   # rejeita as duas sem campo
    assert row["perda"] == 0 and row["cobertura"] == 1.0
    row = curve[curve["limiar"] == 0.7].iloc[0]   # rejeita também uma com campo
    assert row["perda"] == pytest.approx(0.25)
    # Com 4 fotos, nem zero perdas prova 0,5%: nenhum limiar cumpre a meta
    assert de.choose_gate_threshold(curve) is None
    assert de.choose_gate_threshold(curve, max_loss=0.6) == 0.6


def test_gate_threshold_with_enough_photos_rejects_only_the_empty_ones():
    has = np.array([True] * 2000 + [False] * 50)
    scores = np.where(has, 0.9, 0.05)
    curve = de.gate_curve(scores, has, np.ones(len(has)))
    t = de.choose_gate_threshold(curve)
    assert t == 0.9 and curve[curve["limiar"] == t].iloc[0]["cobertura"] == 1.0


def test_photo_score_is_the_largest_box_score():
    assert de.photo_score(_pred([[0, 0, 1, 1]] * 2, [0.3, 0.7], [4, 1])) == 0.7
    assert de.photo_score(_pred([], [], [])) == 0.0


def test_meter_family_follows_the_largest_display_above_the_threshold():
    assert meter_family([1, 2], [0.9, 0.8], 0.5) == "digital"
    assert meter_family([1, 3], [0.6, 0.8], 0.5) == "ciclometrico"   # kvarh conta como ciclométrico
    assert meter_family([4], [0.99], 0.5) is None                     # só número de série (B1)
    assert meter_family([1, 4], [0.4, 0.9], 0.5) is None              # display abaixo do limiar
