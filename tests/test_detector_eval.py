"""Testes da avaliação do detector (training/detector_eval.py)."""
import numpy as np

from training import detector_eval as de
from training.box_labels import Box
from training.triage_eval import cp_upper


def test_photos_with_field_ignores_tiny_boxes():
    boxes = [Box("F-000000000001", "display_digital", 0, 0, 50, 3),   # 3 px de altura: não conta
             Box("F-000000000002", "numero_serie", 0, 0, 40, 4),      # 4 px: conta
             Box("F-000000000002", "display_digital", 0, 0, 2, 2)]
    assert de.photos_with_field(boxes) == {"F-000000000002"}


def test_kish_with_equal_weights_is_the_raw_count():
    assert np.isclose(de.kish_n(np.full(10, 2.3)), 10)
    assert np.isclose(de.kish_n(np.array([1.0, 3.0])), 16 / 10)  # pesos desiguais: menos informação


def test_weighted_bound_with_equal_weights_matches_clopper_pearson():
    lost = np.array([True] + [False] * 199)
    expected = float(cp_upper(np.array(1), 200))
    assert np.isclose(de.weighted_cp_upper(lost, np.full(200, 2.0)), expected)


def test_loss_budget_follows_the_rule_of_three():
    assert de.loss_budget(500) == -1   # 0 perdas em 500: limite de cerca de 0,6%
    assert de.loss_budget(600) == 0    # 0 em 600: abaixo de 0,5%
    assert de.loss_budget(1000) == 1   # 1 perda exige cerca de 950; 2, cerca de 1.260

