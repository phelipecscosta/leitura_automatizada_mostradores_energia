"""Testes das métricas e do registro do D3 (training/evaluation.py)."""

import json
from pathlib import Path

import numpy as np
import pytest

from training import evaluation as ev
from training.labels import MASKED, SCENE_METER_ONLY, LabeledPhoto

SCENE = np.array([0, 0, 1, 2, 2, SCENE_METER_ONLY, 0, 1])
ILLEGIBLE = np.array([0, 1, 0, MASKED, MASKED, 1, 0, 1])


def perfect():
    p_scene = np.full((8, 3), 0.05)
    for i, s in enumerate(SCENE):
        p_scene[i, s if s >= 0 else 0] = 0.9
    p_ill = np.where(ILLEGIBLE == 1, 0.95, 0.05)
    return p_scene, p_ill


def test_perfect_predictions_give_metric_one():
    metric, ap_outros, ap_ill = ev.optimizing_metric(*perfect(), SCENE, ILLEGIBLE)
    assert metric == pytest.approx(1.0) and ap_outros == 1.0 and ap_ill == 1.0


def test_log_loss_uses_partial_rule_and_masks_outros():
    p_scene, p_ill = perfect()
    scene_ll, leg_ll = ev.log_losses(p_scene, p_ill, SCENE, ILLEGIBLE)
    # indeterminado: p_digital + p_ciclométrico = 0,95; os exatos: 0,9
    expected = -(7 * np.log(0.9) + np.log(0.95)) / 8
    assert scene_ll == pytest.approx(expected)
    assert leg_ll == pytest.approx(-np.log(0.95))


def test_identical_models_have_zero_difference():
    model = perfect()
    point, low, high, valid = ev.paired_bootstrap(model, model, SCENE, ILLEGIBLE, n_boot=200)
    assert point == 0.0 and low == 0.0 and high == 0.0 and valid > 0


def test_random_sample_mask_excludes_stress_photos():
    photos = [
        LabeledPhoto("F-1", "2000-01-01", "treino", "rotulagem", Path("x"), 0, 0),
        LabeledPhoto("F-2", "2000-01-01", "treino", "estresse_zero", Path("x"), 0, 0),
    ]
    _, _, random_sample = ev.targets(photos)
    assert random_sample.tolist() == [True, False]


def test_record_run_writes_json_and_predictions(tmp_path):
    photos = [LabeledPhoto("F-1", "2000-01-01", "treino", "rotulagem", Path("x"), 0, 0)]
    path = ev.record_run(tmp_path, "teste", {"semente": 0}, {"metrica": 0.5},
                         photos, np.array([[0.7, 0.2, 0.1]]), np.array([0.3]))
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["configuracao"] == {"semente": 0} and record["commit_produto"]
    lines = path.with_suffix(".csv").read_text(encoding="utf-8").splitlines()
    assert lines[0].endswith("p_outros,p_ilegivel") and lines[1].startswith("F-1,")
