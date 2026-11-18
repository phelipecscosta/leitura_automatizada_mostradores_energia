"""Testes das métricas da leitura do teste do detector (T5c.5), com fotos sintéticas."""
import numpy as np
import pytest

from training.detector_dataset import DetectorPhoto
from training.detector_holdout import evaluate

DIG, CYC, SER = 1, 2, 4
PHOTOS = [
    DetectorPhoto("F-000000000001", "teste", "2026-07-03", 1.0, "g",
                  ((100.0, 50.0, 200.0, 80.0), (110.0, 200.0, 190.0, 215.0)), (DIG, SER), None),
    DetectorPhoto("F-000000000002", "teste", "2026-07-03", 2.0, "g",
                  ((100.0, 50.0, 200.0, 80.0), (110.0, 200.0, 190.0, 215.0)), (CYC, SER), None),
]
TYPES = {"F-000000000001": "digital", "F-000000000002": "ciclometrico"}


def _perfect(photo):
    return {"boxes": np.array(photo.boxes, np.float32), "scores": np.full(2, 0.9, np.float32),
            "labels": np.array(photo.labels)}


def test_perfect_predictions():
    out = evaluate(PHOTOS, [_perfect(p) for p in PHOTOS], 0.0122, 0.19, TYPES)
    assert out["selecao"] == pytest.approx(1.0)
    assert out["porta"]["perda"] == 0 and out["meta_cumprida"]
    assert out["regra_kvarh"]["acerto"] == pytest.approx(1.0) and out["fotos_regra_kvarh"] == 1
    assert out["validador2"]["diverge"] == 0


def test_photo_without_boxes_is_lost_at_the_gate():
    empty = {"boxes": np.zeros((0, 4), np.float32), "scores": np.zeros(0, np.float32),
             "labels": np.zeros(0, int)}
    out = evaluate(PHOTOS, [_perfect(PHOTOS[0]), empty], 0.0122, 0.19, TYPES)
    assert out["porta"]["fotos_com_campo_perdidas"] == 1
    assert out["porta"]["perda"] == pytest.approx(2 / 3)  # pesos 1 e 2
