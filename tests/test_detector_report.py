"""Testes do relatório do detector (training/detector_report.py)."""
import numpy as np
import pytest

from training import detector_report as dr
from training.detector_dataset import DetectorPhoto


def _photo(i, boxes, labels, weight=1.0):
    return DetectorPhoto(f"F-{i:012x}", "desenvolvimento", "2026-05-20", weight, "g",
                         tuple(map(tuple, boxes)), tuple(labels))


def _pred(boxes, scores, labels):
    return {"boxes": np.array(boxes, float).reshape(-1, 4), "scores": np.array(scores, float),
            "labels": np.array(labels, int)}


def test_align_requires_each_photo_once():
    photos = [_photo(1, [], []), _photo(2, [], [])]
    preds = [_pred([], [], []), _pred([[0, 0, 1, 1]], [0.5], [4])]
    out = dr.align([photos[1].pseudonym, photos[0].pseudonym], preds, photos)
    assert len(out[0]["scores"]) == 1 and len(out[1]["scores"]) == 0
    with pytest.raises(ValueError):
        dr.align([photos[0].pseudonym], preds[:1], photos)


def test_gate_at_counts_weighted_loss_and_coverage():
    photos = [_photo(1, [[0, 0, 50, 20]], [1], 2.0), _photo(2, [[0, 0, 50, 20]], [1], 1.0),
              _photo(3, [], [], 3.0)]
    preds = [_pred([[0, 0, 50, 20]], [0.9], [1]), _pred([[0, 0, 50, 20]], [0.2], [1]),
             _pred([], [], [])]
    g = dr.gate_at(photos, preds, 0.5)
    assert g["perda"] == pytest.approx(1 / 3) and g["cobertura"] == 1.0
    assert g["fotos_com_campo_perdidas"] == 1


def test_kvarh_confusion_detects_a_consumption_box_on_the_kvarh_display():
    photos = [_photo(1, [[0, 50, 60, 70]], [3])]
    preds = [_pred([[0, 50, 60, 70]], [0.8], [2])]  # caixa de consumo sobre o kvarh
    assert dr.kvarh_confusion(photos, preds, 0.5)["fracao_confundida"] == 1.0
    assert dr.kvarh_confusion(photos, preds, 0.9)["fracao_confundida"] == 0.0


def test_validator2_fractions():
    photos = [_photo(i, [[0, 0, 50, 20]], [1]) for i in range(4)]
    preds = [_pred([[0, 0, 50, 20]], [0.9], [1]),                         # digital, concorda
             _pred([[0, 0, 50, 20]], [0.9], [2]),                         # ciclométrico: diverge
             _pred([[0, 0, 50, 20], [0, 30, 50, 50]], [0.9, 0.6], [1, 2]),  # duas famílias
             _pred([[0, 0, 50, 20]], [0.9], [4])]                         # só número de série
    types = {p.pseudonym: "digital" for p in photos}
    v = dr.validator2(photos, preds, 0.5, types)
    assert v["diverge"] == 0.25 and v["duas_familias"] == 0.25
    assert v["duas_familias_diverge"] == 0.0 and v["sem_familia"] == 0.25


def _fake(name, selection, viable=True):
    return {"candidato": name, "viavel": viable, "selecao_media": selection,
            "latencia_p95_ms": dr.LATENCY_P95_MS[name]}


def test_choose_takes_the_best_when_the_simplest_is_best(monkeypatch):
    summaries = {"S1": _fake("S1", 0.8), "F1": _fake("F1", 0.7), "F2": _fake("F2", 0.75)}
    monkeypatch.setattr(dr, "bootstrap_seed_mean_diff", lambda *a, **k: np.full(10, -0.05))
    d = dr.choose(summaries, {n: [] for n in summaries}, [])
    assert d["cpu"] == "S1" and d["gpu"] is None


def test_choose_uses_non_inferiority_and_the_gpu_margin(monkeypatch):
    summaries = {"S1": _fake("S1", 0.70), "F1": _fake("F1", 0.73), "F2": _fake("F2", 0.90)}
    calls = []

    def fake(photos, a, b, **k):
        calls.append(1)
        return np.full(10, -0.04) if len(calls) == 1 else np.full(10, 0.12)
    monkeypatch.setattr(dr, "bootstrap_seed_mean_diff", fake)
    d = dr.choose(summaries, {n: [] for n in summaries}, [])
    assert d["cpu"] == "S1" and d["gpu"] == "F2"   # -0,04 > -0,05; +0,12 >= 0,10
    calls.clear()
    monkeypatch.setattr(dr, "bootstrap_seed_mean_diff",
                        lambda *a, **k: np.full(10, -0.06))
    assert dr.choose(summaries, {n: [] for n in summaries}, [])["cpu"] == "F1"


def test_choose_skips_non_viable_candidates():
    summaries = {"S1": _fake("S1", 0.9, viable=False), "F1": _fake("F1", 0.7),
                 "F2": _fake("F2", 0.5, viable=False)}
    d = dr.choose(summaries, {n: [] for n in summaries}, [])
    assert d["cpu"] == "F1" and d["gpu"] is None
