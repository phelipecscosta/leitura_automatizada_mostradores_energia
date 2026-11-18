"""Testes da entrada, do artefato e da decisão do detector do produto (T5c.4)."""
import hashlib

import numpy as np
import pytest
import torch
from PIL import Image

from meter_reader import detector as det

SHAPES = [[(80.0, 20.0), (40.0, 40.0)]] * 6


def _artifact(tmp_path, **changes):
    data = {"formato": det.ARTIFACT_FORMAT, "classes": list(det.DETECTOR_CLASSES),
            "formas": SHAPES, "pesos": det.build_ssd(SHAPES).state_dict(),
            "limiar_porta": 0.0122, "corte_campos": 0.19, "info": {"candidato": "teste"}}
    data.update(changes)
    path = tmp_path / "detector.pt"
    torch.save(data, path)
    return path


def _photo(tmp_path):
    image = np.random.default_rng(0).integers(0, 255, (480, 360, 3), dtype=np.uint8)
    path = tmp_path / "foto.jpg"
    Image.fromarray(image).save(path)
    return path


def test_input_is_enhanced_in_unit_range(tmp_path):
    x = det.load_input(_photo(tmp_path))
    assert tuple(x.shape) == (3, 480, 360) and 0.0 <= float(x.min()) and float(x.max()) <= 1.0


def test_photo_score():
    assert det.photo_score([0.2, 0.7]) == pytest.approx(0.7) and det.photo_score([]) == 0.0


def test_load_checks_sha_format_and_classes(tmp_path):
    path = _artifact(tmp_path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    model = det.load_detector(path, sha)
    assert model.gate_threshold == pytest.approx(0.0122) and model.field_cut == pytest.approx(0.19)
    assert not model.model.training
    with pytest.raises(det.DetectorArtifactError):
        det.load_detector(path, "0" * 64)
    with pytest.raises(det.DetectorArtifactError):
        det.load_detector(_artifact(tmp_path, formato=99))
    with pytest.raises(det.DetectorArtifactError):
        det.load_detector(_artifact(tmp_path, classes=list(reversed(det.DETECTOR_CLASSES))))


def test_decide_gate_family_and_fields(tmp_path):
    model = det.load_detector(_artifact(tmp_path))
    empty = model.decide(np.zeros((0, 4)), [], [])
    assert not empty.passes_gate and empty.fields is None and empty.family is None
    low = model.decide([[0, 0, 10, 10]], [0.01], [2])
    assert not low.passes_gate  # abaixo do limiar da porta
    ok = model.decide([[100, 50, 200, 80]], [0.9], [2])
    assert ok.passes_gate and ok.family == "ciclometrico"
    assert np.array_equal(ok.fields.consumption, [100, 50, 200, 80])


def test_predict_runs_on_a_photo(tmp_path):
    out = det.load_detector(_artifact(tmp_path)).predict(_photo(tmp_path))
    assert out.boxes.shape[1] == 4 and len(out.scores) == len(out.labels)
