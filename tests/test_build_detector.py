"""Testes do empacotamento do artefato do detector (T5c.4)."""
import hashlib

import pytest

from meter_reader import detector as det
from training import build_detector as bd
from training.run_final import save_final

SHAPES = [[(80.0, 20.0), (40.0, 40.0)]] * 6
META = {"candidato": "S1_coco", "epoca_parada": 20, "commit_produto": "abc1234",
        "sha_caixas": "c" * 64, "sha_formas": "f" * 64}


def _final(tmp_path):
    path = tmp_path / "S1_coco_final_s0.pt"
    save_final(path, det.build_ssd(SHAPES), META)
    return path


def test_artifact_loads_in_the_product(tmp_path):
    data = bd.build_artifact(_final(tmp_path), SHAPES, 0.0122, 0.19, {"x": "y"})
    path = tmp_path / "detector.pt"
    sha = bd.save_artifact(path, data)
    assert sha == hashlib.sha256(path.read_bytes()).hexdigest()
    model = det.load_detector(path, sha)
    assert model.gate_threshold == pytest.approx(0.0122) and model.field_cut == pytest.approx(0.19)
    assert model.info["epoca_parada"] == 20 and model.info["x"] == "y"


def test_never_overwrites(tmp_path):
    data = bd.build_artifact(_final(tmp_path), SHAPES, 0.0122, 0.19, {})
    bd.save_artifact(tmp_path / "detector.pt", data)
    with pytest.raises(FileExistsError):
        bd.save_artifact(tmp_path / "detector.pt", data)


def test_rejects_altered_weights_and_bad_thresholds(tmp_path):
    final = _final(tmp_path)
    with pytest.raises(ValueError):
        bd.build_artifact(final, SHAPES, 0.3, 0.19, {})  # limiar acima do corte
    final.write_bytes(final.read_bytes() + b"0")       # pesos alterados
    with pytest.raises(SystemExit):
        bd.build_artifact(final, SHAPES, 0.0122, 0.19, {})
        