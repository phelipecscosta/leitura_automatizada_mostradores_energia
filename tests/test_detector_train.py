"""Testes do treino do detector (training/detector_train.py e run_detector.py)."""
import json

import numpy as np
import pytest
from PIL import Image

from training import detector_train as dt
from training import run_detector as rd
from training.detector_anchors import CANDIDATES
from training.detector_dataset import DetectorPhoto

SHAPES = [[(80.0, 20.0), (40.0, 40.0)]] * 6  # S1: 6 níveis
CFG = dt.DetectorTrainConfig(max_epochs=2, batch=2, num_workers=0, eval_workers=0, amp=False)


def _photos(tmp_path, n):
    rng = np.random.default_rng(0)
    photos = []
    for i in range(n):
        image = rng.integers(0, 60, (480, 360, 3), dtype=np.uint8)
        image[100:130, 50:150] = 255          # um "campo" claro, sempre no mesmo lugar
        path = tmp_path / f"f{i}.jpg"
        Image.fromarray(image).save(path)
        photos.append(DetectorPhoto(f"F-{i:012x}", "desenvolvimento", "2026-05-20", 1.0, "g",
                                    ((50.0, 100.0, 150.0, 130.0),), (1,), path))
    return photos


def test_recipes_follow_the_linear_rule():
    assert dt.RECIPES["S1"][0] == pytest.approx(0.00625)
    assert dt.RECIPES["F1"][0] == pytest.approx(0.01) and dt.RECIPES["F1"][1] == 1e-4


def test_train_one_returns_the_best_epoch_and_history(tmp_path):
    photos = _photos(tmp_path, 5)
    cand = CANDIDATES[0]
    model, best_epoch, history = dt.train_one(cand, SHAPES, photos[:4], photos[4:], 0, "cpu",
                                              CFG, log=lambda m: None, pretrained=False)
    assert best_epoch in (1, 2) and [h[0] for h in history] == [1, 2]
    preds = dt.predict(model, dt.make_loader(dt.DetectorDataset(photos[4:], False), CFG, "cpu",
                                             False), "cpu")
    assert len(preds) == 1 and set(preds[0]) == {"boxes", "scores", "labels"}


def test_drop_last_skips_the_incomplete_batch(tmp_path):
    photos = _photos(tmp_path, 3)
    loader = dt.make_loader(dt.DetectorDataset(photos, True), CFG, "cpu", True)
    assert len(loader) == 1  # 3 fotos, lote de 2: só um lote completo


def test_predictions_round_trip_through_the_part_file(tmp_path):
    preds = [{"boxes": np.array([[1, 2, 3, 4]], np.float32), "scores": np.array([0.5], np.float32),
              "labels": np.array([4])},
             {"boxes": np.zeros((0, 4), np.float32), "scores": np.zeros(0, np.float32),
              "labels": np.zeros(0, int)}]
    path = tmp_path / "parte.npz"
    rd.save_part(path, ["F-000000000001", "F-000000000002"], preds, {"melhor_epoca": 3})
    with np.load(path) as data:
        codes, back = dt.unflatten_predictions(data)
        meta = json.loads(str(data["meta"]))
    assert codes == ["F-000000000001", "F-000000000002"] and meta["melhor_epoca"] == 3
    assert np.array_equal(back[0]["boxes"], preds[0]["boxes"]) and len(back[1]["scores"]) == 0
    assert not path.with_name(path.name + ".tmp").exists()
