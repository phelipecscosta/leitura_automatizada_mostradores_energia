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


def test_save_part_writes_weights_next_to_the_predictions(tmp_path):
    import torch
    path = tmp_path / "parte.npz"
    rd.save_part(path, [], [], {}, {"w": torch.ones(2)})
    assert torch.equal(torch.load(path.with_suffix(".pt"), weights_only=True)["w"], torch.ones(2))
    assert not list(tmp_path.glob("*.tmp"))

def test_setup_with_coco_prefix_changes_only_the_prefix(monkeypatch):
    """Ablação pareada: com a mesma semente, só o prefixo da espinha muda."""
    import torch
    from training import detector_coco as coco
    cand = CANDIDATES[0]
    donor = cand.build(SHAPES).state_dict()
    source = {k: torch.full_like(v, 7.0) if v.is_floating_point() else v for k, v in donor.items()}
    monkeypatch.setattr(coco, "coco_state", lambda: source)  # nenhum download
    plain = dt.setup(cand, SHAPES, CFG, 4, 0, "cpu", pretrained=False)["model"].state_dict()
    ablated = dt.setup(cand, SHAPES, CFG, 4, 0, "cpu", pretrained=False,
                       coco_prefix=True)["model"].state_dict()
    keys = set(coco.prefix_keys(plain, source))
    assert keys and all(torch.equal(ablated[k], source[k]) for k in keys)
    assert all(torch.equal(ablated[k], plain[k]) for k in plain if k not in keys)


def test_variant_name_only_for_s1():
    assert rd.variant_name("S1", False) == "S1" and rd.variant_name("S1", True) == "S1_coco"
    assert rd.variant_name("F1", False) == "F1"
    with pytest.raises(SystemExit):
        rd.variant_name("F1", True)

def test_train_fixed_matches_train_one_at_the_same_epoch(tmp_path):
    """Parado na época escolhida, o treino final dá os mesmos pesos do train_one."""
    import torch
    photos = _photos(tmp_path, 5)
    cand = CANDIDATES[0]
    model, best_epoch, _ = dt.train_one(cand, SHAPES, photos[:4], photos[4:], 0, "cpu", CFG,
                                        log=lambda m: None, pretrained=False)
    fixed = dt.train_fixed(cand, SHAPES, photos[:4], best_epoch, 0, "cpu", CFG,
                           log=lambda m: None, pretrained=False)
    a, b = model.state_dict(), fixed.state_dict()
    assert set(a) == set(b)
    assert all(torch.allclose(a[k].float(), b[k].float(), atol=1e-6) for k in a)
    assert not fixed.training


def test_train_fixed_rejects_stop_outside_the_schedule(tmp_path):
    photos = _photos(tmp_path, 4)
    for stop in (0, CFG.max_epochs + 1):
        with pytest.raises(ValueError):
            dt.train_fixed(CANDIDATES[0], SHAPES, photos, stop, 0, "cpu", CFG,
                           log=lambda m: None, pretrained=False)


def test_median_stop_reads_the_nine_parts(tmp_path):
    epochs = [5, 9, 7, 3, 8, 6, 4, 10, 2]
    for i, epoch in enumerate(epochs):
        rd.save_part(rd.part_path(tmp_path, "S1_coco", f"lote{i // 3}", i % 3), [], [],
                     {"melhor_epoca": epoch})
    stop, read = rd.median_stop(tmp_path, "S1_coco")
    assert stop == 6 and sorted(read) == sorted(epochs)


def test_median_stop_requires_nine_parts(tmp_path):
    rd.save_part(rd.part_path(tmp_path, "S1", "lote0", 0), [], [], {"melhor_epoca": 3})
    with pytest.raises(SystemExit):
        rd.median_stop(tmp_path, "S1")
