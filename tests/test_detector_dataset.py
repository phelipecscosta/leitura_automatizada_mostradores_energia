"""Testes do conjunto de dados do detector (training/detector_dataset.py)."""
import numpy as np
import pytest
import torch
from PIL import Image

from training import detector_dataset as dd
from training.box_labels import Box


def _rows():
    return [{"pseudonimo": f"F-{i:012x}", "conjunto": "desenvolvimento" if i < 6 else "teste",
             "lote": ("2026-05-20", "2026-05-21", "2026-05-22")[i % 3] if i < 6 else "2026-07-03",
             "peso": "2.5"} for i in range(8)]


def _boxes():
    return [Box("F-000000000000", "numero_serie", 10, 20, 90, 37),
            Box("F-000000000000", "display_digital", 50, 100, 200, 160),
            Box("F-000000000001", "display_digital", 5, 5, 8, 6)]   # minúscula: 3 × 1 px


def _photos(paths=None):
    rows = _rows()
    return dd.build_photos(rows, _boxes(), {r["pseudonimo"]: "dig1_quadrado_branco" for r in rows},
                           paths)


def test_build_photos_targets_labels_and_weights():
    photos = _photos()
    first = photos[0]
    assert first.labels == (4, 1)  # numero_serie e display_digital, na ordem de DETECTOR_CLASSES
    target = first.target()
    assert target["boxes"].shape == (2, 4) and target["labels"].tolist() == [4, 1]
    assert first.weight == 2.5 and first.has_field
    empty = photos[2].target()          # foto sem caixas: o fundo inteiro
    assert empty["boxes"].shape == (0, 4) and empty["labels"].numel() == 0
    assert not photos[1].has_field      # só uma caixa minúscula (D3 a ii)


def test_build_photos_rejects_stray_boxes_and_missing_groups():
    rows = _rows()
    with pytest.raises(ValueError):
        dd.build_photos(rows, _boxes() + [Box("F-0000000000ff", "numero_serie", 0, 0, 9, 9)],
                        {r["pseudonimo"]: "g" for r in rows})
    with pytest.raises(ValueError):
        dd.build_photos(rows, _boxes(), {})


def test_folds_use_only_development_and_hold_out_one_lot():
    photos = _photos()
    train, held = dd.fold_indices(photos, "2026-05-20")
    assert set(held) == {0, 3} and set(train) == {1, 2, 4, 5}
    with pytest.raises(ValueError):
        dd.fold_indices(photos, "2026-07-03")   # o teste nunca é dobra


def test_internal_split_is_fixed_and_ignores_input_order():
    photos = [dd.DetectorPhoto(f"F-{i:012x}", "desenvolvimento", "2026-05-20", 1.0, "g", (), ())
              for i in range(100)]
    train = np.arange(100)
    fit, internal = dd.internal_split(photos, train)
    assert len(internal) == 10 and len(fit) == 90 and not set(fit) & set(internal)
    assert np.array_equal(internal, dd.internal_split(photos, train[::-1])[1])


def test_augment_moves_boxes_with_the_image_and_is_deterministic():
    image = torch.zeros(3, 480, 360)
    image[:, 100:120, 50:150] = 1.0   # um "campo" branco
    target = {"boxes": torch.tensor([[50.0, 100.0, 150.0, 120.0]]), "labels": torch.tensor([1])}
    out_image, out = dd.augment(image, target, seed=7)
    again_image, again = dd.augment(image, target, seed=7)
    assert torch.equal(out_image, again_image) and torch.equal(out["boxes"], again["boxes"])
    assert out_image.shape == image.shape and out["labels"].tolist() == [1]
    # A caixa transformada continua sobre o campo branco
    x0, y0, x1, y1 = out["boxes"][0].round().int().tolist()
    inside = out_image[0, y0 + 2:y1 - 2, x0 + 2:x1 - 2]
    assert inside.mean() > 0.8
    assert not torch.equal(dd.augment(image, target, seed=8)[1]["boxes"], out["boxes"])


def test_augment_keeps_empty_targets_empty():
    target = {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.int64)}
    _, out = dd.augment(torch.rand(3, 480, 360), target, seed=1)
    assert out["boxes"].shape == (0, 4)


def test_dataset_reads_enhanced_photo_and_changes_view_by_epoch(tmp_path):
    rng = np.random.default_rng(0)
    paths = {}
    for row in _rows():
        path = tmp_path / f"{row['pseudonimo']}.jpg"
        Image.fromarray(rng.integers(0, 256, (480, 360, 3), dtype=np.uint8)).save(path)
        paths[row["pseudonimo"]] = path
    photos = _photos(paths)
    plain = dd.DetectorDataset(photos, train=False)
    image, target = plain[0]
    assert image.shape == (3, 480, 360) and 0.0 <= float(image.min()) and float(image.max()) <= 1.0
    assert torch.equal(target["boxes"], photos[0].target()["boxes"])
    train = dd.DetectorDataset(photos, train=True)
    first = train[0][1]["boxes"]
    train.set_epoch(1)
    assert not torch.equal(first, train[0][1]["boxes"])
    images, targets = dd.collate([plain[0], plain[1]])
    assert len(images) == 2 and len(targets) == 2
