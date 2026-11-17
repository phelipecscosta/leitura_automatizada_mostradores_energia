"""Testes das âncoras escolhidas pelos dados (training/detector_anchors.py)."""
import numpy as np

from training import detector_anchors as da


def test_shape_iou_known_cases():
    iou = da.shape_iou(np.array([[10.0, 10.0], [10.0, 5.0]]), np.array([[10.0, 10.0]]))
    assert np.allclose(iou[:, 0], [1.0, 0.5])


def test_kmeans_recovers_two_separated_shapes_sorted_by_area():
    rng = np.random.default_rng(0)
    wide = np.column_stack([rng.normal(80, 2, 100), rng.normal(16, 0.5, 100)])
    small = np.column_stack([rng.normal(20, 1, 100), rng.normal(10, 0.5, 100)])
    centers = da.kmeans_shapes(np.vstack([wide, small]), 2)
    assert np.allclose(centers, [[20, 10], [80, 16]], atol=1.5)
    assert np.array_equal(centers, da.kmeans_shapes(np.vstack([wide, small]), 2))  # determinístico


def test_replicate_levels_repeats_the_same_shapes():
    levels = da.replicate_levels(np.array([[1.234, 2.0], [3.0, 4.0]]), 3)
    assert levels == [[(1.23, 2.0), (3.0, 4.0)]] * 3


def test_candidates_follow_the_protocol():
    names = {c.name: c for c in da.CANDIDATES}
    assert set(names) == {"S1", "F1", "F2"}
    assert (names["S1"].levels, names["S1"].per_level) == (6, 6)
    assert (names["F1"].levels, names["F1"].per_level) == (5, 15)
    assert names["F2"].profile == "GPU" and names["F2"].input_size == (600, 800)


def test_model_anchors_count_and_centers_inside_the_input():
    import torch
    cand = da.CANDIDATES[0]
    shapes = [[(80.0, 16.0)] * cand.per_level] * cand.levels
    model = cand.build(shapes)
    anchors, input_hw = da.model_anchors(model, cand.input_size)
    assert input_hw == (480, 360)
    with torch.no_grad():
        images, _ = model.transform([torch.zeros(3, 480, 360)])
        sizes = [f.shape[-2:] for f in model.backbone(images.tensors).values()]
    assert len(anchors) == sum(h * w for h, w in sizes) * cand.per_level
    centers = (anchors[:, :2] + anchors[:, 2:]) / 2
    assert float(centers[:, 0].max()) < 360 and float(centers[:, 1].max()) < 480
    assert float(centers.min()) > 0
