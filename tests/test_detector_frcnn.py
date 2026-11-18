"""Testes da montagem do Faster R-CNN dos candidatos F1 e F2 (training/detector_frcnn.py)."""
import pytest
import torch

from meter_reader import detector as det
from training import detector_frcnn as fr

SSD_SHAPES = [[(8.0 * (i + 1), 4.0 * (i + 1)), (6.0 * (i + 1), 6.0 * (i + 1))] for i in range(6)]
FRCNN_SHAPES = [[(16.0 * (i + 1), 8.0 * (i + 1)), (12.0 * (i + 1), 12.0 * (i + 1))] for i in range(5)]


def test_frcnn_trains_and_predicts_on_a_portrait_photo():
    torch.manual_seed(0)
    model = fr.build_frcnn(FRCNN_SHAPES, 360, 480)
    images = [torch.rand(3, 480, 360) for _ in range(2)]
    target = {"boxes": torch.tensor([[50.0, 60.0, 150.0, 90.0]]), "labels": torch.tensor([4])}
    model.train()
    losses = model(images, [target, target])
    assert all(torch.isfinite(v) for v in losses.values())
    model.eval()
    with torch.no_grad():
        out = model(images[:1])[0]
    assert set(out) == {"boxes", "labels", "scores"}


def test_frcnn_requires_one_shape_list_per_level_with_the_same_count():
    with pytest.raises(ValueError):   # contagens diferentes entre níveis
        fr.build_frcnn([[(1.0, 1.0)]] * 4 + [[(1.0, 1.0), (2.0, 2.0)]], 360, 480)
    with pytest.raises(ValueError):   # 3 níveis, mas a N8 tem 5
        fr.build_frcnn([[(1.0, 1.0)]] * 3, 360, 480)


def test_frcnn_n8_starts_at_stride_8_on_a_native_photo():
    model = fr.build_frcnn(FRCNN_SHAPES, 360, 480).eval()
    with torch.no_grad():
        images, _ = model.transform([torch.rand(3, 480, 360)])
        features = model.backbone(images.tensors)
    height = images.tensors.shape[-2]
    strides = [height // f.shape[-2] for f in features.values()]
    assert strides[:4] == [8, 16, 32, 32]
    assert list(features.values())[-1].shape[-2] == 8  # nível extra: 15 -> 8 linhas


def test_low_latency_rpn_reaches_the_model():
    model = fr.build_frcnn(FRCNN_SHAPES, 360, 480, **fr.LOW_LATENCY_RPN)
    assert model.rpn._post_nms_top_n["testing"] == 150
    assert model.rpn._pre_nms_top_n["testing"] == 150


def test_both_architectures_use_the_same_score_threshold():
    ssd = det.build_ssd(SSD_SHAPES)
    frcnn = fr.build_frcnn(FRCNN_SHAPES, 360, 480)
    assert ssd.score_thresh == det.SCORE_THRESH == frcnn.roi_heads.score_thresh == 0.001
