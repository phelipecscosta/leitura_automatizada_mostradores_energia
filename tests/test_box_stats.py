"""Testes da geometria das caixas (training/box_stats.py)."""
import math

import numpy as np
import pandas as pd
import torch

from training import box_stats as bs
from training.box_labels import Box

SIZES = {"F-000000000001": (360, 480), "F-000000000002": (480, 360),
         "F-000000000003": (360, 480)}  # a terceira foto não tem caixas


def _boxes():
    return [Box("F-000000000001", "numero_serie", 10, 20, 70, 32),     # 60 x 12
            Box("F-000000000001", "display_digital", 0, 0, 100, 50),   # 100 x 50
            Box("F-000000000002", "numero_serie", 5, 5, 45, 15)]       # 40 x 10


def test_box_frame_measures_in_original_pixels():
    frame = bs.box_frame(_boxes(), SIZES)
    serie = frame[frame["classe"] == "numero_serie"]
    assert list(serie["largura"]) == [60, 40] and list(serie["altura"]) == [12, 10]
    assert list(serie["proporcao"]) == [5.0, 4.0]
    assert list(serie["x0"]) == [10, 5] and list(frame["foto_largura"]) == [360, 360, 480]


def test_summary_has_every_class_and_the_quantiles():
    summary = bs.geometry_summary(bs.box_frame(_boxes(), SIZES))
    row = summary.loc[("numero_serie", "largura")]
    assert (row["caixas"], row["min"], row["p50"], row["max"]) == (2, 40, 50, 60)
    empty = summary.loc[("display_kvarh", "altura")]  # classe sem caixas continua na tabela
    assert empty["caixas"] == 0 and math.isnan(empty["p50"])


def test_boxes_per_photo_counts_photos_without_boxes():
    counts = bs.boxes_per_photo(SIZES.keys(), bs.box_frame(_boxes(), SIZES))
    assert counts.to_dict() == {"0": 1, "1": 1, "2": 1, "3 ou mais": 0}


def test_photo_size_counts_by_orientation():
    assert bs.photo_size_counts(SIZES).to_dict() == {"360 x 480": 2, "480 x 360": 1}


def test_tiny_boxes_counts_any_short_side():
    boxes = _boxes() + [Box("F-000000000002", "display_digital", 0, 0, 50, 2.5)]  # 2,5 px de altura
    tiny = bs.tiny_boxes(bs.box_frame(boxes, SIZES))
    assert tiny["display_digital"] == 1 and tiny["numero_serie"] == 0


def test_to_input_scales_x_and_y_separately():
    frame = bs.box_frame([Box("F-000000000001", "display_digital", 0, 0, 360, 480)], SIZES)
    assert bs.to_input(frame, (320, 320)).tolist() == [[0.0, 0.0, 320.0, 320.0]]


def test_best_iou_known_cases_and_chunking():
    anchors = torch.tensor([[0, 0, 10, 10], [100, 100, 110, 110]], dtype=torch.float32)
    boxes = torch.tensor([[0, 0, 10, 10], [0, 0, 5, 10], [50, 50, 60, 60]], dtype=torch.float32)
    best = bs.best_iou(boxes, anchors)
    assert np.allclose(best, [1.0, 0.5, 0.0])
    assert np.allclose(bs.best_iou(boxes, anchors, chunk=1), best)


def test_anchor_summary_fraction_and_median():
    frame = pd.DataFrame({"classe": ["numero_serie", "numero_serie"]})
    summary = bs.anchor_summary(frame, np.array([0.4, 0.6]))
    assert summary.loc["numero_serie", "frac_iou_0_5"] == 0.5
    assert summary.loc["numero_serie", "iou_p50"] == 0.5
    assert summary.loc["display_kvarh", "caixas"] == 0


def test_default_ssdlite_input_is_a_320_square():
    anchors, input_hw = bs.default_anchors("ssdlite320")
    assert input_hw == (320, 320) and anchors.shape[1] == 4
    # As âncoras não são cortadas na borda (as maiores passam da entrada); os centros ficam dentro
    centers = (anchors[:, :2] + anchors[:, 2:]) / 2
    assert float(centers.min()) >= 0.0 and float(centers.max()) <= 320.0
