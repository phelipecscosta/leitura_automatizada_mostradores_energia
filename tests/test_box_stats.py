"""Testes da geometria das caixas (training/box_stats.py)."""
import math

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
    assert list(frame["foto_largura"]) == [360, 360, 480]


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
