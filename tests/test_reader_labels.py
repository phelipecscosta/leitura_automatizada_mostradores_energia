from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from training.reader_labels import CROP_COLUMNS, final_predictions, merge_crops


def _crops(codes):
    return pd.DataFrame([{"pseudonimo": c, "classe": "numero_serie",
                          "x0": 0.0, "y0": 0.0, "x1": 10.0, "y1": 5.0} for c in codes])


def test_merge_crops_marks_origin_and_refuses_overlap():
    out = merge_crops(_crops(["F-2"]), _crops(["F-1"]))
    assert tuple(out.columns) == CROP_COLUMNS
    assert list(out["pseudonimo"]) == ["F-1", "F-2"]
    assert list(out["origem_caixa"]) == ["final", "fora_da_dobra"]
    with pytest.raises(ValueError):
        merge_crops(_crops(["F-1"]), _crops(["F-1"]))


def test_final_predictions_keeps_detector_boxes():
    result = SimpleNamespace(boxes=np.zeros((1, 4)), scores=np.array([0.9]), labels=np.array([4]))
    detector = SimpleNamespace(predict=lambda path: result)
    preds = final_predictions(["a.jpg", "b.jpg"], detector)
    assert len(preds) == 2 and preds[0]["labels"].tolist() == [4]

from training.reader_labels import custom_id, parse_custom_id, split_parts


def test_custom_id_round_trip_and_format():
    cid = custom_id("F-0123456789ab", "display_ciclometrico")
    assert cid == "F-0123456789ab__cic"
    assert parse_custom_id(cid) == ("F-0123456789ab", "display_ciclometrico")


def test_split_parts_respects_both_limits():
    assert split_parts([40, 40, 40, 40], max_bytes=100, max_requests=10) == [0, 2]
    assert split_parts([1, 1, 1, 1, 1], max_bytes=100, max_requests=2) == [0, 2, 4]

from training.reader_labels import numeric_value


def test_numeric_value_strips_zeros_and_refuses_uncertain():
    assert numeric_value("0012345") == "12345"
    assert numeric_value("0000") == "0"
    assert numeric_value("01?45") == ""
    assert numeric_value("0124[56]") == ""
    assert numeric_value("") == ""

from training.reader_labels import second_reading_rows


def test_second_reading_rows_follows_s_b_and_takes_all_cyclometric():
    rows = []
    for i in range(6):   # tipo 1: outra função
        rows.append({"pseudonimo": f"F-a{i}", "classe": "display_digital", "status": "fila_f2",
                     "funcao": "103"})
    for i in range(6):   # tipo 2: função 03 discordante
        rows.append({"pseudonimo": f"F-b{i}", "classe": "display_digital", "status": "fila_f2",
                     "funcao": "03"})
    rows += [{"pseudonimo": "F-c1", "classe": "display_ciclometrico", "status": "discorda",
              "funcao": ""},
             {"pseudonimo": "F-c2", "classe": "display_ciclometrico", "status": "parcial",
              "funcao": ""},
             {"pseudonimo": "F-c3", "classe": "display_ciclometrico", "status": "aceito",
              "funcao": ""}]
    labels = pd.DataFrame(rows).assign(x0=0.0, y0=0.0, x1=1.0, y1=1.0, origem_caixa="final")
    out = second_reading_rows(labels, {"f2_tipo1": 4, "f2_tipo2": 2})
    assert out["grupo"].value_counts().to_dict() == {
        "f2_tipo1": 4, "f2_tipo2": 2, "cic_discorda": 1, "cic_parcial": 1}
    assert "F-c3" not in set(out["pseudonimo"])
