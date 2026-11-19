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
