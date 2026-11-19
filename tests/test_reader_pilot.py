from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from training.reader_pilot import PILOT_COLUMNS, candidates, draw


def _cands(n_dig=10, n_cyc=10, n_ser=10):
    rows = []
    for cls, n in (("display_digital", n_dig), ("display_ciclometrico", n_cyc),
                   ("numero_serie", n_ser)):
        rows += [{"pseudonimo": f"F-{cls[:3]}{i:03d}", "classe": cls,
                  "x0": 0.0, "y0": 0.0, "x1": 10.0, "y1": 5.0} for i in range(n)]
    return pd.DataFrame(rows)


def test_candidates_take_consumption_and_serial():
    photo = SimpleNamespace(pseudonym="F-000000000001")
    pred = {"boxes": np.array([[0, 0, 100, 20], [0, 50, 60, 60]], float),
            "scores": np.array([0.9, 0.8]), "labels": np.array([1, 4])}
    out = candidates([photo], [pred], cut=0.19)
    assert sorted(out["classe"]) == ["display_digital", "numero_serie"]


def test_draw_is_fixed_and_follows_counts():
    counts = {"display_digital": 4, "display_ciclometrico": 3, "numero_serie": 2}
    a = draw(_cands(), counts, stability_n=3)
    b = draw(_cands().sample(frac=1, random_state=1), counts, stability_n=3)  # outra ordem de entrada
    assert a.equals(b)
    assert a["classe"].value_counts().to_dict() == counts
    assert int(a["estabilidade"].sum()) == 3
    assert list(a["ordem"]) == list(range(1, 10))
    assert tuple(a.columns) == PILOT_COLUMNS


def test_draw_refuses_short_pool():
    with pytest.raises(ValueError):
        draw(_cands(n_ser=1), {"numero_serie": 2}, stability_n=0)
