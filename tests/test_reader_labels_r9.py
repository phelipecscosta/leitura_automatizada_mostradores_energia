import pandas as pd
import pytest

from training.reader_labels_r9 import human_judged_keys, r9_rows


def _crops():
    return pd.DataFrame({"pseudonimo": ["A", "B"], "classe": ["display_ciclometrico"] * 2,
                         "x0": [1.0, 2.0], "y0": [1.0, 2.0], "x1": [5.0, 6.0], "y1": [3.0, 4.0],
                         "origem_caixa": ["final", "final"]})


def test_linhas_da_r9():
    judged = pd.DataFrame({"pseudonimo": ["A", "B"], "classe": ["display_ciclometrico"] * 2,
                           "status": ["humano", "ilegivel"], "texto": ["0012[34]", ""]})
    out = r9_rows(judged, _crops()).set_index("pseudonimo")
    assert out.loc["A", "status"] == "humano" and out.loc["A", "texto"] == "0012[34]"
    assert out.loc["A", "valor"] == ""                       # transição: sem valor único
    assert out.loc["B", "texto"] == "" and out.loc["B", "rotacao"] == ""
    assert set(out["origem"]) == {"humano_r9"}


def test_r9_recusa_status_desconhecido_e_repeticao():
    bad = pd.DataFrame({"pseudonimo": ["A"], "classe": ["display_ciclometrico"],
                        "status": ["cortado"], "texto": [""]})
    with pytest.raises(SystemExit):
        r9_rows(bad, _crops())
    twice = pd.DataFrame({"pseudonimo": ["A", "A"], "classe": ["display_ciclometrico"] * 2,
                          "status": ["humano", "humano"], "texto": ["1", "1"]})
    with pytest.raises(SystemExit):
        r9_rows(twice, _crops())


def test_julgados_sem_alvo():
    labels = pd.DataFrame({"pseudonimo": ["A", "B", "C", "D"],
                           "classe": ["display_ciclometrico"] * 4,
                           "status": ["humano", "ilegivel", "aceito", "ilegivel"],
                           "origem": ["humano_r9", "humano_r9", "haiku_csv", "humano_f2"]})
    keys = human_judged_keys(labels, ("aceito", "humano"))
    assert keys == {"B|display_ciclometrico", "D|display_ciclometrico"}
