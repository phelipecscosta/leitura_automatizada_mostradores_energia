import numpy as np
import pandas as pd

from meter_reader.reader_text import TEXT_ALPHABET, encode
from training.reader_pseudo import as_items, candidates, decide, fold_thresholds


def _peaked(text, T=12):
    lp = np.full((T, len(TEXT_ALPHABET) + 1), -20.0)
    lp[:, 0] = 0.0
    for j, k in enumerate(encode(text, TEXT_ALPHABET)):
        lp[2 * j + 1] = -20.0
        lp[2 * j + 1, k] = 0.0
    return lp


def test_decisao_do_pseudo_rotulo():
    assert decide(_peaked("0045"), "45", "display_digital", -1.0) == "0045"   # mantém os zeros
    assert decide(_peaked("0046"), "45", "display_digital", -1.0) is None     # discorda do CSV
    assert decide(_peaked("AB12"), "AB12", "numero_serie", -1.0) == "AB12"
    blurred = np.log(np.full((12, len(TEXT_ALPHABET) + 1), 1 / 37))
    assert decide(blurred, "45", "display_digital", -1.0) is None


def test_candidatos_da_dobra():
    crops = pd.DataFrame({"pseudonimo": ["F-a", "F-b", "F-c", "F-d", "F-e"],
                          "classe": ["display_digital", "numero_serie", "display_digital",
                                     "display_kvarh", "display_digital"]})
    lots = {"F-a": "A", "F-b": "B", "F-c": "A", "F-d": "A", "F-e": "A"}
    out = candidates(crops, lots, "B", targets={"F-c|display_digital"},
                     two_lines={"F-e|display_digital"})
    assert out["pseudonimo"].tolist() == ["F-a"]   # sem o lote avaliado, alvos, kvarh e duas


def test_limiar_so_dos_lotes_de_treino():
    rows = []
    for lot in ("A", "B"):
        for cls in ("display_digital", "display_ciclometrico", "numero_serie"):
            for i in range(50):
                rows.append({"semente": 0, "lote": lot, "classe": cls,
                             "tem_imp_confundivel": True,
                             "imp_confundivel_lido": lot == "B" and i < 10,
                             "logp_imp_confundivel": -0.1 if lot == "B" else -50.0})
    t = fold_thresholds(pd.DataFrame(rows), "B")    # avalia B: só A conta
    assert all(v == -np.inf for v in t.values())


def test_formato_dos_itens():
    items = as_items(pd.DataFrame({"pseudonimo": ["F-a"], "classe": ["display_digital"],
                                   "lote": ["A"], "texto": ["0045"]}))
    r = items.iloc[0]
    assert (r["usa_texto"], r["usa_funcao"], r["usa_duas"], r["origem"]) == (
        True, False, False, "pseudo")
