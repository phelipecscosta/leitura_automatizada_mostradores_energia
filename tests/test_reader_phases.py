import numpy as np
import pandas as pd

from training.reader_phases import make_items_for
from training.reader_rule import bootstrap_mc
from training.reader_torch import BalancedBatchSampler


def _items():
    rows = []
    for lot in ("A", "B"):
        for cls, origem in (("display_digital", "haiku_csv"),
                            ("display_ciclometrico", "sonnet_csv"),
                            ("numero_serie", "haiku_csv")):
            rows.append({"pseudonimo": f"F-{lot}{cls[:3]}", "classe": cls, "lote": lot,
                         "recorte": "inteiro", "texto": "12", "funcao": "", "duas_leituras": False,
                         "usa_texto": True, "usa_funcao": False, "usa_duas": False,
                         "origem": origem})
    return pd.DataFrame(rows)


def test_itens_por_dobra():
    items = _items()
    pseudo = {"A": pd.DataFrame({"pseudonimo": ["F-Bx"], "classe": ["numero_serie"],
                                 "lote": ["B"], "texto": ["99"]}),
              "B": pd.DataFrame(columns=["pseudonimo", "classe", "lote", "texto"])}
    fold_a = make_items_for(items, pseudo)("A")
    assert (fold_a["origem"] == "pseudo").sum() == 1
    no_sonnet = make_items_for(items, None, drop_sonnet=True)("A")
    # o sonnet_csv sai do lote de treino (B) e fica no avaliado (A)
    kept = no_sonnet[no_sonnet["origem"] == "sonnet_csv"]
    assert kept["lote"].tolist() == ["A"]
    only = make_items_for(items, None, only_class="display_ciclometrico")("A")
    assert set(only["classe"]) == {"display_ciclometrico"}


def test_amostrador_com_uma_classe_e_passos():
    items = _items()
    cyc = items[items["classe"] == "display_ciclometrico"]
    sampler = BalancedBatchSampler(cyc, seed=0, text_batch=8, head_batch=0, steps=5)
    batches = list(sampler)
    assert len(batches) == 5 and all(len(b) == 8 for b in batches)


def test_bootstrap_numa_classe():
    m = {"a": {"display_ciclometrico": np.ones((1, 50))},
         "b": {"display_ciclometrico": np.zeros((1, 50))}}
    boot = bootstrap_mc(m, n_boot=20, classes=("display_ciclometrico",))
    assert np.allclose(boot["a"] - boot["b"], 1.0)
