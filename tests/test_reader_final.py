import pandas as pd
import pytest

from training.reader_final import intersect_folds


def _fold(rows):
    return pd.DataFrame(rows, columns=["pseudonimo", "classe", "lote", "texto"])


def test_interseccao_exige_as_duas_dobras_e_o_mesmo_texto():
    folds = {
        "A": _fold([("p1", "d", "B", "123"), ("p2", "d", "B", "45"), ("p3", "d", "C", "7")]),
        "B": _fold([("p4", "d", "A", "9"), ("p3", "d", "C", "7")]),
        "C": _fold([("p1", "d", "B", "123"), ("p2", "d", "B", "045"), ("p4", "d", "A", "9")]),
    }
    out = intersect_folds(folds).sort_values("pseudonimo").reset_index(drop=True)
    # p1, p3 e p4: aceitos nas duas dobras com o mesmo texto; p2: textos diferentes
    assert out["pseudonimo"].tolist() == ["p1", "p3", "p4"]
    assert out["texto"].tolist() == ["123", "7", "9"]


def test_interseccao_recusa_pseudo_rotulo_do_lote_avaliado():
    folds = {"A": _fold([("p1", "d", "A", "1")]), "B": _fold([]), "C": _fold([])}
    with pytest.raises(ValueError):
        intersect_folds(folds)



def test_artefato_carrega_com_weights_only(tmp_path):
    import torch

    from training.reader_final import build_artifact, save_artifact

    state = [{"w": torch.zeros(2)} for _ in range(3)]
    limits = {"d1_logit": 3.0, "limiar_inferior_log": -1.5, "temperatura_texto": 1.4,
              "limiar_funcao": 0.7, "limiar_numero_log": -1.7, "fonte": "texto"}
    data = build_artifact(state, state, limits, {"torch": torch.__version__, "n": 1})
    path = tmp_path / "a.pt"
    save_artifact(path, data)
    back = torch.load(path, map_location="cpu", weights_only=True)
    assert back["info"] == {"torch": str(torch.__version__), "n": 1}
    assert "fonte" not in back["limiares"]
