import pandas as pd

from training.reader_dataset import build_items, fold_masks, internal_mask, resolve_transition


def test_transicao_pela_alternativa_do_csv():
    assert resolve_transition("012[34]5", 1235) == "01235"
    assert resolve_transition("012[34]5", 1245) == "01245"
    assert resolve_transition("012[34]5", 9999) is None
    assert resolve_transition("00123", 123) == "00123"


def _labels():
    return pd.DataFrame({
        "pseudonimo": ["F-a", "F-b", "F-c", "F-d", "F-e"],
        "classe": ["display_digital", "display_digital", "display_ciclometrico",
                   "numero_serie", "display_digital"],
        "status": ["aceito", "humano", "aceito", "aceito", "discorda"],
        "texto": ["0012345", "0004567", "1[23]4", "AB123", "9"],
        "funcao": ["03", "", "", "", ""],
        "duas_leituras": ["False", "True", "", "", "False"],
        "origem": ["haiku_csv", "humano_f2", "haiku_csv", "haiku_csv", "haiku_csv"],
    })


def test_itens_e_mascaras():
    lots = {c: "L1" for c in ["F-a", "F-b", "F-c", "F-d", "F-e", "F-f"]}
    raw = pd.DataFrame({"pseudonimo": ["F-a", "F-f"], "classe": ["display_digital"] * 2,
                        "duas_leituras": ["False", "True"]})
    items, counts = build_items(_labels(), {"F-c": 124}, lots, raw)
    assert counts["transicao_resolvida"] == 1
    digital_a = items[items["pseudonimo"] == "F-a"].iloc[0]
    assert (digital_a["texto"], digital_a["funcao"]) == ("0012345", "03")
    assert digital_a[["usa_texto", "usa_funcao", "usa_duas"]].tolist() == [True, True, True]
    two = items[items["pseudonimo"] == "F-b"].set_index("recorte")
    assert two.loc["inteiro", ["usa_texto", "usa_duas"]].tolist() == [False, True]
    assert two.loc["faixa_cima", ["texto", "usa_texto", "usa_duas"]].tolist() == [
        "0004567", True, False]
    assert items[items["pseudonimo"] == "F-c"].iloc[0]["texto"] == "124"
    serial = items[items["pseudonimo"] == "F-d"].iloc[0]
    assert serial[["usa_funcao", "usa_duas"]].tolist() == [False, False]
    only_head = items[items["pseudonimo"] == "F-f"].iloc[0]
    assert only_head[["usa_texto", "duas_leituras"]].tolist() == [False, True]
    assert "F-e" not in set(items["pseudonimo"])   # não é alvo e não está no bruto


def test_divisao_interna_estavel_e_dobra():
    items = pd.DataFrame({"pseudonimo": [f"F-{i:04d}" for i in range(2000)],
                          "lote": ["A"] * 1000 + ["B"] * 1000})
    mask = internal_mask(items)
    assert 0.07 < mask.mean() < 0.13
    assert (internal_mask(items.iloc[::-1]) == mask[::-1]).all()   # não depende da ordem
    m = fold_masks(items, "B")
    assert not (m["interno"] & m["avaliacao"]).any()
    assert (m["treino"] | m["interno"] | m["avaliacao"]).all()
