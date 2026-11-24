import pandas as pd

from training.reader_function_gold import decide, gold_labels


def test_rotulo_vigente_e_o_julgamento_mais_recente():
    f2 = pd.DataFrame({"pseudonimo": ["a", "b"], "lote": ["L", "L"], "funcao": ["103", ""]})
    r12 = pd.DataFrame({"pseudonimo": ["c"], "lote": ["L"], "status": ["ilegivel"], "funcao": [""]})
    r16 = pd.DataFrame({"pseudonimo": ["a", "c"], "lote": ["L", "L"],
                        "status": ["funcao", "funcao"], "funcao": ["03", "24"]})
    g = gold_labels(f2, [("r12", r12), ("r16", r16)]).set_index("pseudonimo")
    assert tuple(g.loc["a", ["status", "funcao", "fonte"]]) == ("funcao", "03", "r16")
    assert tuple(g.loc["b", ["status", "fonte"]]) == ("vazio", "f2")
    assert tuple(g.loc["c", ["funcao", "fonte"]]) == ("24", "r16")


def test_escolha_pela_r16():
    ok = lambda kept: {"escolhido": {"consumo_mantido": kept}}
    none = {"escolhido": None}
    assert decide({"R12": none, "R12g": none}) is None
    assert decide({"R12": ok(0.60), "R12g": ok(0.605)}) == "R12"
    assert decide({"R12": ok(0.60), "R12g": ok(0.70)}) == "R12g"
    assert decide({"R12": none, "R12g": ok(0.50)}) == "R12g"
