import numpy as np

from training.reader_rule import bootstrap_mc, choose

CLASSES = ("display_digital", "display_ciclometrico", "numero_serie")


def _matrix(rate, n=400, seeds=3, seed=0):
    rng = np.random.default_rng(seed)
    return {cls: (rng.random((seeds, n)) < rate).astype(float) for cls in CLASSES}


def test_bootstrap_pareado_centrado():
    m = {"A": _matrix(0.7), "B": _matrix(0.7)}   # mesmos recortes, mesma cobertura
    boot = bootstrap_mc(m, n_boot=200)
    assert np.allclose(boot["A"], boot["B"])
    assert abs(boot["A"].mean() - m["A"]["display_digital"].mean()) < 0.05


def test_regra_escolhe_o_mais_rapido_nao_inferior():
    mc = {"rapido": 0.69, "lento": 0.70, "fraco": 0.30}
    boot = {"rapido": np.full(100, 0.69) + np.linspace(-0.005, 0.005, 100),
            "lento": np.full(100, 0.70), "fraco": np.full(100, 0.30)}
    winner, table = choose(mc, boot, {"rapido": 10, "lento": 20, "fraco": 5})
    assert winner == "rapido"
    boot["rapido"] = boot["rapido"] - 0.03          # passa de -3 pontos
    mc["rapido"] = 0.66
    assert choose(mc, boot, {"rapido": 10, "lento": 20, "fraco": 5})[0] == "lento"
    assert choose(mc, boot, {"rapido": 10, "lento": 60, "fraco": 5})[0] == ""
