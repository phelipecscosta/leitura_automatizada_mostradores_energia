import numpy as np

from training.reader_function_rule import FALLBACK_RATE, choose, errors_at, risk_curve


def test_erros_com_a_razao_dos_julgados():
    #            julgados (4)          não julgados (3)    F-2 (1)
    approved = [1, 1, 1, 1,             1, 1, 1,            1]
    nc =       [1, 0, 0, 0,             0, 0, 0,            1]
    judged =   [1, 1, 1, 1,             0, 0, 0,            0]
    unjudged = [0, 0, 0, 0,             1, 1, 1,            0]
    # diretos: 1 julgado + 1 da F-2; razão 1/4 em 3 não julgados -> ceil(0,75) = 1
    assert errors_at(approved, nc, judged, unjudged) == 3


def test_sem_julgado_aprovado_usa_a_fracao_geral():
    approved, nc = [0, 1], [0, 0]
    judged, unjudged = [1, 0], [0, 1]
    assert errors_at(approved, nc, judged, unjudged) == int(np.ceil(FALLBACK_RATE))


def test_curva_e_escolha_do_menor_limiar_seguro():
    n = 2000
    p = np.r_[np.linspace(0.5, 1.0, n), [0.6, 0.7]]
    high = np.ones(n + 2, bool)
    nc = np.r_[np.zeros(n, bool), [True, True]]
    zeros = np.zeros(n + 2, bool)
    curve = risk_curve(p, high, nc, zeros, zeros, [0.55, 0.65, 0.75])
    assert list(curve["erros"]) == [2, 1, 0]
    chosen = choose(curve)
    assert chosen is not None and chosen["limite_superior"] <= 0.007
    assert choose(curve, max_risk=1e-6) is None
