import math

import numpy as np

from training.reader_bands import band, lower_threshold


def test_limiar_inferior():
    # 40 recortes: 2 "sim" entre os 35 de menor confiança; dali em diante, só "sim".
    # Com t = 36: 36 abaixo e 3 "sim" (8,3%); com t = 37: 4 em 37 (10,8%), acima de V
    conf = np.arange(40, dtype=float)
    yes = np.zeros(40, dtype=bool)
    yes[[5, 20]] = True
    yes[35:] = True
    out = lower_threshold(conf, yes, max_yes=0.10, min_below=30)
    assert out["log_limiar"] == 36.0 and out["abaixo"] == 36 and out["sim_abaixo"] == 3


def test_limiar_inferior_sem_solucao():
    out = lower_threshold(np.arange(20, dtype=float), np.ones(20, dtype=bool))
    assert out["log_limiar"] is None


def test_sem_leitura_conta_abaixo():
    conf = np.array([-math.inf] * 30 + [0.0] * 10)
    yes = np.array([False] * 30 + [True] * 10)
    out = lower_threshold(conf, yes)
    assert out["log_limiar"] == 0.0 and out["abaixo"] == 30


def test_faixas():
    assert band("012345", -0.1, -0.1, 12345, -1.0, -5.0) == "alta"
    assert band("12345", -0.1, -2.0, 12345, -1.0, -5.0) == "intermediaria"   # CSV abaixo do alto
    assert band("12346", -6.0, -9.0, 12345, -1.0, -5.0) == "muito_baixa"
    assert band("", -math.inf, -9.0, 12345, -1.0, None) == "muito_baixa"
