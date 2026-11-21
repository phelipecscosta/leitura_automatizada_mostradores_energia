import numpy as np

from meter_reader.reader_text import TEXT_ALPHABET, encode
from training.reader_eval import (impostors, same_value, score_item, sequence_logprobs,
                                  value_logprob)


def _peaked(text, T=12):
    """Saída (T, C) que lê `text` com quase certeza: caractere, branco, caractere..."""
    lp = np.full((T, len(TEXT_ALPHABET) + 1), -20.0)
    lp[:, 0] = 0.0
    for j, k in enumerate(encode(text, TEXT_ALPHABET)):
        lp[2 * j + 1] = -20.0
        lp[2 * j + 1, k] = 0.0
    return lp


def test_probabilidade_da_sequencia():
    lp = _peaked("012")
    good, bad = sequence_logprobs(lp, ["012", "013"])
    assert good > -1e-3 and bad < -10


def test_valor_com_zeros_e_serie():
    lp = _peaked("0045")
    assert value_logprob(lp, "45", "display_digital") > -1e-3   # Z-a: 45 = 0045
    assert value_logprob(lp, "0045", "numero_serie") > -1e-3
    assert value_logprob(lp, "45", "numero_serie") < -10


def test_igualdade_numerica():
    assert same_value("0012", "12", "display_ciclometrico")
    assert not same_value("0012", "12", "numero_serie")


def test_impostores():
    u, c = impostors("00385", "display_digital", "F-x|display_digital|inteiro")
    assert u != "385" and len(u) <= 3
    assert c in {"885", "305", "335", "365", "395", "386"}
    assert impostors("2442", "display_digital", "k")[1] is None
    u2, c2 = impostors("AB123", "numero_serie", "k2")
    assert len(u2) == 5 and u2 != "AB123"
    assert impostors("AB123", "numero_serie", "k2") == (u2, c2)   # determinístico


def test_notas_do_item():
    out = score_item(_peaked("0045"), "45", "display_digital", "k")
    assert out["acerto"] and out["logp_rotulo"] > -1e-3
    assert not out["imp_uniforme_lido"]
