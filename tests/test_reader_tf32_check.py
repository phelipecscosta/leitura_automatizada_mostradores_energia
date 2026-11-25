import math

from training.reader_tf32_check import band_changes, verdict


def test_mudancas_de_faixa():
    old, new = math.log(0.20), math.log(0.21)
    confs = [math.log(0.19), math.log(0.205), math.log(0.5), -math.inf]
    assert band_changes(confs, old, new) == 1      # só o 0,205 fica entre os dois


def test_veredito_da_r19():
    assert verdict(0.0099, 5) == {"funcao_mantida": True, "inferior_mantido": True}
    assert verdict(0.0101, 6) == {"funcao_mantida": False, "inferior_mantido": False}
