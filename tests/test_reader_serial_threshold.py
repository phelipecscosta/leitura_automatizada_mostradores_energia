import math

from training.reader_serial_threshold import effect


def test_efeito_so_nos_divergentes():
    conf = [math.log(0.05), math.log(0.05), math.log(0.9), math.log(0.05), math.log(0.9)]
    mismatch = [True, True, True, False, True]
    legible = [False, True, False, False, True]
    out = effect(conf, mismatch, legible, math.log(0.1))
    assert out == {"divergentes": 4, "viram_nao_localizado": 2, "legiveis_entre_eles": 1,
                   "ilegiveis_que_seguem_divergentes": 1}
