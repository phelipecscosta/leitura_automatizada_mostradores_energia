import math

from meter_reader.reader_text import FUNCTION_CLASSES
from training.reader_product_parity import compare, decision, margins

LOW, T = math.log(0.2), 0.7258


def _f(p03):
    out = dict.fromkeys(FUNCTION_CLASSES, 0.0)
    out.update({"03": p03, "24": 1 - p03})
    return out


def _o(text="123", conf=-0.1, function=None, two=None):
    return {"text": text, "log_conf": conf, "function": function, "two_lines": two}


def test_decisao_reune_texto_faixa_d1_e_funcao():
    assert decision(**_o(), log_low=LOW, function_threshold=T) == ("123", False, None, None)
    assert decision(**_o(conf=-3.0), log_low=LOW, function_threshold=T)[1] is True
    assert decision(**_o(text=""), log_low=LOW, function_threshold=T)[1] is True
    d = decision(**_o(function=_f(0.9), two=False), log_low=LOW, function_threshold=T)
    assert d[3][0] is True and d[2] is False


def test_comparacao_conta_so_decisoes_que_mudam():
    a = [_o(conf=-0.1, function=_f(0.90)), _o(conf=-1.0, function=_f(0.80))]
    b = [_o(conf=-0.1 + 1e-6, function=_f(0.90 + 1e-6)), _o(conf=-2.0, function=_f(0.70))]
    r = compare(a, b, LOW, T)
    assert r["decisoes_divergentes"] == 1          # o segundo cai de faixa e perde a função
    assert r["por_parte"] == {"texto": 0, "faixa_muito_baixa": 1, "d1": 0, "funcao": 1}
    assert math.isclose(r["max_dif_log_conf"], 1.0)
    assert math.isclose(r["max_dif_prob_funcao"], 0.1, abs_tol=1e-9)


def test_folgas_ate_os_limiares():
    m = margins([_o(conf=LOW + 0.5, function=_f(T + 0.01)), _o(conf=-math.inf)], LOW, T)
    assert math.isclose(m["folga_limiar_inferior"], 0.5)
    assert math.isclose(m["folga_limiar_funcao"], 0.01, abs_tol=1e-9)
