"""Tabela de decisão do leitor (T6c.1): uma linha da tabela por teste."""

import math

import pytest

from meter_reader.contracts import ExtractionRow, Reason, Status, wrong_function
from meter_reader.validation import FieldReading, FunctionDecision, decide, reading_matches

ROW = ExtractionRow(("A123", "B9"), 12345, None, "a.jpg")
LOW = math.log(0.202)
OK = FieldReading("0012345", -0.1)
SERIAL = FieldReading("A123", -0.1)
F_OK = FunctionDecision(True, "03")
F_103 = FunctionDecision(False, wrong_function("103"), Reason.WRONG_FUNCTION)


def _decide(family="digital", display=OK, serial=SERIAL, function=F_OK, row=ROW):
    if family != "digital":
        function = None
    return decide("a.jpg", row, family, display, serial, function, LOW)


def test_aprova_com_numero_e_leitura_do_csv():
    out = _decide()
    assert (out.status, out.reason) == (Status.APPROVED, None)
    assert (out.meter_number, out.function, out.reading) == ("A123", "03", 12345)


def test_ciclometrico_aprova_sem_funcao():
    out = _decide("ciclometrico")
    assert out.status is Status.APPROVED and out.function is None


@pytest.mark.parametrize("display", [None, FieldReading("", -math.inf),
                                     FieldReading("12345", math.log(0.1))])
def test_muito_baixa_com_numero_que_confere_vai_para_verificar(display):
    out = _decide(display=display)
    assert (out.status, out.reason) == (Status.NEEDS_REVIEW, Reason.LOW_CONFIDENCE)
    assert (out.meter_number, out.reading) == ("A123", None)


@pytest.mark.parametrize("serial", [None, FieldReading("A128", -0.1)])
def test_muito_baixa_sem_numero_que_confere_e_falha_na_extracao(serial):
    out = _decide(display=FieldReading("12345", math.log(0.1)), serial=serial)
    assert (out.status, out.reason) == (Status.REJECTED_LOW_QUALITY, Reason.EXTRACTION_FAILURE)


def test_so_numero_de_serie_que_confere():
    out = decide("a.jpg", ROW, None, None, SERIAL, None, LOW)
    assert (out.status, out.reason) == (Status.NEEDS_REVIEW, Reason.LOW_CONFIDENCE)


def test_sem_numero_de_serie_nao_localizado():
    out = _decide(serial=None)
    assert (out.status, out.reason) == (Status.NEEDS_REVIEW, Reason.METER_NUMBER_NOT_FOUND)
    assert (out.meter_number, out.reading) == (None, 12345)


def test_numero_divergente_vem_antes_da_leitura_e_da_funcao():
    out = _decide(display=FieldReading("12346", -0.1), serial=FieldReading("A128", -0.1),
                  function=F_103)
    assert out.reason is Reason.METER_NUMBER_MISMATCH
    assert (out.meter_number, out.reading, out.function) == ("A128", 12346, "Função incorreta - 103")


def test_leitura_divergente_grava_o_valor_lido():
    out = _decide(display=FieldReading("0012399", -0.1))
    assert (out.status, out.reason, out.reading) == (Status.NEEDS_REVIEW,
                                                      Reason.READING_MISMATCH, 12399)


def test_leitura_com_letra_e_divergente_sem_valor():
    out = _decide(display=FieldReading("12A45", -0.1))
    assert (out.reason, out.reading) == (Reason.READING_MISMATCH, None)


def test_funcao_incorreta():
    out = _decide(function=F_103)
    assert (out.status, out.reason) == (Status.NEEDS_REVIEW, Reason.WRONG_FUNCTION)
    assert out.function == "Função incorreta - 103"


def test_transicao_so_no_ciclometrico():
    assert reading_matches("12346", 12345, "ciclometrico")
    assert reading_matches("12344", 12345, "ciclometrico")
    assert reading_matches("12399", 12400, "ciclometrico")      # vai-um
    assert not reading_matches("12347", 12345, "ciclometrico")
    assert not reading_matches("12346", 12345, "digital")
    out = _decide("ciclometrico", display=FieldReading("12346", -0.1))
    assert (out.status, out.reading) == (Status.APPROVED, 12345)   # vale o CSV (T-a)


def test_decisao_da_funcao_coerente():
    with pytest.raises(ValueError):
        FunctionDecision(True, wrong_function("103"))
    with pytest.raises(ValueError):
        FunctionDecision(False, None, Reason.READING_MISMATCH)
    with pytest.raises(ValueError):
        _decide(function=None)
    with pytest.raises(ValueError):
        decide("a.jpg", ROW, "ciclometrico", OK, SERIAL, F_OK, LOW)



def _probs(**kw):
    from meter_reader.reader_text import FUNCTION_CLASSES
    names = {"vazio": "vazio", "f03": "03", "f003": "003", "f103": "103", "f24": "24",
             "f55": "55", "outro": "outro"}
    out = dict.fromkeys(FUNCTION_CLASSES, 0.0)
    out.update({names[k]: v for k, v in kw.items()})
    return out


T = 0.7258


def test_funcao_aceita_com_03_ou_vazio():
    from meter_reader.validation import function_decision
    assert function_decision(_probs(f03=0.5, f003=0.3, f24=0.2), T) == FunctionDecision(True, "03")
    assert function_decision(_probs(vazio=0.6, f03=0.2, f24=0.2), T) == FunctionDecision(True, None)
    # na igualdade com o limiar, aceita (a curva da R16 usou >=)
    assert function_decision(_probs(f03=T, f24=1 - T), T).accepted


def test_funcao_retida_por_funcao_incorreta():
    from meter_reader.validation import function_decision
    d = function_decision(_probs(f03=0.3, f24=0.7), T)
    assert d == FunctionDecision(False, wrong_function("24"), Reason.WRONG_FUNCTION)
    d = function_decision(_probs(f03=0.3, outro=0.7), T)
    assert d == FunctionDecision(False, None, Reason.WRONG_FUNCTION)


def test_funcao_retida_por_confianca_baixa():
    from meter_reader.validation import function_decision
    d = function_decision(_probs(f03=0.5, f103=0.3, f24=0.2), T)
    assert d == FunctionDecision(False, "03", Reason.LOW_CONFIDENCE)
    d = function_decision(_probs(vazio=0.4, f103=0.35, f24=0.25), T)
    assert d == FunctionDecision(False, None, Reason.LOW_CONFIDENCE)


def test_funcao_exige_todas_as_classes():
    from meter_reader.validation import function_decision
    with pytest.raises(ValueError):
        function_decision({"03": 1.0}, T)
