import pytest

from training.teacher_eval import (ACCEPTED, DISAGREE, FORMAT_ERROR, ILLEGIBLE, PARTIAL,
                                   QUEUE_F2, decimal_drum_suspect, label_status,
                                   reading_values, stable)


def _rec(**resp):
    return {"resposta": resp, "erro": None}


def test_reading_values_expands_transitions_and_strips_zeros():
    assert reading_values("0124[56]") == {1245, 1246}
    assert reading_values("[90]12") == {912, 12}
    assert reading_values("01?45") is None


@pytest.mark.parametrize("cls,rec,status", [
    ("display_digital", _rec(legivel=True, leitura="0012345", funcao="03", duas_leituras=False), ACCEPTED),
    ("display_digital", _rec(legivel=True, leitura="0012345", funcao="", duas_leituras=True), ACCEPTED),
    ("display_digital", _rec(legivel=True, leitura="0012345", funcao="103", duas_leituras=False), QUEUE_F2),
    ("display_digital", _rec(legivel=True, leitura="0099999", funcao="03", duas_leituras=False), QUEUE_F2),
    ("display_ciclometrico", _rec(legivel=True, leitura="0123[45]"), ACCEPTED),
    ("display_ciclometrico", _rec(legivel=True, leitura="0123[56]"), DISAGREE),
    ("display_ciclometrico", _rec(legivel=True, leitura="01?34"), PARTIAL),
    ("display_ciclometrico", _rec(legivel=False, leitura=""), ILLEGIBLE),
    ("numero_serie", _rec(legivel=True, numero="AB123"), ACCEPTED),
    ("numero_serie", _rec(legivel=True, numero="AB124"), DISAGREE),
    ("display_ciclometrico", {"resposta": None, "erro": "Texto principal fora do formato."}, FORMAT_ERROR),
])
def test_label_status(cls, rec, status):
    # CSV: medidor "ab123" (o professor normaliza para maiúsculas), leitura 12345 / 1234
    reading = 1234 if cls == "display_ciclometrico" else 12345
    assert label_status(cls, rec, "ab123", reading) == status


def test_decimal_drum_suspect():
    assert decimal_drum_suspect(_rec(legivel=True, leitura="012345"), 1234)
    assert not decimal_drum_suspect(_rec(legivel=True, leitura="01234"), 1234)
    assert not decimal_drum_suspect(_rec(legivel=False, leitura=""), 1234)


def test_stable_compares_answer_and_error():
    a = _rec(legivel=True, leitura="0123")
    assert stable(a, _rec(legivel=True, leitura="0123"))
    assert not stable(a, _rec(legivel=True, leitura="0124"))

from training.teacher_eval import discord_kind, levenshtein, mask


def test_levenshtein_and_discord_kind():
    assert levenshtein("12345", "12845") == 1
    assert discord_kind("12845", "12345") == "um_caractere"
    assert discord_kind("12345", "9912345") == "contido"
    assert discord_kind("55555", "12345") == "dois_ou_mais"


def test_mask_hides_content():
    assert mask("AB12 3.4") == "AA99 9.9"

def test_function_003_counts_as_consumption():
    rec = _rec(legivel=True, leitura="0012345", funcao="003", duas_leituras=False)
    assert label_status("display_digital", rec, "ab123", 12345) == ACCEPTED
