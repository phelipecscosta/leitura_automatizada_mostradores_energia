import pytest

from meter_reader.reader_text import (FUNCTION_ALPHABET, TEXT_ALPHABET, encode, greedy_decode,
                                      reading_variants)


def test_codifica_e_decodifica():
    indices = encode("0A9", TEXT_ALPHABET)
    assert indices == [1, 11, 10]
    # caminho com brancos e repetições: 0 0 _ A _ _ 9 9
    assert greedy_decode([1, 1, 0, 11, 0, 0, 10, 10], TEXT_ALPHABET) == "0A9"
    # dígito repetido de verdade precisa de um branco entre eles
    assert greedy_decode([2, 0, 2], TEXT_ALPHABET) == "11"


def test_recusa_fora_do_alfabeto():
    with pytest.raises(ValueError):
        encode("03A", FUNCTION_ALPHABET)


def test_variantes_da_leitura():
    assert reading_variants(12345) == ["12345", "012345", "0012345", "00012345"]
    assert reading_variants(0, max_length=3) == ["0", "00", "000"]
    assert reading_variants(123456789) == ["123456789"]   # maior que o limite: só ele
