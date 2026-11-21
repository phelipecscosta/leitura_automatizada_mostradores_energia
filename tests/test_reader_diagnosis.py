import pandas as pd

from training.reader_diagnosis import error_kind, error_profile


def test_tipos_de_erro():
    assert error_kind("0012345", "12345")[0] == "certo"          # zeros à esquerda (Z-a)
    assert error_kind("", "123")[0] == "vazio"
    assert error_kind("12A45", "12345")[0] == "com_letra"
    assert error_kind("1235", "12335")[0] == "falta_repetido"
    assert error_kind("1234", "12345")[0] == "falta_digito"
    assert error_kind("123456", "12345")[0] == "sobra_digito"
    assert error_kind("12385", "12345") == ("um_digito", 2, "4>8")
    assert error_kind("99945", "12345")[0] == "varios_digitos"


def test_perfil_dos_erros():
    items = pd.DataFrame({"pseudonimo": ["A", "B", "C"], "classe": ["display_ciclometrico"] * 3,
                          "recorte": ["inteiro"] * 3, "texto": ["0123", "456", "789"]})
    scores = pd.DataFrame({"chave": ["A|display_ciclometrico|inteiro",
                                     "B|display_ciclometrico|inteiro",
                                     "C|display_ciclometrico|inteiro"],
                           "classe": ["display_ciclometrico"] * 3,
                           "origem": ["haiku_csv", "haiku_csv", "sonnet_csv"],
                           "livre": ["123", "486", None], "acerto": [True, False, False]})
    out = error_profile(scores, items)
    assert out["tipos"]["leituras"].to_dict() == {"certo": 1, "um_digito": 1, "vazio": 1}
    assert out["pares_um_digito"].index.tolist() == ["5>8"]
    assert out["posicao_um_digito"].index.tolist() == [2]
