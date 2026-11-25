from training.reader_rehearsal import (digit_detail, edit_distance, reading_kind, reading_ok,
                                       serial_kind, severity, tally, teacher_kind)

def test_leitura_com_tolerancia_so_no_ciclometrico():
    assert reading_ok("0012345", 12345, "display_digital")
    assert not reading_ok("12346", 12345, "display_digital")
    assert reading_ok("12346", 12345, "display_ciclometrico")
    assert not reading_ok("12A45", 12345, "display_ciclometrico")


def test_contagens_por_campo_e_origem():
    approved = [{"pseudonimo": "p1", "classe": "display_digital", "leitura": 10, "numero": "A1"},
                {"pseudonimo": "p2", "classe": "display_digital", "leitura": 20, "numero": "A2"},
                {"pseudonimo": "p3", "classe": "display_ciclometrico", "leitura": 30,
                 "numero": "A3"}]
    text = {("p1", "display_digital"): [("10", "humano_f2")],
            ("p2", "display_digital"): [("21", "haiku_csv")]}
    serial = {"p1": [("A1", "humano_f2")], "p3": [("B3", "haiku_csv")]}
    gold = {"p1": ("funcao", "03"), "p2": ("funcao", "24")}
    out = tally(approved, text, serial, gold)
    assert out["aprovadas"] == 3
    assert out["leitura"] == {"confere_humano_f2": 1, "diverge_haiku_csv": 1, "sem_rotulo": 1}
    assert out["numero"] == {"confere_humano_f2": 1, "sem_rotulo": 1, "diverge_haiku_csv": 1}
    assert out["funcao"] == {"consumo": 1, "nao_consumo": 1}


def test_tipos_de_divergencia():
    assert edit_distance("ABC", "ABD") == 1 and edit_distance("AB", "ABCD") == 2
    assert serial_kind("AB12", ("ab12", "9")) == "so_maiusculas_minusculas"
    assert serial_kind("9", ("AB12", "9")) == "outro_bloco_do_csv"
    assert serial_kind("XAB12", ("AB12",)) == "lido_contem_o_esperado"
    assert serial_kind("AB1", ("AB12",)) == "esperado_contem_o_lido"
    assert serial_kind("AB13", ("AB12",)) == "um_caractere"
    assert reading_kind(1235, 1234) == "diferenca_de_1"
    assert reading_kind(12340, 1234) == "numero_de_digitos"
    assert reading_kind(1834, 1234) == "um_digito"


def test_comparacao_com_o_professor():
    assert teacher_kind("AB12", None, None, "AB13") == "sem_resposta_do_professor"
    assert teacher_kind("AB12", "AB12", "discorda", "AB13") == "professor_igual_ao_lido"
    assert teacher_kind("AB12", "AB13", "aceito", "AB13") == "professor_igual_ao_csv"
    assert teacher_kind("AB12", "XY99", "discorda", "AB13") == "professor_outro"
    assert teacher_kind("AB12", "", "ilegivel", "AB13") == "professor_ilegivel"



def test_digito_a_mais_ou_a_menos():
    assert digit_detail(123457, 12345) == "digito_a_mais_no_fim"
    assert digit_detail(1234, 12345) == "digito_a_menos_no_fim"
    assert digit_detail(912345, 12345) == "digitos_a_mais_no_inicio"
    assert digit_detail(2345, 12345) == "digitos_a_menos_no_inicio"
    assert digit_detail(99, 12345) == "outro"


def test_gravidade_do_erro():
    assert severity("AB13", "AB12") == "leve_1"
    assert severity("XB13", "AB12") == "medio_2"
    assert severity("9999", "1234") == "grosseiro_3_ou_mais"
