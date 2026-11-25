import pandas as pd

from training.reader_function_r18 import ILLEGIBLE, adopt, choose_t2, class_fn_i


def test_oitava_classe_so_para_o_codigo_interno():
    assert class_fn_i(ILLEGIBLE) == 7
    assert class_fn_i("03") == 1 and class_fn_i("") == 0 and class_fn_i("999") == 6


def test_limiar_t2_menor_limite_e_menor_t_no_empate():
    curve = pd.DataFrame({"limiar": [0.5, 0.6, 0.7, 0.8],
                          "aprovados": [100, 90, 80, 70], "erros": [3, 2, 1, 1],
                          "limite_superior": [0.012, 0.009, 0.008, 0.008],
                          "consumo_mantido": [0.9, 0.8, 0.7, 0.6]})
    assert choose_t2(curve)["limiar"] == 0.7
    assert choose_t2(curve, max_risk=0.001) is None


def test_adocao_pela_r18():
    def cand(kept, ill):
        return {"escolhido": {"consumo_mantido": kept}, "ilegiveis_aprovados": ill}
    r12 = cand(0.61, 0.60)
    assert adopt(r12, cand(0.55, 0.30))
    assert not adopt(r12, cand(0.45, 0.30))          # abaixo do piso
    assert not adopt(r12, cand(0.70, 0.60))          # não corta ilegíveis
    assert not adopt(r12, {"escolhido": None, "ilegiveis_aprovados": 0.0})

def test_codigo_interno_cabe_no_alfabeto_e_nao_colide():
    from meter_reader.reader_text import FUNCTION_ALPHABET, encode
    assert encode(ILLEGIBLE, FUNCTION_ALPHABET)
    assert len(ILLEGIBLE) > 6
