import pandas as pd

from training.reader_function_r12 import add_function_labels


def _items():
    return pd.DataFrame([
        {"pseudonimo": "A", "classe": "display_digital", "lote": "L", "recorte": "inteiro",
         "texto": "", "funcao": "", "duas_leituras": False, "usa_texto": False,
         "usa_funcao": False, "usa_duas": True, "origem": "marca_professor"}])


def test_funcao_da_r12():
    judged = pd.DataFrame({"pseudonimo": ["A", "B", "C", "D"], "lote": ["L"] * 4,
                           "status": ["funcao", "funcao", "vazio", "ilegivel"],
                           "funcao": ["103", "24", "", ""]})
    out = add_function_labels(_items(), judged, {"B": True}).set_index("pseudonimo")
    assert len(out) == 3                                     # A atualizado; B e C novos; D fora
    assert out.loc["A", "funcao"] == "103" and out.loc["A", "usa_funcao"]
    assert out.loc["A", "usa_duas"]                          # a marca de duas leituras continua
    assert out.loc["B", "recorte"] == "faixa_cima"           # professor marcou duas leituras
    assert out.loc["C", "funcao"] == "" and out.loc["C", "usa_funcao"]
    assert out.loc["C", "origem"] == "humano_r12"
    