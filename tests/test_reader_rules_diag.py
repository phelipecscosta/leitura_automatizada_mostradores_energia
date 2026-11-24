import numpy as np
import pandas as pd

from meter_reader.fields import FieldSelection
from training.reader_rules_diag import field_combinations, selection_situation


def test_combinacoes_dos_campos():
    crops = pd.DataFrame({"pseudonimo": ["a", "a", "b", "c"],
                          "classe": ["display_digital", "numero_serie",
                                     "display_ciclometrico", "numero_serie"]})
    out = field_combinations(crops, {"a", "b", "c", "d"}).set_index("pseudonimo")
    assert tuple(out.loc["a"]) == ("digital", "display_e_serie")
    assert tuple(out.loc["b"]) == ("ciclometrico", "so_display")
    assert tuple(out.loc["c"]) == ("", "so_serie")
    assert tuple(out.loc["d"]) == ("", "nenhum")


def test_situacao_da_selecao():
    box = np.zeros(4)
    many = FieldSelection("ciclometrico", None, None, "mais_de_dois_displays")
    assert selection_situation(many) == "mais_de_dois_displays"
    assert selection_situation(FieldSelection(None, None, box)) == "sem_display_com_serie"
    assert selection_situation(FieldSelection(None, None, None)) == "sem_display_sem_serie"
    assert selection_situation(FieldSelection("digital", box, None)) == "consumo_identificado"
