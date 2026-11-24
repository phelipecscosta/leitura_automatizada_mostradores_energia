import pandas as pd
import pytest

from training.reader_r15_sample import draw


def _labels(n_ok=5):
    rows = [{"pseudonimo": f"F-{i}", "classe": "display_digital", "status": "aceito",
             "origem": "haiku_csv"} for i in range(n_ok)]
    rows += [{"pseudonimo": "F-x", "classe": "display_digital", "status": "aceito",
              "origem": "humano_f2"},
             {"pseudonimo": "F-y", "classe": "numero_serie", "status": "aceito",
              "origem": "haiku_csv"},
             {"pseudonimo": "F-z", "classe": "display_digital", "status": "parcial",
              "origem": "haiku_csv"}]
    return pd.DataFrame(rows)


def test_sorteio_so_dos_aceitos_digitais_do_professor_e_fixo():
    lots = {f"F-{i}": "L" for i in range(5)}
    a = draw(_labels(), lots, n=3)
    assert list(a["ordem"]) == [1, 2, 3]
    assert set(a["pseudonimo"]) <= set(lots)
    assert a.equals(draw(_labels().sample(frac=1, random_state=1), lots, n=3))


def test_recusa_amostra_maior_que_a_populacao():
    with pytest.raises(ValueError):
        draw(_labels(2), {"F-0": "L", "F-1": "L"}, n=3)
