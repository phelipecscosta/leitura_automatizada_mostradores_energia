import pandas as pd

from training.reader_r16_list import to_judge


def test_so_aprovaveis_sem_julgamento_humano_em_ordem_fixa():
    table = pd.DataFrame({"pseudonimo": ["a", "b", "c", "d", "e"],
                          "lote": ["L1", "L1", "L2", "L2", "L2"],
                          "alta": [True, True, False, True, True],
                          "rotulo": ["aceito_professor", "humano", "marca_professor",
                                     "marca_professor", "sem_rotulo"]})
    out = to_judge(table)
    assert set(out["pseudonimo"]) == {"a", "d", "e"}
    assert list(out["ordem"]) == [1, 2, 3]
    assert out.equals(to_judge(table.iloc[::-1]))
