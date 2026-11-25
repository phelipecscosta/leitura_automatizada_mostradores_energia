import numpy as np

from training.reader_thresholds import committee_pick


def test_comite_escolhe_a_maior_media_e_ignora_vazias():
    # 3 candidatos x 3 sementes x 2 itens
    probs = np.array([[[0.9, 0.0], [0.1, 0.0], [0.2, 0.0]],     # candidato da semente 0
                      [[0.3, 0.5], [0.8, 0.6], [0.7, 0.7]],     # candidato da semente 1
                      [[0.3, 0.2], [0.8, 0.1], [0.7, 0.1]]])    # igual ao 1 no item 0
    texts = [["123", ""], ["128", "45"], ["128", "40"]]
    out, p = committee_pick(probs, texts)
    assert out == ["128", "45"]
    assert np.isclose(p[0], 0.6) and np.isclose(p[1], 0.6)


def test_comite_sem_leitura():
    out, p = committee_pick(np.zeros((3, 3, 1)), [[""], [""], [""]])
    assert out == [""] and p[0] == 0.0
