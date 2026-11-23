import numpy as np

from training.reader_heads_eval import boot_macro_recall, f1_threshold, stratified_weights


def test_pesos_estratificados():
    groups = np.array([0] * 5 + [1] * 3)
    w = stratified_weights(groups, 50, np.random.default_rng(0))
    assert np.allclose(w[:, groups == 0].sum(1), 5) and np.allclose(w[:, groups == 1].sum(1), 3)


def test_revocacao_media_pontual():
    y = np.array([1, 1, 3, 3])
    hits = np.array([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=float)     # duas sementes
    out = boot_macro_recall(y, hits, np.ones((1, 4)))
    assert np.isclose(out[0], (0.75 + 0.5) / 2)


def test_limiar_f1():
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    assert f1_threshold(y, s) == 0.8
