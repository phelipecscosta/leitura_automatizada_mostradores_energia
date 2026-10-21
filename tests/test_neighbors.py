"""Testes da pureza dos vizinhos (H2), com dados sintéticos."""

import numpy as np

from training.neighbors import (
    balanced_purity,
    cosine_neighbors,
    neighbor_purity,
    paired_bootstrap_purity,
    standardize,
)


def test_standardize_zero_mean_unit_std_and_constant_column() -> None:
    x = np.array([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]])
    z = standardize(x)
    np.testing.assert_allclose(z[:, 0].mean(), 0, atol=1e-12)
    np.testing.assert_allclose(z[:, 0].std(), 1)
    np.testing.assert_array_equal(z[:, 1], 0)  # constante: sem divisão por zero


def test_cosine_neighbors_excludes_self_and_ignores_length() -> None:
    # Duas direções; o comprimento do vetor não muda o cosseno
    x = np.array([[1, 0], [10, 0.1], [0, 1], [0.1, 20]], dtype=float)
    nn = cosine_neighbors(x, k=1)
    assert nn[:, 0].tolist() == [1, 0, 3, 2]
    assert not (nn == np.arange(4)[:, None]).any()


def test_purity_on_separated_groups_is_one() -> None:
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal([5, 0], 0.1, (6, 2)), rng.normal([0, 5], 0.1, (6, 2))])
    labels = np.array([0] * 6 + [1] * 6)
    purity = neighbor_purity(cosine_neighbors(x, k=5), labels)
    np.testing.assert_array_equal(purity, 1.0)


def test_balanced_purity_weights_classes_equally() -> None:
    # 3 fotos da classe 0 com pureza 1 e 1 foto da classe 1 com pureza 0
    purity, labels = np.array([1.0, 1.0, 1.0, 0.0]), np.array([0, 0, 0, 1])
    assert balanced_purity(purity, labels) == 0.5      # e não 0,75 (média simples)


def test_bootstrap_of_a_constant_difference_is_that_constant() -> None:
    labels = np.array([0] * 20 + [1] * 5)
    purity_b = np.linspace(0, 1, 25)
    point, low, high = paired_bootstrap_purity(purity_b + 0.1, purity_b, labels, n_boot=200)
    np.testing.assert_allclose([point, low, high], 0.1)


def test_bootstrap_is_deterministic_and_brackets_the_point() -> None:
    rng = np.random.default_rng(1)
    labels = np.array([0] * 40 + [1] * 8)
    a, b = rng.random(48), rng.random(48)
    first = paired_bootstrap_purity(a, b, labels, n_boot=300)
    assert first == paired_bootstrap_purity(a, b, labels, n_boot=300)
    assert first[1] <= first[0] <= first[2]
