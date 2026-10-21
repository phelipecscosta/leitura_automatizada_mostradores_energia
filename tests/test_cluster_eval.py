"""Testes do mapeamento dos clusters e das métricas da triagem de ML."""

import numpy as np
import pytest

from training.cluster_eval import (
    cluster_outros_rates,
    clustering_metrics,
    kmeans_clusters,
    l2_normalize,
    leave_one_lot_out,
    meter_recall_lower,
    one_hot,
)


def test_rates_are_smoothed_and_mapping_needs_strict_majority() -> None:
    clusters = np.array([0, 0, 0, 1, 1])
    is_outros = np.array([1, 1, 0, 1, 0])        # prevalência 0,6
    rates, mapped = cluster_outros_rates(clusters, is_outros, n_clusters=3)
    np.testing.assert_allclose(rates, [(2 + 0.6) / 4, (1 + 0.6) / 3, 0.6])
    assert mapped.tolist() == [True, False, False]  # empate (cluster 1) e vazio (2): medidor


def test_measured_lot_never_maps_itself() -> None:
    # No lote A, o cluster 0 só tem "outros"; no lote B, só medidores
    clusters = np.array([0, 0, 0, 0])
    is_outros = np.array([1, 1, 0, 0])
    lots = np.array(["A", "A", "B", "B"])
    score, predicted = leave_one_lot_out(one_hot(clusters, 2), is_outros, lots)
    assert predicted.tolist() == [False, False, True, True]  # cada lote usa o outro
    np.testing.assert_allclose(score[:2], (0 + 0.0) / 3)     # prevalência de B = 0
    np.testing.assert_allclose(score[2:], (2 + 1.0) / 3)     # prevalência de A = 1


def test_soft_membership_weights_the_rates() -> None:
    membership = np.array([[0.5, 0.5], [1, 0], [0, 1]])
    is_outros = np.array([0, 1, 0])
    lots = np.array(["A", "B", "B"])
    score, _ = leave_one_lot_out(membership, is_outros, lots)
    rates, _ = cluster_outros_rates(np.array([0, 1]), np.array([1, 0]), 2)
    np.testing.assert_allclose(score[0], 0.5 * rates[0] + 0.5 * rates[1])


def test_meter_recall_lower_bound() -> None:
    is_outros = np.array([0] * 100)
    assert meter_recall_lower(np.zeros(100, dtype=bool), is_outros) == pytest.approx(1 - 0.0295, abs=1e-3)
    assert meter_recall_lower(np.ones(100, dtype=bool), is_outros) == pytest.approx(0.0)


def test_metrics_of_all_meter_prediction() -> None:
    # Mandar tudo para "medidor": revocação de medidor 1, mas nenhum "outros"
    is_outros = np.array([0] * 8 + [1] * 2)
    metrics = clustering_metrics(np.linspace(0, 1, 10), np.zeros(10, dtype=bool), is_outros)
    assert metrics["revocacao_medidor"] == 1.0 and metrics["revocacao_outros"] == 0.0
    assert metrics["acuracia_balanceada"] == 0.5 and metrics["ap_outros"] == 1.0


def test_kmeans_is_deterministic_and_uses_unit_vectors() -> None:
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal([5, 0], 0.1, (10, 2)), rng.normal([0, 5], 0.1, (10, 2))])
    first = kmeans_clusters(x * rng.uniform(1, 9, (20, 1)), 2, seed=0)  # comprimentos variados
    assert len(set(first[:10])) == 1 and len(set(first[10:])) == 1 and first[0] != first[10]
    np.testing.assert_array_equal(first, kmeans_clusters(x * rng.uniform(1, 9, (20, 1)), 2, seed=0))
    np.testing.assert_allclose(np.linalg.norm(l2_normalize(x), axis=1), 1)
