"""Testes do mapeamento dos clusters e das métricas da triagem de ML."""

import numpy as np
import pytest

from training.cluster_eval import (
    cluster_outros_rates,
    clustering_metrics,
    kmeans_clusters,
    l2_normalize,
    leave_one_lot_out,
    mapped_outros_score,
    mean_ap,
    meter_recall_lower,
    one_hot,
    paired_bootstrap_mean_ap,
    typical_photos,
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



def test_mean_ap_averages_seeds() -> None:
    is_outros = np.array([0, 0, 1, 1])
    perfect, inverted = np.array([0.1, 0.2, 0.8, 0.9]), np.array([0.9, 0.8, 0.2, 0.1])
    assert mean_ap(np.stack([perfect, perfect]), is_outros) == 1.0
    # Invertida: AP = (1/3 + 2/4) / 2 = 5/12 (os dois "outros" no fim da ordem)
    assert mean_ap(np.stack([perfect, inverted]), is_outros) == pytest.approx((1 + 5 / 12) / 2)


def test_paired_bootstrap_mean_ap() -> None:
    rng = np.random.default_rng(0)
    is_outros = np.array([0] * 60 + [1] * 10)
    good = np.stack([is_outros + rng.normal(0, 0.3, 70) for _ in range(3)])
    noise = np.stack([rng.random(70) for _ in range(3)])
    point, low, high = paired_bootstrap_mean_ap(good, noise, is_outros, n_boot=300)
    assert low > 0 and low <= point <= high                 # o bom vence o ruído
    same = paired_bootstrap_mean_ap(good, good, is_outros, n_boot=50)
    assert same == (0.0, 0.0, 0.0)                          # mesmas predições: diferença nula



def test_typical_photos_ranks_members_and_pages() -> None:
    probs = np.array([
        [0.9, 0.1], [0.6, 0.4], [0.8, 0.2],   # cluster 0, típicas: 0, 2, 1
        [0.3, 0.7], [0.45, 0.55],             # cluster 1
    ])
    assert typical_photos(probs, 0).tolist() == [0, 2, 1]
    assert typical_photos(probs, 0, start=0, count=2).tolist() == [0, 2]
    assert typical_photos(probs, 0, start=2, count=2).tolist() == [1]   # segunda rodada
    assert typical_photos(probs, 1).tolist() == [3, 4]


def test_mapped_outros_score_sums_mapped_clusters() -> None:
    membership = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    score = mapped_outros_score(membership, np.array([False, True, True]))
    np.testing.assert_allclose(score, [0.3, 0.9])
