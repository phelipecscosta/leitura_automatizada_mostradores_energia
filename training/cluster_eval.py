"""Mapeamento dos clusters e métricas da triagem de ML (protocolo de ML, 1.5 e 1.6).

Os rótulos entram só aqui, depois do agrupamento. Cada lote é avaliado com o
mapeamento e as frações de "outros" calculados nos outros dois lotes, para
que o lote medido nunca mapeie a si mesmo.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score

from training.risk_coverage import clopper_pearson_upper

SMOOTHING_M = 1.0  # peso da prevalência na suavização das frações (seção 1.6)


def l2_normalize(features: np.ndarray) -> np.ndarray:
    """Vetores de comprimento 1: a distância euclidiana passa a ordenar como o cosseno."""
    return features / np.linalg.norm(features, axis=1, keepdims=True)


def kmeans_clusters(features: np.ndarray, n_clusters: int, seed: int) -> np.ndarray:
    """Cluster de cada foto pelo K-means sobre os vetores normalizados (seção 1.4)."""
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    return model.fit_predict(l2_normalize(features))


def one_hot(clusters: np.ndarray, n_clusters: int) -> np.ndarray:
    """Pertinência rígida (n, K), para usar o K-means como o SCAN."""
    return np.eye(n_clusters)[clusters]


def cluster_outros_rates(clusters: np.ndarray, is_outros: np.ndarray,
                         n_clusters: int) -> tuple[np.ndarray, np.ndarray]:
    """(fração suavizada de "outros" por cluster, cluster mapeado como "outros").

    O mapeamento exige maioria estrita de "outros"; cluster vazio ou empatado
    fica "medidor", o lado seguro (E09).
    """
    prior = is_outros.mean()
    n_outros = np.bincount(clusters[is_outros == 1], minlength=n_clusters)
    n_total = np.bincount(clusters, minlength=n_clusters)
    rates = (n_outros + SMOOTHING_M * prior) / (n_total + SMOOTHING_M)
    return rates, n_outros > n_total - n_outros


def leave_one_lot_out(membership: np.ndarray, is_outros: np.ndarray,
                      lots: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(escore de "outros", previsão rígida de "outros") de cada foto rotulada.

    `membership` é a pertinência (n, K) das fotos rotuladas aos clusters.
    """
    clusters = membership.argmax(axis=1)
    n_clusters = membership.shape[1]
    score = np.empty(len(lots))
    predicted = np.empty(len(lots), dtype=bool)
    for lot in np.unique(lots):
        measured = lots == lot
        rates, mapped = cluster_outros_rates(clusters[~measured], is_outros[~measured], n_clusters)
        score[measured] = membership[measured] @ rates
        predicted[measured] = mapped[clusters[measured]]
    return score, predicted


def meter_recall_lower(predicted_outros: np.ndarray, is_outros: np.ndarray,
                       alpha: float = 0.05) -> float:
    """Limite inferior unilateral (1 - alpha) de Clopper-Pearson da revocação de medidor."""
    meters = is_outros == 0
    lost = int(predicted_outros[meters].sum())  # medidores mandados para "outros"
    return 1.0 - clopper_pearson_upper(lost, int(meters.sum()), alpha)


def clustering_metrics(score: np.ndarray, predicted_outros: np.ndarray,
                       is_outros: np.ndarray) -> dict:
    """Métrica otimizadora, restrição e descritivas (seção 1.6)."""
    meters = is_outros == 0
    return {
        "ap_outros": float(average_precision_score(is_outros, score)),
        "revocacao_medidor": float((~predicted_outros[meters]).mean()),
        "revocacao_medidor_lim_inf": meter_recall_lower(predicted_outros, is_outros),
        "revocacao_outros": float(predicted_outros[~meters].mean()),
        "acuracia_balanceada": float(balanced_accuracy_score(is_outros, predicted_outros)),
        "f1_macro": float(f1_score(is_outros, predicted_outros, average="macro", zero_division=0)),
        "outros_previstos": int(predicted_outros.sum()),
    }
