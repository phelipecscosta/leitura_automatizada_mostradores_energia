"""Pureza dos vizinhos mais próximos por extrator (H2 do protocolo de ML).

Protocolo de ML, seções 1.4 e 1.7: vizinhos por similaridade de cosseno,
buscados entre as próprias fotos avaliadas, sem a foto; pureza balanceada
(média, entre as classes, da pureza média das fotos de cada classe);
bootstrap pareado e estratificado por classe sobre as fotos.
"""

from __future__ import annotations

import numpy as np


def standardize(features: np.ndarray) -> np.ndarray:
    """Escore z por coluna; coluna constante vira 0 (só a baseline usa)."""
    std = features.std(axis=0)
    return (features - features.mean(axis=0)) / np.where(std > 0, std, 1.0)


def cosine_neighbors(features: np.ndarray, k: int) -> np.ndarray:
    """Índices (n, k) dos k vizinhos de maior cosseno, sem a própria foto."""
    unit = features / np.linalg.norm(features, axis=1, keepdims=True)
    similarity = unit @ unit.T
    np.fill_diagonal(similarity, -np.inf)  # a foto nunca é vizinha de si mesma
    # Ordenação estável: empates resolvidos pelo índice, sempre do mesmo jeito
    return np.argsort(-similarity, axis=1, kind="stable")[:, :k]


def neighbor_purity(neighbors: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Fração dos vizinhos de cada foto com a mesma classe que ela."""
    return (labels[neighbors] == labels[:, None]).mean(axis=1)


def balanced_purity(purity: np.ndarray, labels: np.ndarray) -> float:
    """Média, entre as classes, da pureza média das fotos de cada classe."""
    return float(np.mean([purity[labels == c].mean() for c in np.unique(labels)]))


def paired_bootstrap_purity(purity_a: np.ndarray, purity_b: np.ndarray, labels: np.ndarray,
                            n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Diferença (A - B) da pureza balanceada: (pontual, IC 2,5%, IC 97,5%).

    Os vizinhos ficam fixos; reamostram-se as fotos, dentro de cada classe,
    para que todas as reamostragens tenham as mesmas contagens por classe.
    """
    point = balanced_purity(purity_a, labels) - balanced_purity(purity_b, labels)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(labels == c) for c in np.unique(labels)]
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        diffs[b] = (balanced_purity(purity_a[idx], labels[idx])
                    - balanced_purity(purity_b[idx], labels[idx]))
    low, high = np.percentile(diffs, [2.5, 97.5])
    return float(point), float(low), float(high)
