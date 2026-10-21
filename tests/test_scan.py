"""Testes do SCAN com dados sintéticos e configuração reduzida (CPU)."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from training.neighbors import cosine_neighbors
from training.scan import (
    ScanConfig,
    ScanDivergence,
    _check_sanity,
    evaluation_loss,
    mine_neighbors,
    scan_loss,
)


def _blobs(n_per: int = 40, n_views: int = 3, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Dois grupos bem separados; as vistas são a foto com um pouco de ruído."""
    rng = np.random.default_rng(seed)
    centers = np.zeros((2, 8))
    centers[0, 0], centers[1, 1] = 5.0, 5.0
    labels = np.repeat([0, 1], n_per)
    original = centers[labels] + rng.normal(0, 0.3, (2 * n_per, 8))
    views = np.stack([original] + [original + rng.normal(0, 0.1, original.shape)
                                   for _ in range(n_views)], axis=1)
    return views.astype(np.float32), labels


def test_mine_neighbors_matches_cosine_and_excludes_self() -> None:
    views, _ = _blobs()
    nn = mine_neighbors(views[:, 0], k=5, chunk=16)   # vários blocos
    assert not (nn == np.arange(len(nn))[:, None]).any()
    expected = cosine_neighbors(views[:, 0], 5)
    assert all(set(a) == set(b) for a, b in zip(nn, expected))


def test_scan_loss_rewards_agreement_and_spread() -> None:
    logits = torch.eye(4) * 50                       # 4 fotos, cada uma num cluster
    total, consistency, entropy = scan_loss(logits, logits, entropy_weight=5.0)
    assert consistency.item() == pytest.approx(0, abs=1e-4)
    assert entropy.item() == pytest.approx(np.log(4), abs=1e-4)
    disagree = scan_loss(logits, logits.roll(1, dims=0), 5.0)[1]
    assert disagree.item() > 5                       # vizinho em outro cluster custa caro


def test_evaluation_loss_prefers_consistent_clustering() -> None:
    views, labels = _blobs()
    nbrs = torch.as_tensor(mine_neighbors(views[:, 0], 5))
    good = torch.as_tensor(np.eye(2)[labels] * 0.98 + 0.01, dtype=torch.float32)
    bad = good[torch.randperm(len(good), generator=torch.Generator().manual_seed(0))]
    assert evaluation_loss(good, nbrs) < evaluation_loss(bad, nbrs)


def test_sanity_rule() -> None:
    spread = torch.eye(3)
    _check_sanity([3.0, 2.0, 1.0], spread)           # passa
    _check_sanity([1.0, 3.0, 2.0], spread)           # oscila, mas não cresce sempre: passa
    with pytest.raises(ScanDivergence):
        _check_sanity([1.0, float("nan")], spread)
    with pytest.raises(ScanDivergence):
        _check_sanity([1.0, 2.0, 3.0], spread)       # crescimento contínuo
    with pytest.raises(ScanDivergence):
        _check_sanity([2.0, 1.0], torch.ones(3, 3))  # tudo num cluster
