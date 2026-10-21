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
    run_scan,
    scan_loss,
    self_label,
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
    with pytest.raises(ScanDivergence):               # quase-colapso: 2 de 4 exigidos
        _check_sanity([2.0, 1.0], torch.eye(4)[[0, 0, 1, 1]], min_occupied=3)



def test_run_scan_separates_blobs_and_is_deterministic() -> None:
    views, labels = _blobs()
    nbrs = mine_neighbors(views[:, 0], 5)
    cfg = replace(ScanConfig(), n_clusters=2, n_neighbors=5, n_heads=3, epochs=30,
                  batch_size=32, sl_epochs=5, entropy_weight=2.0)
    probs, info = run_scan(views, nbrs, seed=0, cfg=cfg)
    clusters = probs.argmax(axis=1)
    assert probs.shape == (80, 2)
    np.testing.assert_allclose(probs.sum(axis=1), 1, atol=1e-5)
    assert len(set(clusters[labels == 0])) == 1 and len(set(clusters[labels == 1])) == 1
    assert clusters[0] != clusters[-1]
    assert info["scan"]["cabeca"] in range(3)
    again, _ = run_scan(views, nbrs, seed=0, cfg=cfg)
    np.testing.assert_allclose(probs, again, atol=1e-6)


def test_self_label_without_confident_photos_keeps_the_head() -> None:
    views, _ = _blobs()
    head = torch.nn.Linear(8, 2)
    torch.nn.init.zeros_(head.weight)                 # saída uniforme: confiança 0,5
    torch.nn.init.zeros_(head.bias)
    kept, info = self_label(head, views, ScanConfig(), seed=0)
    assert info["sem_fotos_confiantes"]
    assert torch.equal(kept.weight, head.weight)



def test_unit_norm_makes_the_scan_scale_invariant() -> None:
    views, _ = _blobs()
    nbrs = mine_neighbors(views[:, 0], 5)
    cfg = replace(ScanConfig(), n_clusters=2, n_neighbors=5, n_heads=2, epochs=5,
                  batch_size=32, sl_epochs=2, entropy_weight=2.0)
    small, _ = run_scan(views, nbrs, seed=0, cfg=cfg)
    large, _ = run_scan(views * 50, nbrs, seed=0, cfg=cfg)   # norma como a do DINOv2
    np.testing.assert_allclose(small, large, atol=1e-5)
