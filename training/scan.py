"""SCAN sobre características congeladas (protocolo de ML, seções 1.3 e 1.4).

Implementação própria, a partir do artigo (Van Gansbeke et al., 2020), com os
hiperparâmetros da configuração oficial para espinha congelada e só a cabeça
linear treinada. O código oficial não é copiado (licença CC BY-NC 4.0).

Fases: (1) vizinhos por cosseno na foto original; (2) agrupamento com a perda
SCAN, em várias cabeças, ficando a de menor perda de avaliação; (3)
auto-rotulagem com as fotos de confiança acima do limiar. As vistas
aumentadas vêm do cache (posições 1 a V); a posição 0 é a foto original.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

EPS = 1e-8


@dataclass(frozen=True)
class ScanConfig:
    """Valores da seção 1.4 do protocolo (configuração oficial ImageNet-50)."""

    n_clusters: int = 50
    n_neighbors: int = 50
    entropy_weight: float = 5.0
    n_heads: int = 10
    lr: float = 5.0
    momentum: float = 0.9
    epochs: int = 100
    batch_size: int = 512
    # Auto-rotulagem
    threshold: float = 0.99
    sl_lr: float = 0.03
    sl_epochs: int = 25
    ema_alpha: float = 0.999


class ScanDivergence(RuntimeError):
    """A fase não passou na verificação de sanidade (perda inválida ou colapso)."""


def mine_neighbors(features: np.ndarray, k: int, device: str = "cpu",
                   chunk: int = 2048) -> np.ndarray:
    """Índices (n, k) dos k vizinhos de maior cosseno, sem a própria foto."""
    x = F.normalize(torch.as_tensor(features, dtype=torch.float32, device=device), dim=1)
    out = []
    for start in range(0, len(x), chunk):
        similarity = x[start:start + chunk] @ x.T
        rows = torch.arange(similarity.shape[0], device=device)
        similarity[rows, rows + start] = -float("inf")  # a foto não é vizinha de si mesma
        out.append(similarity.topk(k, dim=1).indices.cpu())
    return torch.cat(out).numpy()


def scan_loss(anchor_logits: torch.Tensor, neighbor_logits: torch.Tensor,
              entropy_weight: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(total, consistência, entropia): foto e vizinho no mesmo cluster, clusters usados."""
    p_anchor = anchor_logits.softmax(dim=1)
    p_neighbor = neighbor_logits.softmax(dim=1)
    similarity = (p_anchor * p_neighbor).sum(dim=1)
    consistency = -torch.log(similarity.clamp_min(EPS)).mean()   # BCE contra 1
    mean_p = p_anchor.mean(dim=0).clamp_min(EPS)
    entropy = -(mean_p * mean_p.log()).sum()
    return consistency - entropy_weight * entropy, consistency, entropy


def evaluation_loss(probs: torch.Tensor, neighbors: torch.Tensor) -> float:
    """Perda de avaliação de uma cabeça, sem aumentação, como no artigo:
    consistência sobre todos os pares foto-vizinho menos a entropia (peso 1)."""
    similarity = (probs.unsqueeze(1) * probs[neighbors]).sum(dim=2)
    consistency = -torch.log(similarity.clamp_min(EPS)).mean()
    mean_p = probs.mean(dim=0).clamp_min(EPS)
    return float(consistency + (mean_p * mean_p.log()).sum())


def _check_sanity(history: list[float], probs: torch.Tensor) -> None:
    """Regra de sanidade sem rótulos (seção 1.4): perda finita, sem crescimento
    contínuo (alta em todas as épocas) e sem colapso num único cluster."""
    if not np.all(np.isfinite(history)):
        raise ScanDivergence("Perda inválida (NaN ou infinita).")
    if len(history) > 1 and np.all(np.diff(history) > 0):
        raise ScanDivergence("Perda em crescimento contínuo.")
    if len(torch.unique(probs.argmax(dim=1))) == 1:
        raise ScanDivergence("Todas as fotos num único cluster.")
