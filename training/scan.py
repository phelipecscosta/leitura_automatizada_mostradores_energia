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
import math
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
    # Revisão de 20/10 (seção 1.10): vetores de comprimento 1 antes da cabeça,
    # na mesma geometria dos vizinhos e do K-means; False reproduz a original
    unit_norm: bool = True
    # Quase-colapso: menos desta fração dos clusters ocupada ao fim do agrupamento
    min_occupied_fraction: float = 0.5


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


def _check_sanity(history: list[float], probs: torch.Tensor, min_occupied: int = 2) -> None:
    """Regra de sanidade sem rótulos (seção 1.4): perda finita, sem crescimento
    contínuo (alta em todas as épocas) e pelo menos `min_occupied` clusters ocupados."""
    if not np.all(np.isfinite(history)):
        raise ScanDivergence("Perda inválida (NaN ou infinita).")
    if len(history) > 1 and np.all(np.diff(history) > 0):
        raise ScanDivergence("Perda em crescimento contínuo.")
    if len(torch.unique(probs.argmax(dim=1))) < min_occupied:
        raise ScanDivergence("Colapso: poucos clusters ocupados.")



class MultiHead(nn.Module):
    """Várias cabeças lineares independentes: saída (cabeças, lote, K)."""

    def __init__(self, dim: int, n_clusters: int, n_heads: int) -> None:
        super().__init__()
        self.heads = nn.ModuleList(nn.Linear(dim, n_clusters) for _ in range(n_heads))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([head(x) for head in self.heads])

def _features(views: np.ndarray, cfg: ScanConfig, device: str) -> torch.Tensor:
    """Tensor (n, 1 + V, d) das características, normalizadas se a configuração pedir."""
    feats = torch.as_tensor(views, dtype=torch.float32, device=device)
    return F.normalize(feats, dim=2) if cfg.unit_norm else feats


def _random_views(n: int, n_views: int, generator: torch.Generator) -> torch.Tensor:
    """Uma vista aumentada sorteada por foto (posições 1 a n_views)."""
    return torch.randint(1, n_views + 1, (n,), generator=generator)


def train_scan(views: np.ndarray, neighbors: np.ndarray, cfg: ScanConfig, seed: int,
               device: str = "cpu", lr: float | None = None) -> tuple[nn.Linear, dict]:
    """Fase de agrupamento. Devolve a cabeça escolhida e o diagnóstico."""
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    feats = _features(views, cfg, device)
    nbrs = torch.as_tensor(neighbors, device=device)
    n, n_aug = feats.shape[0], feats.shape[1] - 1

    model = MultiHead(feats.shape[2], cfg.n_clusters, cfg.n_heads).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr or cfg.lr, momentum=cfg.momentum)
    history = []
    for _ in range(cfg.epochs):
        epoch_losses = []
        for batch in torch.randperm(n, generator=generator).split(cfg.batch_size):
            pick = torch.randint(0, nbrs.shape[1], (len(batch),), generator=generator)
            partner = nbrs[batch.to(device), pick.to(device)]
            xa = feats[batch.to(device), _random_views(len(batch), n_aug, generator).to(device)]
            xn = feats[partner, _random_views(len(batch), n_aug, generator).to(device)]
            out_a, out_n = model(xa), model(xn)
            losses = torch.stack([scan_loss(a, b, cfg.entropy_weight)[0]
                                  for a, b in zip(out_a, out_n)])
            optimizer.zero_grad()
            losses.sum().backward()   # todas as cabeças no mesmo passo
            optimizer.step()
            epoch_losses.append(losses.mean().item())
        history.append(float(np.mean(epoch_losses)))

    with torch.no_grad():
        probs = model(feats[:, 0]).softmax(dim=2)       # foto original, todas as cabeças
        head_losses = [evaluation_loss(p, nbrs) for p in probs]
    best = int(np.argmin(head_losses))
    _check_sanity(history, probs[best],
                  min_occupied=max(2, math.ceil(cfg.min_occupied_fraction * cfg.n_clusters)))
    return copy.deepcopy(model.heads[best]), {
        "cabeca": best, "perdas_avaliacao": head_losses, "historico": history,
        "lr": lr or cfg.lr,
    }


def self_label(head: nn.Linear, views: np.ndarray, cfg: ScanConfig, seed: int,
               device: str = "cpu", lr: float | None = None) -> tuple[nn.Linear, dict]:
    """Fase de auto-rotulagem: confiança na foto original, perda numa vista aumentada."""
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    feats = _features(views, cfg, device)
    n, n_aug = feats.shape[0], feats.shape[1] - 1

    model = copy.deepcopy(head).to(device)
    shadow = copy.deepcopy(model)      # média móvel exponencial dos pesos
    optimizer = torch.optim.SGD(model.parameters(), lr=lr or cfg.sl_lr, momentum=cfg.momentum)
    history, confident = [], []
    for _ in range(cfg.sl_epochs):
        epoch_losses, epoch_confident = [], 0
        for batch in torch.randperm(n, generator=generator).split(cfg.batch_size):
            idx = batch.to(device)
            with torch.no_grad():
                max_p, target = model(feats[idx, 0]).softmax(dim=1).max(dim=1)
            mask = max_p > cfg.threshold
            if not mask.any():
                continue
            logits = model(feats[idx, _random_views(len(batch), n_aug, generator).to(device)])
            loss = F.cross_entropy(logits[mask], target[mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():   # a média passa a ser o modelo, como no artigo
                for p_shadow, p in zip(shadow.parameters(), model.parameters()):
                    p_shadow.mul_(cfg.ema_alpha).add_(p, alpha=1 - cfg.ema_alpha)
                    p.copy_(p_shadow)
            epoch_losses.append(loss.item())
            epoch_confident += int(mask.sum())
        history.append(float(np.mean(epoch_losses)) if epoch_losses else float("nan"))
        confident.append(epoch_confident / n)

    info = {"historico": history, "fracao_confiante": confident, "lr": lr or cfg.sl_lr}
    trained = [h for h in history if not np.isnan(h)]
    if not trained:   # nenhuma foto passou do limiar: a cabeça do agrupamento fica como está
        return copy.deepcopy(head), {**info, "sem_fotos_confiantes": True}
    with torch.no_grad():
        probs = model(feats[:, 0]).softmax(dim=1)
    _check_sanity(trained, probs)
    return model, info


def _with_lr_rule(phase, *args, start_lr: float, max_tries: int = 4, **kwargs):
    """Divide o lr por 10 enquanto a fase não passar na sanidade."""
    lr = start_lr
    for _ in range(max_tries):
        try:
            return phase(*args, lr=lr, **kwargs)
        except ScanDivergence:
            lr /= 10
    raise ScanDivergence("Sem lr válido depois das divisões previstas.")


def run_scan(views: np.ndarray, neighbors: np.ndarray, seed: int, device: str = "cpu",
             cfg: ScanConfig = ScanConfig()) -> tuple[np.ndarray, dict]:
    """Agrupamento e auto-rotulagem. Devolve P(cluster | foto original) (n, K) e o diagnóstico."""
    head, scan_info = _with_lr_rule(train_scan, views, neighbors, cfg, seed, device,
                                    start_lr=cfg.lr)
    head, sl_info = _with_lr_rule(self_label, head, views, cfg, seed, device,
                                  start_lr=cfg.sl_lr)
    with torch.no_grad():
        probs = head(_features(views, cfg, device)[:, 0]).softmax(dim=1).cpu().numpy()
    return probs, {"scan": scan_info, "autorrotulagem": sl_info}
