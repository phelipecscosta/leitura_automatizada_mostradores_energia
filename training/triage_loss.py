"""Perda da triagem (protocolo do Lab02, seção 2.4).

- Cena: entropia cruzada sobre logits com peso por classe; o indeterminado
  usa a perda parcial -log(p_digital + p_ciclométrico).
- Legibilidade: BCE sobre logits, mascarada nas fotos de "outros", com
  pos_weight = negativos/positivos.
- Total: cena + legibilidade (lambda = 1).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from meter_reader.triage import SCENE_CLASSES
from training.labels import MASKED, SCENE_METER_ONLY

N_SCENE = len(SCENE_CLASSES)
METER_SLOTS = [SCENE_CLASSES.index("digital"), SCENE_CLASSES.index("ciclometrico")]
PARTIAL_INDEX = N_SCENE  # posição do peso do indeterminado no vetor de pesos


def scene_weights(targets: torch.Tensor) -> torch.Tensor:
    """Pesos da cena: N/(3·n_c) por classe e, na última posição, o do indeterminado.

    O indeterminado recebe o peso médio de um medidor de tipo conhecido.
    Classe ausente no treino recebe peso zero.
    """
    exact = targets[targets != SCENE_METER_ONLY]
    counts = torch.bincount(exact, minlength=N_SCENE).float()
    weights = torch.where(counts > 0, counts.sum() / (N_SCENE * counts.clamp(min=1)),
                          torch.zeros_like(counts))
    meter_counts = counts[METER_SLOTS]
    partial = (weights[METER_SLOTS] * meter_counts).sum() / meter_counts.sum().clamp(min=1)
    return torch.cat([weights, partial.view(1)])


def scene_loss(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Entropia cruzada ponderada, com rótulo parcial no indeterminado."""
    log_p = F.log_softmax(logits, dim=1)  # a perda recebe logits; o softmax é feito aqui
    exact = targets != SCENE_METER_ONLY
    # clamp: o índice -1 do indeterminado não pode ser usado no gather
    exact_nll = -log_p.gather(1, targets.clamp(min=0).unsqueeze(1)).squeeze(1)
    partial_nll = -torch.logsumexp(log_p[:, METER_SLOTS], dim=1)  # -log(p_d + p_c)
    per_example = torch.where(exact, exact_nll, partial_nll)

    index = torch.where(exact, targets, torch.full_like(targets, PARTIAL_INDEX))
    w = weights.to(logits.device)[index]
    return (w * per_example).sum() / w.sum()  # média ponderada, como o weight do PyTorch


def legibility_pos_weight(targets: torch.Tensor) -> torch.Tensor:
    """Negativos/positivos entre as fotos com legibilidade (positivo = ilegível)."""
    valid = targets[targets != MASKED]
    return (valid == 0).sum().float() / (valid == 1).sum().clamp(min=1).float()


def legibility_loss(logits: torch.Tensor, targets: torch.Tensor,
                    pos_weight: torch.Tensor) -> torch.Tensor:
    """BCE sobre logits, só nas fotos de medidor."""
    mask = targets != MASKED
    if not mask.any():
        return logits.sum() * 0.0  # mantém o grafo, sem contribuir
    return F.binary_cross_entropy_with_logits(
        logits[mask], targets[mask].float(), pos_weight=pos_weight.to(logits.device)
    )


def triage_loss(scene_logits, legibility_logits, scene_targets, illegible_targets,
                weights, pos_weight):
    """Perda total e as duas parcelas: (total, cena, legibilidade)."""
    scene = scene_loss(scene_logits, scene_targets, weights)
    legibility = legibility_loss(legibility_logits, illegible_targets, pos_weight)
    return scene + legibility, scene, legibility
