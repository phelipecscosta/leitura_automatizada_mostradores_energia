"""Modelo final da triagem (ET3b, T3b.5).

As cabeças finais são uma única camada linear por saída, equivalente a:
média das sementes sobre a entrada padronizada, menos a correção dos pesos
de classe, dividida pela temperatura (protocolo do Lab02, seções 2.9 a 2.11).
Composição de funções lineares é linear: nada muda na inferência.
"""

from __future__ import annotations

import numpy as np
import torch

from training.cross_validation import LinearHeads


def fold_heads(heads: list[LinearHeads], mean: np.ndarray, std: np.ndarray,
               scene_log_w: np.ndarray, leg_log_pw: float,
               t_scene: float, t_leg: float) -> LinearHeads:
    """Incorpora sementes, padronização, correção e temperatura em LinearHeads."""
    mu = torch.as_tensor(mean, dtype=torch.float64)
    sigma = torch.as_tensor(std, dtype=torch.float64)

    def fold(weights, biases, log_offset, temperature):
        w = torch.stack(weights).double().mean(0)   # média das sementes
        b = torch.stack(biases).double().mean(0)
        w = w / sigma                               # padronização: W(x - mu)/sigma
        b = b - w @ mu
        b = b - torch.as_tensor(log_offset, dtype=torch.float64)  # correção dos pesos
        return w / temperature, b / temperature                   # temperatura

    ws, bs = fold([h.scene_head.weight for h in heads], [h.scene_head.bias for h in heads],
                  scene_log_w, t_scene)
    wl, bl = fold([h.legibility_head.weight for h in heads],
                  [h.legibility_head.bias for h in heads], leg_log_pw, t_leg)

    folded = LinearHeads(len(mean))
    with torch.no_grad():
        folded.scene_head.weight.copy_(ws.float())
        folded.scene_head.bias.copy_(bs.float())
        folded.legibility_head.weight.copy_(wl.float())
        folded.legibility_head.bias.copy_(bl.float())
    return folded
