"""Margem do recorte do leitor (T6a.2).

Mede quanto a caixa prevista precisa crescer para conter a caixa anotada, nas
predições fora da dobra do desenvolvimento. Os números do teste 03/07 não são
usados para ajustar nada (E68, E88).
"""

from __future__ import annotations

import numpy as np


def expansion_needed(pred, gt) -> np.ndarray:
    """Fração da largura (esquerda, direita) e da altura (topo, base) da caixa
    prevista que falta para ela conter a anotada; 0 nos lados já cobertos."""
    px0, py0, px1, py1 = (float(v) for v in pred)
    gx0, gy0, gx1, gy1 = (float(v) for v in gt)
    w, h = px1 - px0, py1 - py0
    if w <= 0 or h <= 0:
        raise ValueError("Caixa prevista sem área.")
    need = np.array([(px0 - gx0) / w, (gx1 - px1) / w, (py0 - gy0) / h, (gy1 - py1) / h])
    return np.clip(need, 0.0, None)


def weighted_quantile(values, weights, q: float) -> float:
    """Quantil ponderado: o menor valor cuja fração acumulada dos pesos alcança q."""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if len(v) == 0:
        return float("nan")
    order = np.argsort(v, kind="stable")
    v, w = v[order], w[order]
    cum = np.cumsum(w) / w.sum()
    idx = min(int(np.searchsorted(cum, q, side="left")), len(v) - 1)
    return float(v[idx])
