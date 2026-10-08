"""Curva cobertura × risco da rejeição na triagem (protocolo do Lab02, seção 2.10; E2).

Rejeita-se a foto quando P(rejeitável) >= t. Para cada limiar candidato,
calculam-se a cobertura da rejeição, o risco, a perda e os limites
superiores unilaterais de Clopper-Pearson (95%) do risco e da perda.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) para X ~ Binomial(n, p), somada no espaço logarítmico."""
    if p <= 0:
        return 1.0
    if p >= 1:
        return 1.0 if k >= n else 0.0
    log_p, log_q = math.log(p), math.log1p(-p)
    total = sum(
        math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                 + i * log_p + (n - i) * log_q)
        for i in range(k + 1)
    )
    return min(1.0, total)


def clopper_pearson_upper(k: int, n: int, alpha: float = 0.05) -> float:
    """Limite superior unilateral (1 - alpha) de uma proporção com k sucessos em n."""
    if n == 0 or k >= n:
        return 1.0
    low, high = k / n, 1.0
    for _ in range(60):  # bisseção: erro menor que 1e-17
        mid = (low + high) / 2
        if _binom_cdf(k, n, mid) > alpha:
            low = mid
        else:
            high = mid
    return high


def rejection_curve(score: np.ndarray, rejectable: np.ndarray,
                    legible_meter: np.ndarray, alpha: float = 0.05) -> pd.DataFrame:
    """Uma linha por limiar candidato (cada valor distinto do escore), do maior ao menor."""
    score = np.asarray(score, dtype=float)
    n_rejectable, n_legible = int(rejectable.sum()), int(legible_meter.sum())
    rows = []
    for t in np.unique(score)[::-1]:
        rejected = score >= t
        n = int(rejected.sum())
        losses = int((rejected & legible_meter).sum())
        rows.append({
            "threshold": float(t),
            "rejected": n,
            "losses": losses,
            "coverage": int((rejected & rejectable).sum()) / n_rejectable,
            "risk": losses / n,
            "loss": losses / n_legible,
            "risk_upper": clopper_pearson_upper(losses, n, alpha),
            "loss_upper": clopper_pearson_upper(losses, n_legible, alpha),
        })
    return pd.DataFrame(rows)


def choose_threshold(curve: pd.DataFrame, max_risk: float, max_loss: float):
    """Linha de maior cobertura com os dois limites superiores dentro das metas; None se não houver."""
    feasible = curve[(curve["risk_upper"] <= max_risk) & (curve["loss_upper"] <= max_loss)]
    if feasible.empty:
        return None
    return feasible.loc[feasible["coverage"].idxmax()]


def min_rejections_without_error(max_risk: float, alpha: float = 0.05) -> int:
    """Menor número de rejeições, todas sem erro, que demonstra risco <= max_risk."""
    n = 1
    while clopper_pearson_upper(0, n, alpha) > max_risk:
        n += 1
    return n
