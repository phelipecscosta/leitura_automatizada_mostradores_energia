"""Calibração por temperatura (protocolo do Lab02, seção 2.11; E1).

A temperatura divide os logits por T > 0. A perda logarítmica é convexa no
inverso da temperatura (beta = 1/T), então uma busca por seção áurea num
intervalo fixo encontra o mínimo, sem dependência nova. As perdas são sem
pesos de classe: a calibração deve refletir as frequências reais.
"""

from __future__ import annotations

import numpy as np

from meter_reader.triage import SCENE_CLASSES
from training.labels import MASKED, SCENE_METER_ONLY

METER_SLOTS = [SCENE_CLASSES.index("digital"), SCENE_CLASSES.index("ciclometrico")]
BETA_RANGE = (0.01, 20.0)  # T entre 0,05 e 100
GOLDEN = (np.sqrt(5) - 1) / 2


def _log_softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def scene_nll(logits: np.ndarray, scene: np.ndarray) -> float:
    """Perda logarítmica média da cena, com a regra parcial do indeterminado."""
    logp = _log_softmax(np.asarray(logits, dtype=np.float64))
    exact = scene != SCENE_METER_ONLY
    lp_exact = logp[np.arange(len(scene)), np.clip(scene, 0, None)]
    # log(p_digital + p_ciclométrico), calculado sem sair do espaço logarítmico
    lp_meter = np.logaddexp(logp[:, METER_SLOTS[0]], logp[:, METER_SLOTS[1]])
    return float(-np.where(exact, lp_exact, lp_meter).mean())


def legibility_nll(logits: np.ndarray, illegible: np.ndarray) -> float:
    """Perda logarítmica média da legibilidade, só nos medidores."""
    meters = illegible != MASKED
    z = np.asarray(logits, dtype=np.float64)[meters]
    y = illegible[meters]
    # -log sigmoide(z) = log(1 + e^-z); -log(1 - sigmoide(z)) = log(1 + e^z)
    return float(np.where(y == 1, np.logaddexp(0, -z), np.logaddexp(0, z)).mean())


def _golden_min(f, low: float, high: float, tol: float = 1e-6) -> float:
    """Mínimo de uma função unimodal no intervalo [low, high]."""
    a, b = low, high
    c, d = b - GOLDEN * (b - a), a + GOLDEN * (b - a)
    fc, fd = f(c), f(d)
    while b - a > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - GOLDEN * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + GOLDEN * (b - a)
            fd = f(d)
    return (a + b) / 2


def fit_temperature(nll, logits: np.ndarray, labels: np.ndarray) -> float:
    """Temperatura T que minimiza nll(logits / T, rótulos)."""
    beta = _golden_min(lambda b: nll(np.asarray(logits) * b, labels), *BETA_RANGE)
    return 1.0 / beta
