"""Calibração por temperatura (protocolo do Lab02, seção 2.11; E1).

A temperatura divide os logits por T > 0. A perda logarítmica é convexa no
inverso da temperatura (beta = 1/T), então uma busca por seção áurea num
intervalo fixo encontra o mínimo, sem dependência nova. As perdas são sem
pesos de classe: a calibração deve refletir as frequências reais.
"""

from __future__ import annotations

import numpy as np
import torch

from training import triage_loss as tl
from meter_reader.triage import SCENE_CLASSES
from training.labels import MASKED, SCENE_METER_ONLY
from dataclasses import dataclass

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


def scene_probabilities(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Softmax dos logits de cena divididos pela temperatura."""
    return np.exp(_log_softmax(np.asarray(logits, dtype=np.float64) / temperature))


def legibility_probabilities(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Sigmoide dos logits de legibilidade divididos pela temperatura."""
    return 1 / (1 + np.exp(-np.asarray(logits, dtype=np.float64) / temperature))


def reliability_bins(prob: np.ndarray, label: np.ndarray, n_bins: int = 10):
    """Por faixa de largura igual: (confiança média, frequência observada, contagem).

    Faixas vazias recebem NaN na confiança e na frequência.
    """
    idx = np.clip((prob * n_bins).astype(int), 0, n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins)
    with np.errstate(invalid="ignore"):
        conf = np.bincount(idx, weights=prob, minlength=n_bins) / counts
        freq = np.bincount(idx, weights=label.astype(float), minlength=n_bins) / counts
    return conf, freq, counts


def ece(prob: np.ndarray, label: np.ndarray, n_bins: int = 10) -> float:
    """ECE binário: média das diferenças por faixa, ponderada pela contagem."""
    conf, freq, counts = reliability_bins(prob, label, n_bins)
    filled = counts > 0
    return float((counts[filled] * np.abs(conf[filled] - freq[filled])).sum() / counts.sum())


def ece_bootstrap(prob: np.ndarray, label: np.ndarray, n_boot: int = 2000,
                  seed: int = 0) -> tuple[float, float]:
    """IC de 95% do ECE por bootstrap sobre as fotos."""
    rng = np.random.default_rng(seed)
    values = [ece(prob[i], label[i]) for i in rng.integers(0, len(prob), (n_boot, len(prob)))]
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


@dataclass(frozen=True)
class NestedCalibration:
    """Resultado da calibração aninhada por lote (protocolo 2.11)."""
    p_scene: np.ndarray                     # (n, 3), cada lote com T ajustada nos outros
    p_illegible: np.ndarray                 # (n,), idem
    per_lot: dict[str, tuple[float, float]]  # lote -> (T da cena, T da legibilidade)
    final: tuple[float, float]              # ajustadas com todos os lotes (implantação)


def calibrate_nested(scene_logits: np.ndarray, illegible_logits: np.ndarray,
                     scene: np.ndarray, illegible: np.ndarray,
                     lots: np.ndarray) -> NestedCalibration:
    """Para cada lote, ajusta as temperaturas nos outros e aplica no lote deixado de fora."""
    p_scene = np.full(scene_logits.shape, np.nan)
    p_illegible = np.full(illegible_logits.shape, np.nan)
    per_lot = {}
    for lot in sorted(set(lots)):
        fit, held = lots != lot, lots == lot
        t_scene = fit_temperature(scene_nll, scene_logits[fit], scene[fit])
        t_leg = fit_temperature(legibility_nll, illegible_logits[fit], illegible[fit])
        p_scene[held] = scene_probabilities(scene_logits[held], t_scene)
        p_illegible[held] = legibility_probabilities(illegible_logits[held], t_leg)
        per_lot[str(lot)] = (t_scene, t_leg)
    final = (fit_temperature(scene_nll, scene_logits, scene),
             fit_temperature(legibility_nll, illegible_logits, illegible))
    return NestedCalibration(p_scene, p_illegible, per_lot, final)

def log_weights(scene: np.ndarray, illegible: np.ndarray) -> tuple[np.ndarray, float]:
    """Log dos pesos que o treino usa com estes rótulos: (cena (3,), legibilidade).

    A entropia cruzada ponderada aprende probabilidades proporcionais a
    w_c × p(c | foto); subtrair log(w_c) do logit desfaz o efeito
    (protocolo 2.11, revisão de 08/10). O peso do indeterminado não entra,
    porque não corresponde a um logit.
    """
    scene_t = torch.as_tensor(np.asarray(scene, dtype=np.int64))
    ill_t = torch.as_tensor(np.asarray(illegible, dtype=np.int64))
    w = tl.scene_weights(scene_t)[: len(SCENE_CLASSES)].double().numpy()
    if (w == 0).any():
        raise ValueError("Classe de cena ausente no treino: a correção não é definida.")
    return np.log(w), float(np.log(tl.legibility_pos_weight(ill_t).item()))


def fold_offsets(scene: np.ndarray, illegible: np.ndarray,
                 lots: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Correção por foto: o log dos pesos do treino da dobra em que ela foi predita.

    Vale para a validação por lote com todo o treino (train_fraction = 1):
    o treino da dobra de cada lote são os outros lotes. Uso: logit − correção.
    """
    scene_off = np.zeros((len(scene), len(SCENE_CLASSES)))
    ill_off = np.zeros(len(scene))
    for lot in sorted(set(lots)):
        held = lots == lot
        log_w, log_pw = log_weights(scene[~held], illegible[~held])
        scene_off[held] = log_w
        ill_off[held] = log_pw
    return scene_off, ill_off
