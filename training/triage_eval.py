"""Avaliação comum dos candidatos da triagem de cinco classes (T4f.2).

Protocolo: exploracao/triagem_README.md, seções 1.6 e 1.7. Os candidatos
entregam logits (dos 12 grupos, ou das 5 classes na ablação "amplo"); este
módulo soma por classe do produto, calibra, pontua e escolhe o limiar.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from meter_reader.contracts import PRODUCT_CLASSES
from training.calibration import fit_temperature
from training.group_labels import TRAINING_GROUP_CLASS

N_CLASSES = len(PRODUCT_CLASSES)
DETERMINED = (PRODUCT_CLASSES.index("digital"), PRODUCT_CLASSES.index("ciclometrico"))
REJECTABLE = tuple(i for i in range(N_CLASSES) if i not in DETERMINED)
BROAD_CLASS = tuple(range(N_CLASSES))  # ablação "amplo": cada saída já é uma classe


def class_log_probs(logits: np.ndarray, temperature: float = 1.0,
                    output_class=TRAINING_GROUP_CLASS) -> np.ndarray:
    """log P(classe) (n, 5): softmax das saídas / T e soma por classe, em log."""
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - z.max(axis=1, keepdims=True)
    log_p = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    output_class = np.asarray(output_class)
    out = np.empty((len(z), N_CLASSES))
    for c in range(N_CLASSES):
        cols = log_p[:, output_class == c]
        top = cols.max(axis=1, keepdims=True)
        # logsumexp: log da soma das probabilidades dos grupos da classe
        out[:, c] = (top + np.log(np.exp(cols - top).sum(axis=1, keepdims=True)))[:, 0]
    return out


def class_nll(logits: np.ndarray, labels: np.ndarray, output_class=TRAINING_GROUP_CLASS) -> float:
    """Perda logarítmica média nas classes do produto, sem pesos (seção 1.6)."""
    log_p = class_log_probs(logits, 1.0, output_class)
    return float(-log_p[np.arange(len(labels)), labels].mean())


@dataclass(frozen=True)
class Calibrated:
    """Probabilidades calibradas de forma aninhada por lote (seção 1.6)."""

    probs: np.ndarray                 # (n, 5); cada lote com a T ajustada nos outros
    per_lot: dict[str, float]         # lote -> T usada nas predições dele
    final: float                      # T ajustada com todos os lotes (implantação)


def calibrate_nested(logits: np.ndarray, labels: np.ndarray, lots: np.ndarray,
                     output_class=TRAINING_GROUP_CLASS) -> Calibrated:
    """Recebe só o universo de avaliação (pareadas), com os logits fora da dobra."""
    def nll(z, y):
        return class_nll(z, y, output_class)

    probs = np.full((len(labels), N_CLASSES), np.nan)
    per_lot = {}
    for lot in sorted(set(lots)):
        fit, held = lots != lot, lots == lot
        t = fit_temperature(nll, logits[fit], labels[fit])
        probs[held] = np.exp(class_log_probs(logits[held], t, output_class))
        per_lot[str(lot)] = t
    return Calibrated(probs, per_lot, fit_temperature(nll, logits, labels))


def rejection_score(probs: np.ndarray) -> np.ndarray:
    """P(rejeitável) = P(indeterminado) + P(múltiplos) + P(outros) (seção 1.6, R-a)."""
    return probs[:, list(REJECTABLE)].sum(axis=1)


def rejection_destination(probs: np.ndarray) -> np.ndarray:
    """Classe de destino de uma rejeição: a maior das três rejeitáveis."""
    return np.asarray(REJECTABLE)[probs[:, list(REJECTABLE)].argmax(axis=1)]


# ---------------------------------------------------------------------
# Curva perda × cobertura, limiar da meta e bootstrap (seção 1.7)
# ---------------------------------------------------------------------

import pandas as pd  # noqa: E402
from scipy.stats import beta  # noqa: E402

MAX_LOSS = 0.005    # meta L2: perda <= 0,5% pelo limite superior (seção 1.7)
ALPHA = 0.05        # limite superior unilateral de 95%
N_BOOT = 2000       # reamostragens do bootstrap pareado
BOOT_SEED = 0


def cp_upper(k: np.ndarray, n: int, alpha: float = ALPHA) -> np.ndarray:
    """Limite superior unilateral de Clopper-Pearson, vetorizado (quantil da Beta)."""
    k = np.asarray(k)
    upper = beta.ppf(1 - alpha, k + 1, np.maximum(n - k, 1))
    return np.where(k >= n, 1.0, upper)


def loss_coverage_curve(score: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Uma linha por limiar (cada valor distinto do escore), do maior ao menor.

    Rejeita-se quando score >= limiar. Perda: determinados rejeitados ÷
    determinados. Cobertura: rejeitáveis rejeitados ÷ rejeitáveis.
    """
    determined = np.isin(labels, DETERMINED)
    order = np.argsort(-score, kind="stable")
    s = score[order]
    losses = np.cumsum(determined[order])
    covered = np.cumsum(~determined[order])
    # Último índice de cada valor distinto: os empates são rejeitados juntos
    last = np.flatnonzero(np.append(s[1:] != s[:-1], True))
    n_det, n_rej = int(determined.sum()), int((~determined).sum())
    return pd.DataFrame({
        "threshold": s[last],
        "losses": losses[last],
        "loss": losses[last] / n_det,
        "loss_upper": cp_upper(losses[last], n_det),
        "coverage": covered[last] / n_rej,
    })


def choose_threshold(curve: pd.DataFrame, max_loss: float = MAX_LOSS):
    """Linha de maior cobertura que cumpre a meta; None se nenhuma cumprir.

    Em empate de cobertura, fica o maior limiar (menos perdas).
    """
    feasible = curve[curve["loss_upper"] <= max_loss]
    if feasible.empty:
        return None
    return feasible.loc[feasible["coverage"].idxmax()]  # idxmax: primeira ocorrência


def coverage(score: np.ndarray, threshold: float, labels: np.ndarray) -> float:
    """Cobertura da rejeição num limiar fixo."""
    rejectable = ~np.isin(labels, DETERMINED)
    return float((score[rejectable] >= threshold).mean())


def paired_coverage_diff(score_a, threshold_a, score_b, threshold_b, labels,
                         n_boot: int = N_BOOT, seed: int = BOOT_SEED):
    """Cobertura(a) − cobertura(b), com IC 95% por bootstrap pareado sobre as fotos.

    Os limiares ficam fixos; as mesmas fotos são sorteadas para os dois.
    """
    rejectable = ~np.isin(labels, DETERMINED)
    hit_a = (score_a >= threshold_a) & rejectable
    hit_b = (score_b >= threshold_b) & rejectable
    idx = np.random.default_rng(seed).integers(0, len(labels), (n_boot, len(labels)))
    n_rej = rejectable[idx].sum(axis=1)
    diffs = (hit_a[idx].sum(axis=1) - hit_b[idx].sum(axis=1)) / n_rej
    point = (hit_a.sum() - hit_b.sum()) / rejectable.sum()
    low, high = np.percentile(diffs, [2.5, 97.5])
    return float(point), float(low), float(high)

# ---------------------------------------------------------------------
# Logits fora da dobra gravados (revisão R3; rastreabilidade)
# ---------------------------------------------------------------------

import json  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

from training.evaluation import _product_commit  # noqa: E402

OOF_FOLDER = Path("experimentos") / "triagem5"


def save_oof(work_dir: Path, name: str, codes: list[str], lots: list[str],
             logits_per_seed: np.ndarray, config: dict, labels_sha256: str) -> Path:
    """Grava (sementes, n, saídas) em .npz e os metadados em .json; nunca sobrescreve."""
    logits_per_seed = np.asarray(logits_per_seed, dtype=np.float32)
    if logits_per_seed.ndim != 3 or logits_per_seed.shape[1] != len(codes):
        raise ValueError("Logits devem ter a forma (sementes, fotos, saídas).")
    if np.isnan(logits_per_seed).any():
        raise ValueError("Há fotos sem logit fora da dobra.")

    folder = work_dir / OOF_FOLDER
    folder.mkdir(parents=True, exist_ok=True)
    base = folder / name
    if base.with_suffix(".npz").exists():
        raise FileExistsError("Resultado já gravado; use outro nome para uma nova execução.")

    np.savez(base.with_suffix(".npz"), codes=np.array(codes), lots=np.array(lots),
             logits=logits_per_seed)
    record = {
        "nome": name,
        "data": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "commit_produto": _product_commit(),
        "sha256_rotulos": labels_sha256,
        "sementes": int(logits_per_seed.shape[0]),
        "saidas": int(logits_per_seed.shape[2]),
        "configuracao": config,
    }
    base.with_suffix(".json").write_text(json.dumps(record, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    return base.with_suffix(".npz")


def load_oof(work_dir: Path, name: str, codes: list[str]) -> np.ndarray:
    """Logits combinados pela média das sementes (E31), na ordem de `codes`."""
    data = np.load(work_dir / OOF_FOLDER / f"{name}.npz")
    if not np.array_equal(data["codes"], np.array(codes)):
        raise ValueError("Os pseudônimos gravados não correspondem à lista de fotos.")
    return data["logits"].mean(axis=0)
