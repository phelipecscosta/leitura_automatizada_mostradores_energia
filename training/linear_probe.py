"""Sonda linear sobre características congeladas: C0 e C1 (protocolo, 1.3 e 1.4).

Regressão logística multinomial (lbfgs, C = 1,0, até 2.000 iterações) sobre
as características padronizadas pelo treino de cada dobra. No treino, cada
foto entra com todas as vistas; na avaliação, só a original. Uma execução por
dobra (revisão R1): o lbfgs é determinístico.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

from training.group_dataset import GroupPhoto, fold_indices
from training.triage_eval import N_CLASSES

C_REG = 1.0
MAX_ITER = 2000
WEIGHTINGS = ("W-b", "W-d")


def class_weights(product_class: np.ndarray) -> np.ndarray:
    """Peso de cada classe do produto na W-d: N / (5 · n_classe)."""
    counts = np.bincount(product_class, minlength=N_CLASSES)
    if (counts == 0).any():
        raise ValueError("Classe do produto ausente no treino.")
    return len(product_class) / (N_CLASSES * counts)


@dataclass(frozen=True)
class ProbeFold:
    held_out_lot: str
    n_train_photos: int
    n_iter: int  # iterações do lbfgs; igual a MAX_ITER sugere falta de convergência


def fit_probe(x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None):
    """Treina com todas as vistas: x (fotos, vistas, dim). Devolve (modelo, média, desvio)."""
    n, views, dim = x.shape
    rows = x.reshape(n * views, dim).astype(np.float64)
    mean, std = rows.mean(axis=0), rows.std(axis=0)
    std[std < 1e-6] = 1.0  # dimensão constante: não amplifica ruído
    weights = None if sample_weight is None else np.repeat(sample_weight, views)
    model = LogisticRegression(C=C_REG, max_iter=MAX_ITER)  # lbfgs, multinomial
    model.fit((rows - mean) / std, np.repeat(y, views), sample_weight=weights)
    return model, mean, std


def cross_validate_probe(features: np.ndarray, photos: list[GroupPhoto], targets: np.ndarray,
                         output_class, weighting: str = "W-b"):
    """Logits fora da dobra das pareadas: (logits, índices das fotos, dobras).

    `targets`: índice da saída de cada foto (grupo de treino ou classe, na
    ablação "amplo"); `output_class`: classe do produto de cada saída.
    """
    if weighting not in WEIGHTINGS:
        raise ValueError(f"Ponderação desconhecida: {weighting}")
    output_class = np.asarray(output_class)
    product = np.array([p.product_class for p in photos])
    all_logits, all_index, folds = [], [], []

    for lot in sorted({p.lot for p in photos}):
        train, held = fold_indices(photos, lot)
        if len(np.unique(targets[train])) != len(output_class):
            raise ValueError("Alguma saída não tem fotos no treino desta dobra.")
        w_class = class_weights(product[train]) if weighting == "W-d" else None
        weights = None if w_class is None else w_class[product[train]]

        model, mean, std = fit_probe(features[train], targets[train], weights)
        logits = model.decision_function((features[held, 0].astype(np.float64) - mean) / std)
        if w_class is not None:
            logits = logits - np.log(w_class)[output_class]  # correção da E31, por saída
        all_logits.append(logits)
        all_index.append(held)
        folds.append(ProbeFold(lot, len(train), int(np.max(model.n_iter_))))

    return np.concatenate(all_logits), np.concatenate(all_index), folds
