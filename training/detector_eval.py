"""Avaliação comum do detector (ET5b; protocolo em exploracao/detector_README.md).

Nesta versão (T5b.2, D3 e D4): fotos com campo, tamanho efetivo de Kish,
limite superior de Clopper-Pearson ponderado (Korn e Graubard, 1998) e
orçamento de perdas da segunda porta. As métricas de localização entram na T5b.5.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.stats import beta

from training.box_labels import Box
from training.box_stats import TINY_SIDE
from training.triage_eval import ALPHA

GATE_MAX_LOSS = 0.005        # D3 b 1: perda da segunda porta <= 0,5% pelo limite superior de 95%
MIN_FIELD_SIDE = TINY_SIDE   # D3 a ii: campo com os dois lados >= 4 px (medido na T5b.1)


def photos_with_field(boxes: Iterable[Box], min_side: float = MIN_FIELD_SIDE) -> set[str]:
    """Pseudônimos com ao menos uma caixa (de qualquer classe) com os dois lados >= min_side."""
    return {b.pseudonym for b in boxes if min(b.x1 - b.x0, b.y1 - b.y0) >= min_side}


def kish_n(weights: np.ndarray) -> float:
    """Tamanho efetivo de Kish: (soma dos pesos)² ÷ soma dos quadrados dos pesos."""
    w = np.asarray(weights, dtype=float)
    return float(w.sum() ** 2 / (w ** 2).sum()) if len(w) else 0.0


def _upper(k: float, n: float, alpha: float) -> float:
    """Clopper-Pearson unilateral com k e n reais (quantil da Beta)."""
    return 1.0 if k >= n else float(beta.ppf(1 - alpha, k + 1, n - k))


def weighted_cp_upper(lost: np.ndarray, weights: np.ndarray, alpha: float = ALPHA) -> float:
    """Limite superior da taxa ponderada: taxa × tamanho efetivo faz o papel das perdas (D4 a)."""
    lost, w = np.asarray(lost, dtype=bool), np.asarray(weights, dtype=float)
    n_eff = kish_n(w)
    rate = float((w * lost).sum() / w.sum())
    return _upper(rate * n_eff, n_eff, alpha)


def loss_budget(n_eff: float, max_loss: float = GATE_MAX_LOSS, alpha: float = ALPHA) -> int:
    """Maior número de perdas (em fotos efetivas) cujo limite superior cumpre a meta;
    -1 se nem zero perdas cumpre."""
    k = 0
    while _upper(k, n_eff, alpha) <= max_loss:
        k += 1
    return k - 1


def main() -> None:
    # Importações que leem os dados só aqui: os testes usam as funções acima sem dados reais
    from meter_reader.config import get_work_dir
    from training.box_labels import read_labels
    from training.box_report import SAMPLE_NAME, SAMPLE_SHA256
    from training.build_manifest import file_sha256

    work_dir = get_work_dir()
    if file_sha256(work_dir / SAMPLE_NAME) != SAMPLE_SHA256:
        raise SystemExit("A amostra difere da gravada na T5.2.")
    sample = pd.read_csv(work_dir / SAMPLE_NAME, float_precision="round_trip")
    _, boxes = read_labels(work_dir, sample=set(sample["pseudonimo"]))
    with_field = photos_with_field(boxes)

    # Só o gabarito (caixas e pesos); nenhuma predição é lida (D6)
    rows = []
    for subset, part in sample.groupby("conjunto"):
        has = part["pseudonimo"].isin(with_field).to_numpy()
        w = part["peso"].to_numpy(dtype=float)
        n_eff = kish_n(w[has])
        rows.append({"conjunto": subset, "fotos": len(part), "com_campo": int(has.sum()),
                     "sem_campo": int((~has).sum()), "n_efetivo_com_campo": round(n_eff, 1),
                     "n_efetivo_sem_campo": round(kish_n(w[~has]), 1),
                     "orcamento_de_perdas": loss_budget(n_eff)})
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).set_index("conjunto"))


if __name__ == "__main__":
    main()
