"""Métricas da abertura única do lote de teste (protocolo de ML, seção 1.9).

Triagem de DL: decisão de rejeição do Lab02 (seção 2.10), com o limiar
congelado, e revocação de medidor no colapso binário. Triagem de ML: revocação
de medidor e perda. Todas as proporções com IC bilateral de 95% de
Clopper-Pearson, porque o teste é pequeno e o resultado é lido uma única vez.
"""

from __future__ import annotations

import numpy as np
import torch

from meter_reader.triage import SCENE_CLASSES, rejection_score
from training.labels import MASKED
from training.risk_coverage import clopper_pearson_upper

OUTROS = SCENE_CLASSES.index("outros")


def clopper_pearson_interval(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """IC bilateral (1 - alpha) de Clopper-Pearson de k sucessos em n."""
    if n == 0:
        return float("nan"), float("nan")
    upper = clopper_pearson_upper(k, n, alpha / 2)
    lower = 1.0 - clopper_pearson_upper(n - k, n, alpha / 2)
    return lower, upper


def proportion(k: int, n: int) -> dict:
    """Contagem, proporção e IC de 95%."""
    low, high = clopper_pearson_interval(k, n)
    return {"k": int(k), "n": int(n), "valor": k / n if n else float("nan"),
            "ic_inf": low, "ic_sup": high}


def dl_decisions(scene_logits: np.ndarray, illegible_logits: np.ndarray,
                 threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """(rejeitada, destino "Outros") pela regra do Lab02, com as cabeças finais
    (temperatura e correção já incorporadas): rejeita se P(rejeitável) >= t;
    destino "Outros" se a parcela de "outros" for a maior."""
    p_outros = torch.as_tensor(scene_logits).softmax(dim=1)[:, OUTROS].numpy()
    p_illegible = torch.as_tensor(illegible_logits).sigmoid().numpy()
    rejected = rejection_score(p_outros, p_illegible) >= threshold
    to_outros = rejected & (p_outros >= (1 - p_outros) * p_illegible)
    return rejected, to_outros


def dl_report(rejected: np.ndarray, to_outros: np.ndarray, scene: np.ndarray,
              illegible: np.ndarray) -> dict:
    """Cobertura, risco e perda da rejeição (Lab02, 2.10) e revocação de medidor."""
    is_outros = scene == OUTROS
    meter = ~is_outros
    illegible_meter = meter & (illegible == 1)
    legible_meter = meter & (illegible == 0)
    rejectable = is_outros | illegible_meter
    assert not (meter & (illegible == MASKED)).any(), "Medidor sem legibilidade."
    return {
        "cobertura_rejeicao": proportion((rejected & rejectable).sum(), rejectable.sum()),
        "risco": proportion((rejected & legible_meter).sum(), rejected.sum()),
        "perda": proportion((rejected & legible_meter).sum(), legible_meter.sum()),
        "revocacao_medidor": proportion((meter & ~to_outros).sum(), meter.sum()),
        "destino_trocado": int((rejected & illegible_meter & to_outros).sum()),
    }


def ml_report(predicted_outros: np.ndarray, is_outros: np.ndarray) -> dict:
    """Revocação de medidor (colapso binário) e contagens da triagem de ML."""
    meter = is_outros == 0
    return {
        "revocacao_medidor": proportion((meter & ~predicted_outros).sum(), meter.sum()),
        "perda": proportion((meter & predicted_outros).sum(), meter.sum()),
        "outros_rejeitados": int((~meter & predicted_outros).sum()),
        "outros_no_teste": int((~meter).sum()),
    }
