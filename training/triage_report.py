"""Relatórios da triagem de cinco classes num limiar fixo (protocolo, 1.7 a 1.9; T4g.6).

As mesmas funções servem ao ensaio, nos logits fora da dobra da execução R3,
e à abertura única do teste 03/07, com a temperatura e o limiar congelados.
Entradas: probabilidades das 5 classes antes e depois da temperatura, o
limiar, e por foto a classe do rótulo, o lote, o grupo e a condição.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from meter_reader.contracts import PRODUCT_CLASSES
from training.calibration import ece
from training.holdout import proportion
from training.triage_eval import (DETERMINED, MAX_LOSS, REJECTABLE, cp_upper,
                                  rejection_destination, rejection_score)

MULTIPLOS = PRODUCT_CLASSES.index("multiplos")


def summary(rejected: np.ndarray, labels: np.ndarray, max_loss: float = MAX_LOSS) -> dict:
    """Perda (com o limite superior unilateral e a meta) e cobertura (com IC 95%)."""
    determined = np.isin(labels, DETERMINED)
    losses, n_det = int((rejected & determined).sum()), int(determined.sum())
    upper = float(cp_upper(np.array([losses]), n_det)[0])
    cov = proportion(int((rejected & ~determined).sum()), int((~determined).sum()))
    return {"perdas": losses, "determinados": n_det, "perda": losses / n_det,
            "perda_limite_superior": upper, "meta_cumprida": upper <= max_loss,
            "rejeitaveis": cov["n"], "cobertos": cov["k"], "cobertura": cov["valor"],
            "cobertura_ic_inf": cov["ic_inf"], "cobertura_ic_sup": cov["ic_sup"]}


def by_subset(rejected: np.ndarray, labels: np.ndarray, keys: np.ndarray) -> pd.DataFrame:
    """Perdas e cobertura por lote ou por condição (seção 1.8)."""
    determined = np.isin(labels, DETERMINED)
    rows = {}
    for key in sorted(set(keys)):
        d, r = (keys == key) & determined, (keys == key) & ~determined
        rows[key] = {"perdas": int((rejected & d).sum()), "determinados": int(d.sum()),
                     "cobertos": int((rejected & r).sum()), "rejeitaveis": int(r.sum()),
                     "cobertura": (rejected & r).sum() / r.sum() if r.sum() else np.nan}
    return pd.DataFrame(rows).T


def confusion(probs: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Matriz de confusão das classes do produto (linhas: rótulo; colunas: mais provável)."""
    names = list(PRODUCT_CLASSES)
    table = pd.crosstab(pd.Categorical([names[i] for i in labels], names),
                        pd.Categorical([names[i] for i in probs.argmax(1)], names),
                        rownames=["rótulo"], colnames=["prevista"], dropna=False)
    return table


def separation(probs: np.ndarray, rejected: np.ndarray, labels: np.ndarray,
               groups: np.ndarray) -> dict:
    """Destino trocado, múltiplos e separação entre duas leituras e múltiplos (E52)."""
    dest = rejection_destination(probs)
    hit = rejected & ~np.isin(labels, DETERMINED)
    two, mult = groups == "cic_duas_leituras", labels == MULTIPLOS
    return {
        "destino_trocado": int((dest[hit] != labels[hit]).sum()), "rejeitaveis_cobertos": int(hit.sum()),
        "multiplos": int(mult.sum()),
        # Mais provável múltiplos: o critério da E65 ("identificados")
        "multiplos_mais_provavel_multiplos": int((probs[mult].argmax(1) == MULTIPLOS).sum()),
        # Rejeitados (qualquer destino): os que não passam para o detector (gatilho da E52)
        "multiplos_rejeitados": int((rejected & mult).sum()),
        # Rejeitados com destino múltiplos: os que vão para a pasta "Rejeitadas / Múltiplos"
        "multiplos_rejeitados_como_multiplos": int((rejected & mult & (dest == MULTIPLOS)).sum()),
        "duas_leituras": int(two.sum()),
        "duas_leituras_mais_provavel_multiplos": int((probs[two].argmax(1) == MULTIPLOS).sum()),
        "duas_leituras_rejeitadas": int(rejected[two].sum()),
    }


def calibration_table(before: np.ndarray, after: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """ECE em 10 faixas de P(rejeitável) e de cada classe de rejeição, antes e depois de T."""
    determined = np.isin(labels, DETERMINED)
    rows = {}
    for tag, probs in (("antes", before), ("depois", after)):
        row = {"P(rejeitável)": ece(rejection_score(probs), ~determined)}
        for c in REJECTABLE:
            row[f"P({PRODUCT_CLASSES[c]})"] = ece(probs[:, c], labels == c)
        rows[tag] = row
    return pd.DataFrame(rows).T


def full_report(before: np.ndarray, after: np.ndarray, threshold: float, labels: np.ndarray,
                lots: np.ndarray, groups: np.ndarray, conditions: np.ndarray) -> dict:
    """Todos os relatórios da seção 1.8, num limiar fixo, sobre as probabilidades `after`."""
    rejected = rejection_score(after) >= threshold
    return {"resumo": summary(rejected, labels),
            "por_lote": by_subset(rejected, labels, lots),
            "por_condicao": by_subset(rejected, labels, conditions),
            "confusao": confusion(after, labels),
            "separacao": separation(after, rejected, labels, groups),
            "calibracao": calibration_table(before, after, labels)}
