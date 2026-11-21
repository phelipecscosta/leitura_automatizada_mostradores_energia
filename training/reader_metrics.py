"""Limiar alto, M-c e M-a do leitor (protocolo do leitor, seções 1.7 e 1.8; R2-b).

Por classe e por semente: o limiar alto é o menor que mantém o limite superior
unilateral de 95% (Clopper-Pearson) da taxa de confirmação dos impostores
confundíveis em no máximo 5% (Y-b). Um impostor é confirmado quando a leitura livre
é igual a ele e a probabilidade dele é de pelo menos o limiar. A cobertura de uma
classe é a fração dos alvos com a leitura livre igual ao rótulo e a probabilidade do
rótulo de pelo menos o limiar. M-c = média simples das três classes; vale a média
das três sementes. Os limiares ficam em escala logarítmica. Só números.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import beta

CLASSES = ("display_digital", "display_ciclometrico", "numero_serie")
MAX_FALSE_CONFIRMATION = 0.05   # Y-b
ALPHA = 0.05                    # limite superior unilateral de 95%


def cp_upper(k: int, n: int, alpha: float = ALPHA) -> float:
    """Limite superior unilateral de Clopper-Pearson para k sucessos em n."""
    if n == 0:
        return 1.0
    return 1.0 if k >= n else float(beta.ppf(1 - alpha, k + 1, n - k))


def high_threshold(scores: pd.DataFrame) -> dict:
    """Limiar alto (log) de uma classe e semente, pelos impostores confundíveis."""
    n = int(scores["tem_imp_confundivel"].sum())
    read = scores.loc[scores["imp_confundivel_lido"].astype(bool), "logp_imp_confundivel"]
    read = np.sort(read.to_numpy(float))[::-1]           # mais prováveis primeiro
    k_max = 0
    while k_max < len(read) and cp_upper(k_max + 1, n) <= MAX_FALSE_CONFIRMATION:
        k_max += 1
    if k_max >= len(read):
        log_t = -np.inf                                   # nenhum corte necessário
    else:
        log_t = float(np.nextafter(read[k_max], np.inf))  # exclui o (k_max + 1)-ésimo
    confirmed = int((read >= log_t).sum())
    return {"log_limiar": log_t, "impostores": n, "confirmados": confirmed,
            "limite_superior": cp_upper(confirmed, n)}


def covered(scores: pd.DataFrame, log_t: float) -> np.ndarray:
    """Alvos confirmados no limiar: leitura livre igual ao rótulo e P(rótulo) >= t."""
    return (scores["acerto"].astype(bool) & (scores["logp_rotulo"] >= log_t)).to_numpy()


def uniform_rate(scores: pd.DataFrame, log_t: float) -> float:
    read = scores["imp_uniforme_lido"].astype(bool) & (scores["logp_imp_uniforme"] >= log_t)
    return float(read.mean())


def config_table(scores: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por semente e classe: limiar, impostores, cobertura e acerto."""
    rows = []
    for (seed, cls), part in scores.groupby(["semente", "classe"]):
        t = high_threshold(part)
        rows.append({"semente": seed, "classe": cls, **t,
                     "cobertura": float(covered(part, t["log_limiar"]).mean()),
                     "acerto": float(part["acerto"].astype(bool).mean()),
                     "uniformes": uniform_rate(part, t["log_limiar"]),
                     "alvos": len(part)})
    return pd.DataFrame(rows)


def summary(table: pd.DataFrame) -> dict:
    """M-c e M-a: média das classes em cada semente, depois média das sementes."""
    by_seed = table.groupby("semente")[["cobertura", "acerto"]].mean()
    return {"M-c": float(by_seed["cobertura"].mean()), "M-a": float(by_seed["acerto"].mean()),
            "M-c por semente": by_seed["cobertura"].round(4).tolist()}


def main() -> None:
    from meter_reader.config import get_work_dir
    from meter_reader.reader import CANDIDATES
    from training.reader_dataset import load_items
    from training.reader_eval import score_config

    work_dir = get_work_dir()
    items, _ = load_items()
    with pd.option_context("display.width", 160):
        for config in CANDIDATES:
            table = config_table(score_config(work_dir, config, items))
            shown = table.assign(limiar=np.exp(table["log_limiar"]).round(4))
            print(f"\n== {config} ==")
            print(shown[["semente", "classe", "alvos", "limiar", "impostores", "confirmados",
                         "limite_superior", "uniformes", "cobertura", "acerto"]]
                  .round(4).to_string(index=False))
            s = summary(table)
            print(f"M-c {s['M-c']:.4f} (por semente {s['M-c por semente']}); "
                  f"M-a {s['M-a']:.4f}")


if __name__ == "__main__":
    main()
    