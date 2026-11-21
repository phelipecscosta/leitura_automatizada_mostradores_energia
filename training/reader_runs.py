"""Resumo dos treinos do leitor gravados na pasta de trabalho (T6b.5).

Lê os `meta.json` de cada treino e mostra, por configuração: os treinos concluídos,
a época escolhida, a M-a interna, o tempo e o alarme de subtreino das revisões R3 e
R6 (mais da metade das épocas escolhidas entre 216 e 240). Só números.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from training.reader_train import MAX_EPOCHS, META_NAME, RUNS_DIR

ALARM_FROM = 216   # R6: épocas de 216 a 240


def read_runs(root: Path) -> pd.DataFrame:
    """Uma linha por treino concluído (com `meta.json`)."""
    rows = []
    for meta_path in sorted(root.glob(f"*/*/{META_NAME}")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        rows.append({"configuracao": meta["configuracao"], "lote": meta["lote_avaliado"],
                     "semente": meta["semente"], "epoca": meta["epoca_escolhida"],
                     "epocas_max": meta["epocas_max"], "ma_interna": meta["ma_interna"],
                     "minutos": round(meta["segundos_total"] / 60, 1)})
    return pd.DataFrame(rows)


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    """Por configuração: treinos, M-a interna (média e faixa), épocas e o alarme."""
    g = runs.groupby("configuracao")
    out = pd.DataFrame({
        "treinos": g.size(),
        "ma_media": g["ma_interna"].mean().round(3),
        "ma_min": g["ma_interna"].min(), "ma_max": g["ma_interna"].max(),
        "epoca_mediana": g["epoca"].median(),
        "epocas_no_fim": g["epoca"].apply(lambda e: int((e >= ALARM_FROM).sum())),
        "minutos_medio": g["minutos"].mean().round(1),
    })
    out["alarme_subtreino"] = out["epocas_no_fim"] > out["treinos"] / 2
    return out


def main() -> None:
    from meter_reader.config import get_work_dir

    runs = read_runs(get_work_dir() / RUNS_DIR)
    if runs.empty:
        raise SystemExit("Nenhum treino concluído encontrado.")
    with pd.option_context("display.width", 140):
        print(runs.sort_values(["configuracao", "semente", "lote"]).to_string(index=False))
        print(f"\nResumo (épocas máximas: {MAX_EPOCHS}; alarme a partir da {ALARM_FROM}):")
        print(summarize(runs).to_string())


if __name__ == "__main__":
    main()
