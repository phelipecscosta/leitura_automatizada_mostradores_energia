"""Amostra da revisão R15, item (c): 300 aceitos digitais para o julgamento às cegas
da função (Q-a, ET6c).

Sorteio com a semente em texto "r15-aceitos" entre os recortes digitais aceitos pelo
filtro do CSV (origem haiku_csv) nos rótulos vigentes (R9). Grava só pseudônimo, lote
e ordem na pasta de trabalho, e nunca sobrescreve uma amostra já gravada. Só
contagens são exibidas.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from training.box_sample import shuffled

SAMPLE_SIZE = 300
SEED = "r15-aceitos"
SAMPLE_NAME = Path("experimentos") / "leitor" / "amostra_r15_aceitos.csv"
DIGITAL = "display_digital"


def draw(labels: pd.DataFrame, lots: dict[str, str], n: int = SAMPLE_SIZE,
         seed: str = SEED) -> pd.DataFrame:
    """As n primeiras posições da ordem fixa dos aceitos digitais do professor."""
    pool = labels[(labels["classe"] == DIGITAL) & (labels["status"] == "aceito")
                  & (labels["origem"] == "haiku_csv")]
    codes = shuffled(pool["pseudonimo"].unique(), seed)
    if len(codes) < n:
        raise ValueError("Aceitos digitais insuficientes para a amostra.")
    codes = codes[:n]
    return pd.DataFrame({"ordem": range(1, n + 1), "pseudonimo": codes,
                         "lote": [lots[c] for c in codes]})


def main() -> None:
    from meter_reader.config import get_work_dir
    from training.build_manifest import file_sha256
    from training.group_dataset import LABEL_VERSIONS, load_group_photos
    from training.reader_dataset import R9_LABELS_SHA256
    from training.reader_labels_r9 import R9_LABELS_NAME

    work_dir = get_work_dir()
    out_path = work_dir / SAMPLE_NAME
    if out_path.exists():
        raise SystemExit("A amostra da R15 já existe e não é sobrescrita.")
    if file_sha256(work_dir / R9_LABELS_NAME) != R9_LABELS_SHA256:
        raise SystemExit("Os rótulos do leitor não são os da revisão R9 (SHA-256 diferente).")
    labels = pd.read_csv(work_dir / R9_LABELS_NAME, dtype=str, keep_default_na=False)
    lots = {p.pseudonym: p.lot for p in load_group_photos(LABEL_VERSIONS["r4"])}
    sample = draw(labels, lots)
    sample.to_csv(out_path, index=False, encoding="utf-8", lineterminator="\n")
    pool = int(((labels["classe"] == DIGITAL) & (labels["status"] == "aceito")
                & (labels["origem"] == "haiku_csv")).sum())
    print(f"Aceitos digitais do professor: {pool}; sorteados: {len(sample)}")
    print("Por lote: " + ", ".join(f"{k} {v}" for k, v in
                                   sample["lote"].value_counts().sort_index().items()))
    print("SHA-256 da amostra:", file_sha256(out_path))


if __name__ == "__main__":
    main()
