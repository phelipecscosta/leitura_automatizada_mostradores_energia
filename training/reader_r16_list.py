"""Lista da população-ouro (revisão R16, item b; ET6c).

Os aprováveis digitais sem julgamento humano, lidos da tabela gravada pela medida de
custo (`reader_gold_cost`), numa ordem sorteada com a semente em texto "r16-ouro".
Grava só ordem, pseudônimo e lote na pasta de trabalho e nunca sobrescreve uma lista
já gravada. Só contagens são exibidas.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from training.box_sample import shuffled

SEED = "r16-ouro"
LIST_NAME = Path("experimentos") / "leitor" / "lista_r16_ouro.csv"


def to_judge(table: pd.DataFrame, seed: str = SEED) -> pd.DataFrame:
    """Aprováveis sem julgamento humano, na ordem fixa do sorteio."""
    pending = table[table["alta"].astype(bool) & (table["rotulo"] != "humano")]
    lots = dict(zip(pending["pseudonimo"], pending["lote"]))
    codes = shuffled(lots, seed)
    return pd.DataFrame({"ordem": range(1, len(codes) + 1), "pseudonimo": codes,
                         "lote": [lots[c] for c in codes]})


def main() -> None:
    from meter_reader.config import get_work_dir
    from training.build_manifest import file_sha256
    from training.reader_gold_cost import TABLE_NAME

    work_dir = get_work_dir()
    out_path = work_dir / LIST_NAME
    if out_path.exists():
        raise SystemExit("A lista da R16 já existe e não é sobrescrita.")
    table = pd.read_csv(work_dir / TABLE_NAME, dtype={"pseudonimo": str, "lote": str})
    out = to_judge(table)
    out.to_csv(out_path, index=False, encoding="utf-8", lineterminator="\n")
    print(f"A julgar: {len(out)}; por lote: "
          f"{out['lote'].value_counts().sort_index().to_dict()}")
    print("SHA-256 da tabela de aprováveis:", file_sha256(work_dir / TABLE_NAME))
    print("SHA-256 da lista:", file_sha256(out_path))


if __name__ == "__main__":
    main()
