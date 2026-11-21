"""Revisão R9 do protocolo do leitor: nova versão dos rótulos, com os ciclométricos sem
alvo rotulados às cegas por humano.

O arquivo novo é o final da ET6a (que não muda) mais uma linha por recorte julgado,
com a origem "humano_r9" e o status da rotulagem: "humano" (alvo de treino),
"ilegivel", "obstruido", "girado" ou "nao_ciclometrico" (sem alvo). Só contagens são
exibidas.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from training.reader_labels import FINAL_COLUMNS, numeric_value

R9_JUDGMENT_NAME = Path("experimentos") / "leitor" / "rotulagem_r9_ciclometricos.csv"
R9_LABELS_NAME = "rotulos_leitor_r9.csv"
R9_ORIGIN = "humano_r9"
R9_STATUSES = ("humano", "ilegivel", "obstruido", "girado", "nao_ciclometrico")
TEXT_COLUMNS = ("texto", "valor", "funcao", "duas_leituras", "rotacao")


def r9_rows(judged: pd.DataFrame, crops: pd.DataFrame) -> pd.DataFrame:
    """Linhas do arquivo de rótulos para os recortes julgados na R9. `crops` traz as
    caixas (lista de recortes do desenvolvimento)."""
    if judged.duplicated(["pseudonimo", "classe"]).any():
        raise SystemExit("Recorte julgado duas vezes na rotulagem R9.")
    bad = set(judged["status"]) - set(R9_STATUSES)
    if bad:
        raise SystemExit(f"Status desconhecido na rotulagem R9: {sorted(bad)}")
    boxes = crops.set_index(["pseudonimo", "classe"])
    rows = []
    for r in judged.to_dict("records"):
        b = boxes.loc[(r["pseudonimo"], r["classe"])]
        target = r["status"] == "humano"
        rows.append({"pseudonimo": r["pseudonimo"], "classe": r["classe"],
                     **{c: b[c] for c in ("x0", "y0", "x1", "y1", "origem_caixa")},
                     "status": r["status"], "texto": r["texto"] if target else "",
                     "valor": numeric_value(r["texto"]) if target else "", "funcao": "",
                     "duas_leituras": "False",
                     "rotacao": "0" if target else "",   # lido como está (R7)
                     "origem": R9_ORIGIN})
    return pd.DataFrame(rows, columns=list(FINAL_COLUMNS))


def human_judged_keys(labels: pd.DataFrame, target_statuses) -> set[str]:
    """Chaves "pseudônimo|classe" dos recortes que um humano julgou sem alvo (F-2 e R9):
    não podem receber pseudo-rótulo (o julgamento humano prevalece)."""
    judged = labels["origem"].isin(["humano_f2", R9_ORIGIN]) & ~labels["status"].isin(
        target_statuses)
    return set(labels.loc[judged, "pseudonimo"] + "|" + labels.loc[judged, "classe"])


def main() -> None:
    from meter_reader.config import get_work_dir
    from training.build_manifest import file_sha256
    from training.reader_labels import CROPS_NAME, FINAL_LABELS_NAME, labels_name
    from training.teacher import MODEL
    from training.teacher_eval import reading_values

    work_dir = get_work_dir()
    dtypes = {c: str for c in TEXT_COLUMNS}
    final = pd.read_csv(work_dir / FINAL_LABELS_NAME, keep_default_na=False, dtype=dtypes,
                        float_precision="round_trip")
    judged = pd.read_csv(work_dir / R9_JUDGMENT_NAME, dtype=str, keep_default_na=False)
    crops = pd.read_csv(work_dir / CROPS_NAME, float_precision="round_trip")
    raw = pd.read_csv(work_dir / labels_name(MODEL), keep_default_na=False, dtype=dtypes)

    new = r9_rows(judged, crops)
    out = pd.concat([final, new], ignore_index=True)[list(FINAL_COLUMNS)]
    if out.duplicated(["pseudonimo", "classe"]).any():
        raise SystemExit("Recorte julgado na R9 já tinha linha no arquivo final da ET6a.")
    path = work_dir / R9_LABELS_NAME
    out.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")

    # Conferências (só contagens)
    status_raw = dict(zip(raw["pseudonimo"] + "|" + raw["classe"], raw["status"]))
    text_raw = dict(zip(raw["pseudonimo"] + "|" + raw["classe"], raw["texto"]))
    key = judged["pseudonimo"] + "|" + judged["classe"]
    print("R9: status humano x status do professor (Haiku):")
    print(pd.crosstab(judged["status"], key.map(status_raw).fillna("sem_resposta")).to_string())
    kept = judged[judged["decisao"] == "manteve"]
    agree = 0
    for r in kept.to_dict("records"):
        mine = reading_values(r["texto"]) or set()
        theirs = reading_values(text_raw.get(f"{r['pseudonimo']}|{r['classe']}", "") or "?") or set()
        agree += bool(mine & theirs)
    print(f"\nMantidos contra o CSV: {len(kept)}; com a mesma leitura do Haiku: {agree}")
    print("\nNovo arquivo, por classe, origem e status:")
    print(out.groupby(["classe", "origem", "status"]).size().to_string())
    print("\nSHA-256 do novo arquivo:", file_sha256(path))


if __name__ == "__main__":
    main()
