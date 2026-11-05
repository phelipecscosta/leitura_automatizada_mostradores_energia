"""Aplica as correções da revisão R3 aos rótulos da rotulagem completa (T4f.8).

Revisão R3 do protocolo da triagem (opção B): as fotos não são movidas; as
correções ficam em <pasta de trabalho>/revisao_r3/revisao_r3.csv, e uma nova
versão dos rótulos é gravada ao lado da original, que nunca é alterada.

Uso, a partir da raiz do produto:
    python -m training.label_corrections
"""

from __future__ import annotations

import csv

from meter_reader.config import get_work_dir
from training.build_group_labels import COLUMNS
from training.build_manifest import file_sha256
from training.group_dataset import DEV_LABELS, GroupLabelFileError
from training.group_labels import FOLDERS

R3_LABELS_NAME = "rotulos_grupos_r3.csv"
VALID_PAIRS = set(FOLDERS.values())


def apply_corrections(rows: list[dict], review: list[dict]) -> tuple[list[dict], int]:
    """Devolve (novas linhas, número de fotos alteradas). Mensagens só com contagens."""
    by_code = {r["pseudonimo"]: dict(r) for r in rows}
    changed = 0
    for line, item in enumerate(review, start=2):
        current = (item["grupo_atual"], item["condicao_atual"])
        final = (item["grupo_final"], item["condicao_final"])
        row = by_code.get(item["pseudonimo"])
        if row is None:
            raise GroupLabelFileError(f"Revisão, linha {line}: pseudônimo fora dos rótulos.")
        if (row["grupo"], row["condicao"]) != current:
            raise GroupLabelFileError(f"Revisão, linha {line}: rótulo atual diferente do arquivo.")
        if final not in VALID_PAIRS:
            raise GroupLabelFileError(f"Revisão, linha {line}: par grupo e condição inexistente.")
        if final != current:
            row["grupo"], row["condicao"] = final
            changed += 1
    return sorted(by_code.values(), key=lambda r: r["pseudonimo"]), changed


def main() -> None:
    work_dir = get_work_dir()
    name, sha256 = DEV_LABELS
    source = work_dir / name
    if file_sha256(source) != sha256:
        raise SystemExit("Rótulos originais com impressão digital diferente da esperada.")
    with source.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    with (work_dir / "revisao_r3" / "revisao_r3.csv").open(encoding="utf-8", newline="") as file:
        review = list(csv.DictReader(file))

    new_rows, changed = apply_corrections(rows, review)
    output = work_dir / R3_LABELS_NAME
    if output.exists():
        raise SystemExit("A versão R3 já existe; ela nunca é sobrescrita.")
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(new_rows)
    print(f"Fotos alteradas: {changed} de {len(review)} revisadas")
    print(f"SHA-256 da versão R3: {file_sha256(output)}")


if __name__ == "__main__":
    main()
