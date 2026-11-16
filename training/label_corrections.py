"""Aplica correções aos rótulos da rotulagem completa, sem mover fotos.

Cada revisão gera uma nova versão dos rótulos ao lado da anterior, que nunca é
alterada nem sobrescrita.
- R3 (T4f.8): <pasta de trabalho>/revisao_r3/revisao_r3.csv sobre os rótulos
  originais do desenvolvimento -> rotulos_grupos_r3.csv.
- R4 (ET5a, T5.4): <pasta de trabalho>/revisao_caixas/revisao_caixas.csv, a
  revisão dos grupos guiada pelas caixas anotadas. As linhas do desenvolvimento
  partem da R3 -> rotulos_grupos_r4.csv; as do teste partem dos rótulos
  originais do teste -> rotulos_grupos_teste_r4.csv. A abertura do teste (E68)
  usou a versão anterior e não é recalculada.

Uso, a partir da raiz do produto:
    python -m training.label_corrections          (R3)
    python -m training.label_corrections caixas   (R4)
"""

from __future__ import annotations

import csv
import sys

from meter_reader.config import get_work_dir
from training.build_manifest import file_sha256
from training.group_dataset import DEV_LABELS, DEV_LABELS_R3, TEST_LABELS, GroupLabelFileError
from training.group_labels import FOLDERS

R3_LABELS_NAME = "rotulos_grupos_r3.csv"
R4_LABELS_NAME = "rotulos_grupos_r4.csv"
TEST_R4_LABELS_NAME = "rotulos_grupos_teste_r4.csv"
BOX_REVIEW = ("revisao_caixas", "revisao_caixas.csv")
SUBSETS = ("desenvolvimento", "teste")
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


def split_review(review: list[dict]) -> dict[str, list[dict]]:
    """Separa a revisão das caixas por conjunto; recusa conjunto desconhecido."""
    parts = {subset: [] for subset in SUBSETS}
    for line, item in enumerate(review, start=2):
        if item["conjunto"] not in parts:
            raise GroupLabelFileError(f"Revisão, linha {line}: conjunto desconhecido.")
        parts[item["conjunto"]].append(item)
    return parts


def _read(path) -> tuple[list[str], list[dict]]:
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader.fieldnames or ()), list(reader)


def _new_version(work_dir, source: tuple[str, str], review: list[dict], output_name: str) -> None:
    """Confere a impressão digital da origem, aplica a revisão e grava a nova versão."""
    name, sha256 = source
    if file_sha256(work_dir / name) != sha256:
        raise SystemExit(f"{name}: impressão digital diferente da esperada.")
    columns, rows = _read(work_dir / name)
    new_rows, changed = apply_corrections(rows, review)
    output = work_dir / output_name
    if output.exists():
        raise SystemExit(f"{output_name} já existe; uma versão nunca é sobrescrita.")
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(new_rows)
    print(f"{output_name}: {changed} fotos alteradas de {len(review)} revisadas; "
          f"SHA-256 {file_sha256(output)}")


def main_r3() -> None:
    work_dir = get_work_dir()
    _, review = _read(work_dir / "revisao_r3" / "revisao_r3.csv")
    _new_version(work_dir, DEV_LABELS, review, R3_LABELS_NAME)


def main_r4() -> None:
    work_dir = get_work_dir()
    _, review = _read(work_dir.joinpath(*BOX_REVIEW))
    parts = split_review(review)
    _new_version(work_dir, DEV_LABELS_R3, parts["desenvolvimento"], R4_LABELS_NAME)
    _new_version(work_dir, TEST_LABELS, parts["teste"], TEST_R4_LABELS_NAME)


if __name__ == "__main__":
    if sys.argv[1:] == ["caixas"]:
        main_r4()
    elif not sys.argv[1:]:
        main_r3()
    else:
        raise SystemExit("Uso: python -m training.label_corrections [caixas]")
