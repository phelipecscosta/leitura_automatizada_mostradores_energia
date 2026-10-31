"""Converte as pastas de cluster_Rotulado num arquivo de rótulos por pseudônimo (T4e.1).

Cada foto recebe grupo e condição (pela pasta), lote, conjunto e pareamento
(pelo manifesto). A classe do produto não é gravada: sai do grupo, em
training.group_labels. O arquivo fica na pasta de trabalho (opção L-a) e
nunca contém nomes reais.

Uso, a partir da raiz do repositório do produto:
    python -m training.build_group_labels
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from meter_reader.config import get_work_dir
from training.build_manifest import MANIFEST_NAME, file_sha256
from training.build_sample import assign_sets
from training.group_labels import folder_label, product_class
from training.labels import TEST_SUBSET
from training.pseudonym import KEY_NAME, build_mapping, load_key

LABELED_DIR = "cluster_Rotulado"
LABELS_NAME = "rotulos_grupos.csv"
COLUMNS = ("pseudonimo", "lote", "conjunto", "pareada", "grupo", "condicao")


class GroupLabelError(ValueError):
    """Pastas de rótulos inconsistentes com o manifesto."""


def build_group_labels(found: list[tuple[str, str]], manifest: list[dict],
                       mapping: dict[str, str]) -> list[dict]:
    """Uma linha por foto rotulada, ordenada pelo pseudônimo.

    `found` traz pares (pasta relativa, nome do arquivo). As mensagens de
    erro mostram só contagens, nunca nomes.
    """
    by_name = {r["arquivo"]: r for r in manifest}
    sets = assign_sets({r["lote"] for r in manifest})
    names = [name for _, name in found]

    repeated = sum(1 for count in Counter(names).values() if count > 1)
    if repeated:
        raise GroupLabelError(f"{repeated} foto(s) em mais de uma pasta.")
    unknown = sum(1 for name in names if name not in by_name)
    if unknown:
        raise GroupLabelError(f"{unknown} foto(s) fora do manifesto.")
    in_test = sum(1 for name in names if sets[by_name[name]["lote"]] == TEST_SUBSET)
    if in_test:
        raise GroupLabelError(f"{in_test} foto(s) do lote de teste.")
    broken = sum(1 for name in names if by_name[name]["integridade"] != "")
    if broken:
        raise GroupLabelError(f"{broken} foto(s) não íntegra(s).")

    rows = []
    for folder, name in found:
        group, condition = folder_label(folder)
        record = by_name[name]
        rows.append({
            "pseudonimo": mapping[name],
            "lote": record["lote"],
            "conjunto": sets[record["lote"]],
            "pareada": record["pareada"],
            "grupo": group,
            "condicao": condition,
        })
    return sorted(rows, key=lambda r: r["pseudonimo"])


def unlabeled_counts(rows: list[dict], manifest: list[dict],
                     mapping: dict[str, str]) -> Counter:
    """Fotos íntegras de desenvolvimento sem rótulo, por conjunto."""
    sets = assign_sets({r["lote"] for r in manifest})
    labeled = {r["pseudonimo"] for r in rows}
    return Counter(
        sets[r["lote"]] for r in manifest
        if r["integridade"] == "" and sets[r["lote"]] != TEST_SUBSET
        and mapping[r["arquivo"]] not in labeled
    )


def _scan(root: Path) -> list[tuple[str, str]]:
    """Pares (pasta relativa, nome) de todos os .jpg; outros arquivos são ignorados."""
    return [
        (path.parent.relative_to(root).as_posix(), path.name)
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() == ".jpg"
    ]


def main() -> None:
    work_dir = get_work_dir()
    with (work_dir / MANIFEST_NAME).open(encoding="utf-8", newline="") as file:
        manifest = list(csv.DictReader(file))
    mapping = build_mapping(load_key(work_dir / KEY_NAME), "arquivo",
                            [r["arquivo"] for r in manifest])

    rows = build_group_labels(_scan(work_dir / LABELED_DIR), manifest, mapping)
    output = work_dir / LABELS_NAME
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    # Resumo: só contagens
    per_class = Counter(product_class(r["grupo"]) for r in rows)
    orphans = Counter(product_class(r["grupo"]) for r in rows if r["pareada"] != "True")
    print(f"Fotos rotuladas: {len(rows)}")
    for name in sorted(per_class):
        print(f"  {name:14s} {per_class[name]:6d}   órfãs: {orphans[name]:5d}")
    print("Órfãs por conjunto:", dict(Counter(r["conjunto"] for r in rows if r["pareada"] != "True")))
    print("Sem rótulo, por conjunto:", dict(unlabeled_counts(rows, manifest, mapping)))
    print(f"SHA-256 de {LABELS_NAME}: {file_sha256(output)}")


if __name__ == "__main__":
    main()
