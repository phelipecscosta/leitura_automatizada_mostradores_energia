"""Converte as pastas rotuladas num arquivo de rótulos por pseudônimo (T4e.1 e T4e.4).

Cada foto recebe grupo e condição (pela pasta), lote, conjunto e pareamento
(pelo manifesto). A classe do produto não é gravada: sai do grupo, em
training.group_labels. O arquivo fica na pasta de trabalho (opção L-a) e
nunca contém nomes reais.

Uso, a partir da raiz do repositório do produto:
    python -m training.build_group_labels          (desenvolvimento)
    python -m training.build_group_labels teste    (lote de teste)
"""

from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from pathlib import Path

from meter_reader.config import get_work_dir
from training.build_manifest import MANIFEST_NAME, file_sha256
from training.build_sample import assign_sets
from training.group_labels import folder_label, product_class
from training.labels import TEST_SUBSET
from training.pseudonym import KEY_NAME, build_mapping, load_key

DEVELOPMENT = "desenvolvimento"
# Parte -> (pasta rotulada na pasta de trabalho, arquivo de rótulos gerado)
SOURCES = {
    DEVELOPMENT: ("cluster_Rotulado", "rotulos_grupos.csv"),
    TEST_SUBSET: ("teste_Rotulado", "rotulos_grupos_teste.csv"),
}
COLUMNS = ("pseudonimo", "lote", "conjunto", "pareada", "grupo", "condicao")
# Posição de tipicidade que os scripts de cópia acrescentam (0001_ = mais típica)
_RANK_PREFIX = re.compile(r"^\d{4}_")


class GroupLabelError(ValueError):
    """Pastas de rótulos inconsistentes com o manifesto."""


def original_name(name: str, known) -> str:
    """Nome do manifesto. O prefixo de tipicidade só é retirado se o restante
    existir no manifesto; um nome truncado continua desconhecido."""
    if name in known:
        return name
    stripped = _RANK_PREFIX.sub("", name, count=1)
    return stripped if stripped != name and stripped in known else name


def build_group_labels(found: list[tuple[str, str]], manifest: list[dict],
                       mapping: dict[str, str], part: str = DEVELOPMENT) -> list[dict]:
    """Uma linha por foto rotulada, ordenada pelo pseudônimo.

    `found` traz pares (pasta relativa, nome do arquivo). As mensagens de
    erro mostram só contagens, nunca nomes.
    """
    if part not in SOURCES:
        raise ValueError(f"Parte desconhecida: {part}")
    by_name = {r["arquivo"]: r for r in manifest}
    sets = assign_sets({r["lote"] for r in manifest})
    found = [(folder, original_name(name, by_name)) for folder, name in found]
    names = [name for _, name in found]

    repeated = sum(1 for count in Counter(names).values() if count > 1)
    if repeated:
        raise GroupLabelError(f"{repeated} foto(s) em mais de uma pasta.")
    unknown = sum(1 for name in names if name not in by_name)
    if unknown:
        raise GroupLabelError(f"{unknown} foto(s) fora do manifesto.")

    in_test = [sets[by_name[name]["lote"]] == TEST_SUBSET for name in names]
    if part == DEVELOPMENT and any(in_test):
        raise GroupLabelError(f"{sum(in_test)} foto(s) do lote de teste.")
    if part == TEST_SUBSET:
        outside = in_test.count(False)
        if outside:
            raise GroupLabelError(f"{outside} foto(s) fora do lote de teste.")
        orphans = sum(1 for name in names if by_name[name]["pareada"] != "True")
        if orphans:
            raise GroupLabelError(f"{orphans} órfã(s) no teste.")
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


def _in_universe(record: dict, subset: str, part: str) -> bool:
    """Foto que deveria estar rotulada nesta parte."""
    if record["integridade"] != "":
        return False
    if part == TEST_SUBSET:
        return subset == TEST_SUBSET and record["pareada"] == "True"
    return subset != TEST_SUBSET


def unlabeled_counts(rows: list[dict], manifest: list[dict],
                     mapping: dict[str, str], part: str = DEVELOPMENT) -> Counter:
    """Fotos do universo da parte que ficaram sem rótulo, por conjunto."""
    sets = assign_sets({r["lote"] for r in manifest})
    labeled = {r["pseudonimo"] for r in rows}
    return Counter(
        sets[r["lote"]] for r in manifest
        if _in_universe(r, sets[r["lote"]], part) and mapping[r["arquivo"]] not in labeled
    )


def _scan(root: Path) -> list[tuple[str, str]]:
    """Pares (pasta relativa, nome) de todos os .jpg; outros arquivos são ignorados."""
    return [
        (path.parent.relative_to(root).as_posix(), path.name)
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() == ".jpg"
    ]


def main(argv: list[str]) -> None:
    part = argv[0] if argv else DEVELOPMENT
    if len(argv) > 1 or part not in SOURCES:
        raise SystemExit("Uso: python -m training.build_group_labels [teste]")
    folder_name, labels_name = SOURCES[part]

    work_dir = get_work_dir()
    with (work_dir / MANIFEST_NAME).open(encoding="utf-8", newline="") as file:
        manifest = list(csv.DictReader(file))
    mapping = build_mapping(load_key(work_dir / KEY_NAME), "arquivo",
                            [r["arquivo"] for r in manifest])

    rows = build_group_labels(_scan(work_dir / folder_name), manifest, mapping, part)
    output = work_dir / labels_name
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    # Resumo: só contagens
    per_class = Counter(product_class(r["grupo"]) for r in rows)
    orphans = Counter(product_class(r["grupo"]) for r in rows if r["pareada"] != "True")
    print(f"Parte: {part} | fotos rotuladas: {len(rows)}")
    for name in sorted(per_class):
        print(f"  {name:14s} {per_class[name]:6d}   órfãs: {orphans[name]:5d}")
    print("Órfãs por conjunto:", dict(Counter(r["conjunto"] for r in rows if r["pareada"] != "True")))
    print("Sem rótulo, por conjunto:", dict(unlabeled_counts(rows, manifest, mapping, part)))
    print(f"SHA-256 de {labels_name}: {file_sha256(output)}")


if __name__ == "__main__":
    main(sys.argv[1:])
