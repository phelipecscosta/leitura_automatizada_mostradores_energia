"""Fotos sem rótulo que o SCAN agrupa (protocolo de ML, seção 1.1).

Universo: todas as fotos íntegras dos três lotes de desenvolvimento, órfãs
incluídas. O lote de teste fica de fora: ele não participa do agrupamento
(E11, regra 4) e só é lido na abertura única (seção 1.9).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from meter_reader.config import get_data_dir, get_work_dir
from training.build_manifest import MANIFEST_NAME
from training.build_sample import assign_sets
from training.labels import TEST_SUBSET
from training.pseudonym import KEY_NAME, build_mapping, load_key


@dataclass(frozen=True)
class UnlabeledPhoto:
    """Uma foto do universo do agrupamento. Nunca exibir `path` em saídas."""

    pseudonym: str
    lot: str
    path: Path


def development_photos(manifest: list[dict], mapping: dict[str, str],
                       paths: dict[str, Path]) -> list[UnlabeledPhoto]:
    """Fotos íntegras fora do lote de teste, ordenadas pelo pseudônimo."""
    sets = assign_sets({r["lote"] for r in manifest})
    photos = [
        UnlabeledPhoto(mapping[r["arquivo"]], r["lote"], paths[r["arquivo"]])
        for r in manifest
        if r["integridade"] == "" and sets[r["lote"]] != TEST_SUBSET
    ]
    return sorted(photos, key=lambda p: p.pseudonym)


def load_development_photos() -> list[UnlabeledPhoto]:
    """Lê o manifesto e a chave da pasta de trabalho e localiza as fotos."""
    work_dir, data_dir = get_work_dir(), get_data_dir()
    with (work_dir / MANIFEST_NAME).open(encoding="utf-8", newline="") as file:
        manifest = list(csv.DictReader(file))
    mapping = build_mapping(load_key(work_dir / KEY_NAME), "arquivo",
                            [r["arquivo"] for r in manifest])
    paths = {p.name: p for p in data_dir.rglob("*") if p.suffix.lower() == ".jpg"}
    return development_photos(manifest, mapping, paths)
