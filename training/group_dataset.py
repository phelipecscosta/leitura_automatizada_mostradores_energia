"""Rótulos da triagem de cinco classes prontos para o treino (T4f.1).

Lê o arquivo gerado por training.build_group_labels (rotulos_grupos.csv ou
rotulos_grupos_teste.csv, na pasta de trabalho) com validação estrita.
Protocolo: exploracao/triagem_README.md, seções 1.1 e 1.2. As mensagens de
erro mostram só o número da linha e a coluna, nunca o valor.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

from training.build_group_labels import COLUMNS
from training.build_manifest import file_sha256
from training.build_sample import SETS_BY_RANK
from training.group_labels import (
    CONDITIONS, GROUPS, TRAINING_GROUP_CLASS, TRAINING_GROUPS, training_group,
)

_PSEUDONYM = re.compile(r"^F-[0-9a-f]{12}$")
_LOT = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_PAIRED = {"True": True, "False": False}


class GroupLabelFileError(ValueError):
    """Arquivo de rótulos da rotulagem completa inconsistente."""


@dataclass(frozen=True)
class GroupPhoto:
    """Uma foto rotulada. Nunca exibir `path` em saídas versionadas."""

    pseudonym: str
    lot: str
    subset: str
    paired: bool
    group: str          # grupo original do arquivo (14 possíveis)
    condition: str
    path: Path | None = None  # preenchido na junção com o manifesto

    @property
    def target(self) -> int:
        """Índice do grupo de treino (fusão aplicada) em TRAINING_GROUPS."""
        return TRAINING_GROUPS.index(training_group(self.group))

    @property
    def product_class(self) -> int:
        """Índice da classe do produto em PRODUCT_CLASSES."""
        return TRAINING_GROUP_CLASS[self.target]


def read_group_labels(path: Path, expected_sha256: str | None = None) -> list[GroupPhoto]:
    """Lê e valida o arquivo; confere a impressão digital quando informada."""
    if expected_sha256 is not None and file_sha256(path) != expected_sha256:
        raise GroupLabelFileError("Impressão digital do arquivo de rótulos diferente da esperada.")

    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise GroupLabelFileError("Cabeçalho diferente do esperado.")
        rows = list(reader)

    # (coluna, regra de validade) verificadas em toda linha
    checks = (
        ("pseudonimo", lambda v: bool(_PSEUDONYM.match(v))),
        ("lote", lambda v: bool(_LOT.match(v))),
        ("conjunto", lambda v: v in SETS_BY_RANK),
        ("pareada", lambda v: v in _PAIRED),
        ("grupo", lambda v: v in GROUPS),
        ("condicao", lambda v: v in CONDITIONS),
    )
    photos, seen = [], set()
    # start=2: a linha 1 é o cabeçalho; assim o número bate com o editor
    for line, row in enumerate(rows, start=2):
        for column, valid in checks:
            if not valid(row[column]):
                raise GroupLabelFileError(f"Linha {line}: valor inválido na coluna {column}.")
        if row["pseudonimo"] in seen:
            raise GroupLabelFileError(f"Linha {line}: pseudônimo repetido.")
        seen.add(row["pseudonimo"])
        photos.append(GroupPhoto(
            pseudonym=row["pseudonimo"], lot=row["lote"], subset=row["conjunto"],
            paired=_PAIRED[row["pareada"]], group=row["grupo"], condition=row["condicao"],
        ))
    return photos

# ---------------------------------------------------------------------
# Junção com o manifesto e universos das dobras
# ---------------------------------------------------------------------

from dataclasses import replace  # noqa: E402

import numpy as np  # noqa: E402

from meter_reader.config import get_data_dir, get_work_dir  # noqa: E402
from training.build_manifest import MANIFEST_NAME  # noqa: E402
from training.pseudonym import KEY_NAME, build_mapping, load_key  # noqa: E402

# Impressões digitais congeladas no protocolo (seção 1.1) e na E58. Uma versão
# corrigida pela revisão R3 ganha constante própria; estas nunca mudam.
DEV_LABELS = ("rotulos_grupos.csv",
              "e521acd484c3b46d3b5cd312edc327dbc3e830917a5d1185030ff2384cd3d5b4")
TEST_LABELS = ("rotulos_grupos_teste.csv",
               "e98bee885c5ae8898c40b45b2174b853cf81d28a2028332e511f01c72e3645c1")
# Versão corrigida pela revisão R3 (training.label_corrections); a original nunca muda
DEV_LABELS_R3 = ("rotulos_grupos_r3.csv",
                 "c5fda76dc5512c39f46cfffa2731f85fcdc887a744608866834672a11306ff54")
LABEL_VERSIONS = {"v1": DEV_LABELS, "r3": DEV_LABELS_R3}


def attach_paths(photos: list[GroupPhoto], manifest: list[dict],
                 mapping: dict[str, str], paths: dict[str, Path]) -> list[GroupPhoto]:
    """Preenche o caminho de cada foto e confere lote e pareamento com o manifesto.

    `mapping`: nome real -> pseudônimo; `paths`: nome real -> caminho no disco.
    As mensagens mostram só contagens.
    """
    by_code = {mapping[r["arquivo"]]: r for r in manifest}
    unknown = sum(1 for p in photos if p.pseudonym not in by_code)
    if unknown:
        raise GroupLabelFileError(f"{unknown} pseudônimo(s) fora do manifesto.")

    records = [by_code[p.pseudonym] for p in photos]
    wrong_lot = sum(1 for p, r in zip(photos, records) if p.lot != r["lote"])
    wrong_pair = sum(1 for p, r in zip(photos, records) if p.paired != (r["pareada"] == "True"))
    missing = sum(1 for r in records if r["arquivo"] not in paths)
    if wrong_lot or wrong_pair or missing:
        raise GroupLabelFileError(
            f"Divergências com o manifesto: {wrong_lot} de lote, "
            f"{wrong_pair} de pareamento, {missing} sem arquivo no disco.")
    return [replace(p, path=paths[r["arquivo"]]) for p, r in zip(photos, records)]


def load_group_photos(labels: tuple[str, str] = DEV_LABELS) -> list[GroupPhoto]:
    """Rótulos da pasta de trabalho, conferidos e com os caminhos das fotos."""
    work_dir = get_work_dir()
    name, sha256 = labels
    photos = read_group_labels(work_dir / name, sha256)
    with (work_dir / MANIFEST_NAME).open(encoding="utf-8", newline="") as file:
        manifest = list(csv.DictReader(file))
    mapping = build_mapping(load_key(work_dir / KEY_NAME), "arquivo",
                            [r["arquivo"] for r in manifest])
    paths = {p.name: p for p in get_data_dir().rglob("*") if p.suffix.lower() == ".jpg"}
    return attach_paths(photos, manifest, mapping, paths)


def fold_indices(photos: list[GroupPhoto], held_out_lot: str) -> tuple[np.ndarray, np.ndarray]:
    """(treino, avaliação) de uma dobra (protocolo, seções 1.1 e 1.2).

    Treino: os outros lotes, pareadas e órfãs. Avaliação: o lote deixado de
    fora, só as pareadas (as órfãs são rejeitadas antes da triagem, E11).
    """
    lots = np.array([p.lot for p in photos])
    paired = np.array([p.paired for p in photos])
    if held_out_lot not in lots:
        raise ValueError("Lote deixado de fora inexistente.")
    return np.flatnonzero(lots != held_out_lot), np.flatnonzero((lots == held_out_lot) & paired)


# ---------------------------------------------------------------------
# Mínimo por grupo e divisão interna das épocas
# ---------------------------------------------------------------------

MIN_GROUP_SIZE = 30         # critério da fusão U-b (protocolo, seção 1.1)
INTERNAL_FRACTION = 0.10    # protocolo, seção 1.2 (opção E-b)
INTERNAL_SEED = 0           # fixa: a mesma divisão para todos os candidatos


def check_group_sizes(photos: list[GroupPhoto], minimum: int = MIN_GROUP_SIZE) -> dict[str, int]:
    """Fotos por grupo de treino; falha se algum ficar abaixo do mínimo.

    Protege contra rótulos corrigidos (revisão R3) que esvaziem um grupo.
    """
    counts = {g: 0 for g in TRAINING_GROUPS}
    for p in photos:
        counts[TRAINING_GROUPS[p.target]] += 1
    small = [g for g, n in counts.items() if n < minimum]
    if small:
        raise GroupLabelFileError(f"Grupo(s) de treino com menos de {minimum} fotos: {small}.")
    return counts


def internal_split(photos: list[GroupPhoto], train: np.ndarray,
                   fraction: float = INTERNAL_FRACTION,
                   seed: int = INTERNAL_SEED) -> tuple[np.ndarray, np.ndarray]:
    """(ajuste, interna) dentro do treino de uma dobra.

    A parte interna reúne `fraction` das pareadas de cada grupo de treino,
    sorteadas com semente fixa; as órfãs ficam sempre no ajuste.
    """
    rng = np.random.default_rng(seed)
    internal = []
    for target in range(len(TRAINING_GROUPS)):
        pool = np.array([i for i in train if photos[i].paired and photos[i].target == target])
        size = round(fraction * len(pool))
        if size:
            internal.extend(rng.choice(pool, size=size, replace=False).tolist())
    internal = np.sort(np.array(internal, dtype=int))
    return np.setdiff1d(train, internal), internal
