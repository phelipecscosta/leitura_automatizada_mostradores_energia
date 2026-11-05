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
