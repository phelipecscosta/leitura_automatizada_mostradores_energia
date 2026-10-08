"""Rótulos do Lab02 prontos para o treino (T3.1).

O arquivo de rótulos (lab2_rotulos.csv) fica no repositório de pesquisa e
usa só pseudônimos. Este módulo recebe o caminho dele como argumento, para
que o pacote training não dependa do repositório de pesquisa.

Regras (protocolo de avaliação, seção 2; E19 e E22):
- rótulo final: o da adjudicação (ADJ), se existir; senão, o do R1;
- as linhas do R2 servem só ao kappa e são ignoradas aqui;
- cena com 3 classes; o indeterminado vira rótulo parcial ("é medidor");
- legibilidade codificada como "ilegível" (a classe positiva da métrica),
  mascarada nas fotos de "outros".
"""

from __future__ import annotations

import csv
from pathlib import Path

SCENE_CLASSES = ("digital", "ciclometrico", "outros")
SCENE_METER_ONLY = -1  # indeterminado: sabe-se só que é medidor
MASKED = -1  # legibilidade não se aplica (foto de "outros")

_METER_TYPES = {"digital", "ciclometrico", "indeterminado"}
_ILLEGIBLE = {"legivel": 0, "ilegivel": 1}


class LabelError(ValueError):
    """Arquivo de rótulos inconsistente."""


def read_final_labels(path: Path) -> dict[str, dict]:
    """Pseudônimo -> linha do rótulo final (ADJ, se existir; senão, R1)."""
    r1: dict[str, dict] = {}
    adj: dict[str, dict] = {}
    with path.open(encoding="utf-8", newline="") as file:
        # start=2: a linha 1 é o cabeçalho; assim o número bate com o editor
        for line, row in enumerate(csv.DictReader(file), start=2):
            rater = row["rotulador"]
            if rater == "R2":
                continue
            if rater not in ("R1", "ADJ"):
                raise LabelError(f"Linha {line}: rotulador desconhecido.")
            target = r1 if rater == "R1" else adj
            if row["nome_arquivo"] in target:
                raise LabelError(f"Linha {line}: rotulador repetido para a mesma foto.")
            target[row["nome_arquivo"]] = row

    orphans = adj.keys() - r1.keys()
    if orphans:
        raise LabelError(f"{len(orphans)} adjudicação(ões) sem rótulo do R1.")
    return {**r1, **adj}  # a adjudicação substitui o R1


def encode(row: dict) -> tuple[int, int]:
    """Converte uma linha final nas metas das duas cabeças: (cena, ilegível)."""
    scene_n1, scene_n2, legibility = row["cena_n1"], row["cena_n2"], row["legibilidade"]

    if scene_n1 == "outros":
        # Em "outros" o tipo e a legibilidade não existem (esquema, versão 4)
        if scene_n2 or legibility:
            raise LabelError("Foto de outros com tipo ou legibilidade preenchidos.")
        return SCENE_CLASSES.index("outros"), MASKED

    if scene_n1 != "medidor" or scene_n2 not in _METER_TYPES or legibility not in _ILLEGIBLE:
        raise LabelError("Rótulo fora do vocabulário do esquema.")

    if scene_n2 == "indeterminado":
        scene = SCENE_METER_ONLY
    else:
        scene = SCENE_CLASSES.index(scene_n2)
    return scene, _ILLEGIBLE[legibility]
