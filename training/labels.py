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
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from meter_reader.config import get_data_dir, get_work_dir
from training.build_manifest import file_sha256
from training.labeling_sheet import SHEET_NAME

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

TEST_SUBSET = "teste"
_SCENE_NAMES = {**dict(enumerate(SCENE_CLASSES)), SCENE_METER_ONLY: "indeterminado"}


@dataclass(frozen=True)
class LabeledPhoto:
    """Uma foto rotulada, pronta para o treino.

    O campo `path` aponta para a foto do cliente: nunca exibi-lo em saídas.
    """

    pseudonym: str
    lot: str
    subset: str     # treino ou validacao (teste só com include_test=True)
    role: str       # cega, rotulagem ou estresse_zero
    path: Path
    scene: int      # índice em SCENE_CLASSES, ou SCENE_METER_ONLY
    illegible: int  # 1 ilegível, 0 legível, ou MASKED


def join_with_sheet(final: dict[str, dict], sheet: list[dict], data_dir: Path,
                    include_test: bool = False) -> list[LabeledPhoto]:
    """Junta os rótulos finais com a folha de rotulagem, pelo pseudônimo.

    O lote de teste fica de fora por padrão: ele é aberto uma única vez,
    na ET4 (protocolo de avaliação, seção 2.1).
    """
    by_code = {row["pseudonimo"]: row for row in sheet}
    missing = final.keys() - by_code.keys()
    if missing:
        raise LabelError(f"{len(missing)} foto(s) rotulada(s) ausente(s) da folha.")

    photos = []
    for code in sorted(final):  # ordem estável, independente do arquivo
        row, info = final[code], by_code[code]
        if row["lote"] != info["lote"]:
            raise LabelError("Lote divergente entre os rótulos e a folha.")
        if info["conjunto"] == TEST_SUBSET and not include_test:
            continue
        scene, illegible = encode(row)
        photos.append(LabeledPhoto(
            pseudonym=code, lot=info["lote"], subset=info["conjunto"],
            # A folha guarda o caminho relativo à pasta de dados (T2.5)
            role=info["papel"], path=data_dir / info["caminho"],
            scene=scene, illegible=illegible,
        ))
    return photos


def load_labeled_photos(labels_path: Path, include_test: bool = False) -> list[LabeledPhoto]:
    """Lê os rótulos (caminho recebido) e a folha (pasta de trabalho)."""
    sheet_path = get_work_dir() / SHEET_NAME
    with sheet_path.open(encoding="utf-8", newline="") as file:
        sheet = list(csv.DictReader(file))
    return join_with_sheet(read_final_labels(labels_path), sheet, get_data_dir(), include_test)


def main(argv: list[str]) -> None:
    if len(argv) != 1:
        raise SystemExit("Uso: python -m training.labels CAMINHO_DO_LAB2_ROTULOS")
    labels_path = Path(argv[0])
    photos = load_labeled_photos(labels_path)

    # Só contagens e a impressão digital; nunca pseudônimos nem caminhos
    counts = Counter(
        (p.lot, p.role, _SCENE_NAMES[p.scene],
         {1: "ilegivel", 0: "legivel", MASKED: "-"}[p.illegible])
        for p in photos
    )
    for key, n in sorted(counts.items()):
        print(" ".join(f"{k:14}" for k in key), f"{n:4}")
    print(f"Fotos fora do teste: {len(photos)}")
    print(f"Arquivos de foto ausentes: {sum(not p.path.is_file() for p in photos)}")
    print(f"SHA-256 dos rótulos: {file_sha256(labels_path)}")


if __name__ == "__main__":
    main(sys.argv[1:])
