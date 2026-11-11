"""Caixas anotadas para o detector: formato, gravação atômica e validação (ET5a, T5.3).

Esquema: 02_dataset.md, seção 5. Dois arquivos na pasta de trabalho:
- caixas.csv: uma linha por caixa (pseudônimo, classe, x0, y0, x1, y1), em
  pixels da foto original, com a origem no canto superior esquerdo;
- caixas_fotos.csv: uma linha por foto anotada (pseudônimo, número de caixas,
  segundos, data e hora). É o registro de "foto anotada": uma foto com 0 caixas
  está anotada e não tem mostrador; uma foto fora dele ainda não foi anotada.
A ferramenta de anotação (local/ da pesquisa) e o treino do detector usam este
módulo. As mensagens de erro mostram a linha e a coluna, nunca os valores.
"""
from __future__ import annotations

import csv
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

BOX_CLASSES = ("display_digital", "display_ciclometrico", "display_kvarh", "numero_serie")
BOXES_NAME = "caixas.csv"
PHOTOS_NAME = "caixas_fotos.csv"
BOX_COLUMNS = ("pseudonimo", "classe", "x0", "y0", "x1", "y1")
PHOTO_COLUMNS = ("pseudonimo", "n_caixas", "segundos", "data_hora")
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"

_PSEUDONYM = re.compile(r"^F-[0-9a-f]{12}$")


class BoxLabelError(ValueError):
    """Arquivo de caixas inconsistente."""


@dataclass(frozen=True)
class Box:
    pseudonym: str
    cls: str
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass(frozen=True)
class PhotoRecord:
    pseudonym: str
    n_boxes: int
    seconds: float
    timestamp: str  # TIME_FORMAT


def check_box(box: Box, width: int | None = None, height: int | None = None) -> None:
    """Classe permitida, área positiva e coordenadas dentro da foto (se o tamanho for dado)."""
    if box.cls not in BOX_CLASSES:
        raise BoxLabelError("Classe de caixa fora do esquema.")
    if not (0 <= box.x0 < box.x1 and 0 <= box.y0 < box.y1):
        raise BoxLabelError("Caixa sem área positiva ou com coordenada negativa.")
    if width is not None and box.x1 > width or height is not None and box.y1 > height:
        raise BoxLabelError("Caixa fora dos limites da foto.")


def _write_atomic(path: Path, columns: tuple[str, ...], rows: list[dict]) -> None:
    """Temporário na mesma pasta e os.replace: um arquivo nunca fica pela metade."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def write_labels(work_dir: Path, photos: list[PhotoRecord], boxes: list[Box]) -> None:
    """Reescreve os dois arquivos inteiros. As caixas vão primeiro; o registro das
    fotos, que marca o que está anotado, vai por último."""
    # Arredonda antes de validar: o que é conferido é o que vai para o arquivo
    boxes = [Box(b.pseudonym, b.cls, *(round(v, 1) for v in (b.x0, b.y0, b.x1, b.y1)))
             for b in boxes]
    counts: dict[str, int] = {}
    for box in boxes:
        check_box(box)
        counts[box.pseudonym] = counts.get(box.pseudonym, 0) + 1
    codes = [p.pseudonym for p in photos]
    if len(set(codes)) != len(codes):
        raise BoxLabelError("Foto registrada mais de uma vez.")
    if any(counts.get(p.pseudonym, 0) != p.n_boxes for p in photos) or set(counts) - set(codes):
        raise BoxLabelError("Número de caixas diferente do registro das fotos.")
    _write_atomic(work_dir / BOXES_NAME, BOX_COLUMNS, [
        {"pseudonimo": b.pseudonym, "classe": b.cls, "x0": f"{b.x0:.1f}", "y0": f"{b.y0:.1f}",
         "x1": f"{b.x1:.1f}", "y1": f"{b.y1:.1f}"} for b in boxes])
    _write_atomic(work_dir / PHOTOS_NAME, PHOTO_COLUMNS, [
        {"pseudonimo": p.pseudonym, "n_caixas": p.n_boxes, "segundos": f"{p.seconds:.1f}",
         "data_hora": p.timestamp} for p in photos])


def _rows(path: Path, columns: tuple[str, ...]) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != columns:
            raise BoxLabelError(f"{path.name}: cabeçalho diferente do esperado.")
        return list(reader)


def _number(value: str, line: int, column: str, name: str) -> float:
    try:
        number = float(value)
    except ValueError:
        number = math.nan
    if not math.isfinite(number):  # recusa texto, nan e inf
        raise BoxLabelError(f"{name}, linha {line}: valor inválido na coluna {column}.")
    return number


def read_labels(work_dir: Path, sample: set[str] | None = None,
                sizes: dict[str, tuple[int, int]] | None = None
                ) -> tuple[list[PhotoRecord], list[Box]]:
    """Lê e valida os dois arquivos. `sample`: pseudônimos permitidos; `sizes`:
    pseudônimo -> (largura, altura), para conferir os limites das caixas.
    Sem arquivos, devolve listas vazias (nenhuma foto anotada ainda)."""
    boxes_path, photos_path = work_dir / BOXES_NAME, work_dir / PHOTOS_NAME
    if not boxes_path.exists() and not photos_path.exists():
        return [], []
    if not (boxes_path.exists() and photos_path.exists()):
        raise BoxLabelError("Só um dos dois arquivos de caixas existe.")

    photos, seen = [], set()
    for line, row in enumerate(_rows(photos_path, PHOTO_COLUMNS), start=2):
        code = row["pseudonimo"]
        if not _PSEUDONYM.match(code) or (sample is not None and code not in sample):
            raise BoxLabelError(f"{PHOTOS_NAME}, linha {line}: pseudônimo inválido ou fora da amostra.")
        if code in seen:
            raise BoxLabelError(f"{PHOTOS_NAME}, linha {line}: foto repetida.")
        seen.add(code)
        if not row["n_caixas"].isdigit():
            raise BoxLabelError(f"{PHOTOS_NAME}, linha {line}: valor inválido na coluna n_caixas.")
        seconds = _number(row["segundos"], line, "segundos", PHOTOS_NAME)
        try:
            datetime.strptime(row["data_hora"], TIME_FORMAT)
        except ValueError:
            raise BoxLabelError(f"{PHOTOS_NAME}, linha {line}: valor inválido na coluna data_hora.") from None
        if seconds < 0:
            raise BoxLabelError(f"{PHOTOS_NAME}, linha {line}: valor inválido na coluna segundos.")
        photos.append(PhotoRecord(code, int(row["n_caixas"]), seconds, row["data_hora"]))

    boxes, counts = [], {}
    for line, row in enumerate(_rows(boxes_path, BOX_COLUMNS), start=2):
        code = row["pseudonimo"]
        if code not in seen:
            raise BoxLabelError(f"{BOXES_NAME}, linha {line}: caixa de uma foto sem registro.")
        coords = [_number(row[c], line, c, BOXES_NAME) for c in ("x0", "y0", "x1", "y1")]
        box = Box(code, row["classe"], *coords)
        width, height = sizes[code] if sizes is not None else (None, None)
        try:
            check_box(box, width, height)
        except BoxLabelError as error:
            raise BoxLabelError(f"{BOXES_NAME}, linha {line}: {error}") from None
        boxes.append(box)
        counts[code] = counts.get(code, 0) + 1

    wrong = sum(1 for p in photos if counts.get(p.pseudonym, 0) != p.n_boxes)
    if wrong:
        raise BoxLabelError(f"{wrong} foto(s) com número de caixas diferente do registro.")
    return photos, boxes
