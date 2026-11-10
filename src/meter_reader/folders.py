"""Pastas de saída do lote e movimentação das imagens (spec v14, 2.3 e 2.5.2.2; D1 b).

A raiz "Imagens processadas" recebe uma subpasta por lote (DDMMAAAA), com a
estrutura da spec 2.3. Cada imagem vai para a pasta do seu status; as de
status "Não processada" ficam na origem. Um arquivo de mesmo nome no destino
é sobrescrito. Nenhuma função daqui apaga uma foto que não tenha acabado de
ser copiada para o destino.
"""

from __future__ import annotations

import errno
import os
import shutil
from collections.abc import Iterable
from datetime import date
from pathlib import Path

from meter_reader.contracts import DESTINATION_FOLDERS, ControlRow

_ERROR_NOT_SAME_DEVICE = 17  # código do Windows para mover entre discos


def lot_dir(output_root: Path | str, batch_date: date) -> Path:
    """Pasta do lote dentro da raiz: <raiz>/DDMMAAAA."""
    return Path(output_root) / f"{batch_date:%d%m%Y}"


def destination_dirs(lot: Path | str) -> list[Path]:
    """Todas as pastas de destino do lote, na ordem da spec 2.3."""
    return [Path(lot).joinpath(*parts) for parts in DESTINATION_FOLDERS.values()
            if parts is not None]


def create_structure(lot: Path | str) -> None:
    """Cria a estrutura da spec 2.3 dentro da pasta do lote (sem erro se já existir)."""
    for folder in destination_dirs(lot):
        folder.mkdir(parents=True, exist_ok=True)


def _move(src: Path, dst: Path) -> None:
    """Move com sobrescrita; entre discos, copia, troca no destino e só então apaga a origem."""
    try:
        os.replace(src, dst)
        return
    except OSError as exc:
        cross_device = (exc.errno == errno.EXDEV
                        or getattr(exc, "winerror", None) == _ERROR_NOT_SAME_DEVICE)
        if not cross_device:
            raise
    tmp = dst.with_name(f"~tmp_{dst.name}")
    try:
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)  # mesmo disco: troca indivisível
    finally:
        tmp.unlink(missing_ok=True)
    src.unlink()  # a cópia já está completa no destino


def move_images(source_dir: Path | str, rows: Iterable[ControlRow],
                lot: Path | str) -> list[Path]:
    """Move cada imagem para a pasta do seu status. Devolve os caminhos de destino."""
    source_dir, lot = Path(source_dir), Path(lot)
    create_structure(lot)
    moved = []
    for row in rows:
        if row.destination is None:
            continue  # "Não processada": fica na origem
        dst = lot.joinpath(*row.destination, row.photo_name)
        _move(source_dir / row.photo_name, dst)
        moved.append(dst)
    return moved


def stale_copies(rows: Iterable[ControlRow], lot: Path | str) -> list[Path]:
    """Cópias de mesmo nome em outra pasta do lote que não a do status atual (opção c).

    Só lê: nada é apagado. O resumo avisa o utilizador.
    """
    lot = Path(lot)
    folders = destination_dirs(lot)
    found = []
    for row in rows:
        current = None if row.destination is None else lot.joinpath(*row.destination)
        found.extend(folder / row.photo_name for folder in folders
                     if folder != current and (folder / row.photo_name).exists())
    return found



def stale_list_name(batch_date: date) -> str:
    """Nome do arquivo com a lista das cópias antigas do lote."""
    return f"Copias_antigas_{batch_date:%d%m%Y}.txt"


def stale_by_folder(paths: Iterable[Path], lot: Path | str) -> dict[str, int]:
    """Contagem das cópias antigas por pasta, relativa à pasta do lote (vai para o resumo)."""
    lot = Path(lot)
    counts: dict[str, int] = {}
    for p in paths:
        folder = Path(p).parent.relative_to(lot).as_posix()
        counts[folder] = counts.get(folder, 0) + 1
    return counts


def write_stale_list(lot: Path | str, paths: Iterable[Path], batch_date: date) -> Path | None:
    """Grava a lista das cópias antigas na pasta do lote (opção b); sem cópias, apaga a lista anterior.

    [Dados sensíveis] O arquivo contém nomes reais de fotos: fica só na pasta
    de saída, que nunca pode estar dentro de um repositório.
    """
    lot = Path(lot)
    target = lot / stale_list_name(batch_date)
    paths = list(paths)
    if not paths:
        target.unlink(missing_ok=True)  # nunca deixa uma lista desatualizada
        return None
    lines = sorted(Path(p).relative_to(lot).as_posix() for p in paths)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target
