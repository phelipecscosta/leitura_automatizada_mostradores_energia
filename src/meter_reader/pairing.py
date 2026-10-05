"""Pareamento das imagens com o BaseExtracao (spec 2.5, item 2.4).

Funções puras: recebem nomes de arquivo e o lote já lido, sem acessar o
disco. A comparação é pelo nome exato, inclusive maiúsculas (fato F7).
"""

from __future__ import annotations

from collections.abc import Iterable

from meter_reader.contracts import ExtractionBatch, ExtractionRow, PairingResult


def index_by_photo(batch: ExtractionBatch) -> dict[str, ExtractionRow]:
    """Mapeia o nome de cada foto listada para a sua linha."""
    index: dict[str, ExtractionRow] = {}
    for row in batch.rows:
        if row.photo_name is None:
            continue  # linha sem foto: não participa do pareamento
        if row.photo_name in index:
            # O leitor já impede isso; aqui a garantia fica explícita
            raise ValueError("Nome de foto repetido no lote: pareamento ambíguo.")
        index[row.photo_name] = row
    return index


def pair_images(photo_names: Iterable[str], batch: ExtractionBatch) -> list[PairingResult]:
    """Associa cada imagem à sua linha; sem linha, a imagem é órfã."""
    index = index_by_photo(batch)
    return [PairingResult(name, index.get(name)) for name in photo_names]


def rows_without_image(
    photo_names: Iterable[str], batch: ExtractionBatch
) -> list[ExtractionRow]:
    """Linhas com foto listada cujo arquivo não está entre as imagens recebidas."""
    present = set(photo_names)
    return [
        row
        for row in batch.rows
        if row.photo_name is not None and row.photo_name not in present
    ]
