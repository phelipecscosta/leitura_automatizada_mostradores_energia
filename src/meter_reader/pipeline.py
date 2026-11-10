"""Orquestrador do motor (spec v14, 2.5, itens 2 a 14).

Encadeia, para cada imagem: integridade -> pareamento -> triagem -> estágio
de leitura (detector, leitor e validadores). A primeira etapa que rejeita
encerra a imagem; como as etapas seguem a ordem do pipeline, o motivo
gravado é sempre o primeiro que se aplica (spec 2.5, item 1.7).

O orquestrador não acessa a planilha nem move arquivos: só decide a linha
de cada imagem. Os estágios entram por injeção, para que os testes usem
versões falsas e para que o provisório seja trocado sem mexer aqui.
O filtro de Parâmetros (item 9) fica fora do esqueleto (E08; ET8).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Protocol

from meter_reader.contracts import (ControlRow, ExtractionBatch, ExtractionRow,
                                    PairingResult, Reason, Status)
from meter_reader.integrity import check_integrity
from meter_reader.pairing import index_by_photo
from meter_reader.triage import TriageResult


class TriageStage(Protocol):
    """Triagem de cinco classes (spec 2.5, itens 5 a 8)."""

    def predict(self, path: Path) -> TriageResult: ...


class ReadingStage(Protocol):
    """Detector, leitor e validadores (spec 2.5, itens 10 a 14), numa única tomada."""

    version: str  # vai para a aba Execução

    def read(self, path: Path, row: ExtractionRow, meter_type: str) -> ControlRow: ...


class ProvisionalReading:
    """Provisório fail-safe (D2 b): toda imagem que passa pela triagem vai para verificação.

    Não lê nada; existe para que o veículo funcione de ponta a ponta antes do
    detector e do leitor (ET5 e ET6). Nunca aprova uma imagem.
    """

    version = "provisorio"

    def read(self, path: Path, row: ExtractionRow, meter_type: str) -> ControlRow:
        return ControlRow(Path(path).name, None, None, None,
                          Status.NEEDS_REVIEW, Reason.LOW_CONFIDENCE)


def process_image(path: Path, index: Mapping[str, ExtractionRow],
                  triage: TriageStage, reading: ReadingStage) -> ControlRow:
    """Decide a linha da planilha de uma imagem."""
    path = Path(path)
    name = path.name

    integrity = check_integrity(path)            # item 2.3
    if integrity.rejection is not None:
        return ControlRow.rejected(name, integrity.rejection)

    pairing = PairingResult(name, index.get(name))  # item 2.4
    if pairing.rejection is not None:
        return ControlRow.rejected(name, pairing.rejection)

    result = triage.predict(path)                # itens 5 a 8
    if result.rejection is not None:
        return ControlRow.rejected(name, result.rejection)

    out = reading.read(path, pairing.row, result.meter_type)  # itens 10 a 14
    if out.photo_name != name:
        raise ValueError("O estágio de leitura devolveu a linha de outra foto.")
    return out


def process_images(paths: Iterable[Path], batch: ExtractionBatch,
                   triage: TriageStage, reading: ReadingStage) -> list[ControlRow]:
    """Decide as linhas de todas as imagens, na ordem recebida."""
    index = index_by_photo(batch)
    return [process_image(p, index, triage, reading) for p in paths]
