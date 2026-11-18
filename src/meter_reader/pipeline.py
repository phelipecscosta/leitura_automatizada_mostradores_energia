"""Orquestrador do motor (spec v18, 2.5, itens 2 a 14).

Encadeia, para cada imagem: integridade -> pareamento -> triagem -> detector
(segunda porta, Validador 2 e regra do kvarh) -> leitor e validadores. A primeira
etapa que rejeita ou manda para verificação encerra a imagem; como as etapas seguem
a ordem do pipeline, o motivo gravado é sempre o primeiro que se aplica (spec 2.5,
item 1.7).

O orquestrador não acessa a planilha nem move arquivos: só decide a linha de cada
imagem. Os estágios entram por injeção, para que os testes usem versões falsas e
para que o leitor provisório seja trocado na ET6 sem mexer aqui (E70, tomada
dividida na ET5c). O filtro de Parâmetros (item 9) fica fora do esqueleto (E08; ET8).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Protocol

from meter_reader.contracts import (REJECTION_NO_FIELD, ControlRow, ExtractionBatch,
                                    ExtractionRow, PairingResult, Reason, Status)
from meter_reader.detector import DetectionResult
from meter_reader.integrity import check_integrity
from meter_reader.pairing import index_by_photo
from meter_reader.triage import TriageResult


class TriageStage(Protocol):
    """Triagem de cinco classes (spec 2.5, itens 5 a 8)."""

    def predict(self, path: Path) -> TriageResult: ...


class DetectorStage(Protocol):
    """Detector de campos (spec 2.5, itens 10 a 12)."""

    def predict(self, path: Path) -> DetectionResult: ...


class ReaderStage(Protocol):
    """Leitor e validadores (spec 2.5, itens 11, 13 e 14), numa tomada até a ET6."""

    version: str  # vai para a aba Execução

    def read(self, path: Path, row: ExtractionRow, meter_type: str,
             detection: DetectionResult) -> ControlRow: ...


class ProvisionalReading:
    """Leitor provisório fail-safe (D2 b): toda imagem que chega a ele vai para verificação.

    Não lê nada; existe para que o veículo funcione de ponta a ponta antes do leitor
    (ET6). Nunca aprova uma imagem.
    """

    version = "provisorio"

    def read(self, path: Path, row: ExtractionRow, meter_type: str,
             detection: DetectionResult) -> ControlRow:
        return ControlRow(Path(path).name, None, None, None,
                          Status.NEEDS_REVIEW, Reason.LOW_CONFIDENCE)


def _review(name: str, reason: Reason) -> ControlRow:
    return ControlRow(name, None, None, None, Status.NEEDS_REVIEW, reason)


def detector_decision(name: str, detection: DetectionResult, meter_type: str) -> ControlRow | None:
    """Decisões do detector, na ordem do pipeline; None: a imagem segue para o leitor.

    1. Segunda porta (item 10.2): nenhuma caixa, ou nenhuma acima do limiar.
    2. Validador 2 (item 12): família do display mais confiante diferente do tipo da
       triagem; só número de série não é comparado.
    3. Regra do kvarh (item 10.1): display de consumo não identificado.
    """
    if not detection.passes_gate:
        return ControlRow.rejected(name, REJECTION_NO_FIELD)
    if detection.family is not None and detection.family != meter_type:
        return _review(name, Reason.TYPE_MISMATCH)
    if detection.fields is not None and detection.fields.ambiguity is not None:
        return _review(name, Reason.CONSUMPTION_NOT_IDENTIFIED)
    return None


def process_image(path: Path, index: Mapping[str, ExtractionRow], triage: TriageStage,
                  detector: DetectorStage, reader: ReaderStage) -> ControlRow:
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

    detection = detector.predict(path)           # itens 10 a 12
    decided = detector_decision(name, detection, result.meter_type)
    if decided is not None:
        return decided

    out = reader.read(path, pairing.row, result.meter_type, detection)  # itens 11, 13 e 14
    if out.photo_name != name:
        raise ValueError("O estágio de leitura devolveu a linha de outra foto.")
    return out


def process_images(paths: Iterable[Path], batch: ExtractionBatch, triage: TriageStage,
                   detector: DetectorStage, reader: ReaderStage) -> list[ControlRow]:
    """Decide as linhas de todas as imagens, na ordem recebida."""
    index = index_by_photo(batch)
    return [process_image(p, index, triage, detector, reader) for p in paths]
