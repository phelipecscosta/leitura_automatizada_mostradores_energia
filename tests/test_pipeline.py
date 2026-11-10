"""Testes do orquestrador, com triagem falsa e imagens sintéticas (sem dados do cliente)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from meter_reader.contracts import (TRIAGE_REJECTIONS, ControlRow, ExtractionBatch,
                                    ExtractionRow, Reason, Status)
from meter_reader.pipeline import ProvisionalReading, process_images
from meter_reader.triage import TriageResult


def _noise_image(path: Path) -> Path:
    # Imagem com conteúdo (ruído): passa na integridade
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)).save(path)
    return path


def _black_image(path: Path) -> Path:
    # Imagem vazia (preta): reprovada na integridade
    Image.new("RGB", (64, 48)).save(path)
    return path


def _batch(*names: str) -> ExtractionBatch:
    rows = tuple(ExtractionRow(("A1",), 10, None, n) for n in names)
    return ExtractionBatch(date(2026, 5, 20), rows)


class FakeTriage:
    """Triagem falsa: rejeita as fotos indicadas e registra quem foi chamado."""

    def __init__(self, rejected: dict[str, str] | None = None, meter_type: str = "digital"):
        self.rejected = rejected or {}
        self.meter_type = meter_type
        self.calls: list[str] = []

    def predict(self, path: Path) -> TriageResult:
        self.calls.append(path.name)
        dest = self.rejected.get(path.name)
        if dest is not None:
            return TriageResult((0, 0, 0, 0, 1), 1.0, TRIAGE_REJECTIONS[dest], None)
        return TriageResult((1, 0, 0, 0, 0), 0.0, None, self.meter_type)


class SpyReading(ProvisionalReading):
    """Provisório que registra a foto e o tipo recebidos."""

    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def read(self, path, row, meter_type):
        self.calls.append((path.name, meter_type))
        return super().read(path, row, meter_type)


def test_each_stage_rejects_and_stops(tmp_path) -> None:
    black = _black_image(tmp_path / "preta.jpg")      # listada, mas vazia
    orphan = _noise_image(tmp_path / "orfa.jpg")      # íntegra, sem linha
    other = _noise_image(tmp_path / "poste.jpg")      # listada, triagem rejeita
    meter = _noise_image(tmp_path / "medidor.jpg")    # listada, segue
    triage, reading = FakeTriage({"poste.jpg": "outros"}), SpyReading()

    rows = process_images([black, orphan, other, meter],
                          _batch("preta.jpg", "poste.jpg", "medidor.jpg"), triage, reading)

    assert [r.status for r in rows] == [
        Status.REJECTED_CORRUPTED, Status.REJECTED_NOT_LISTED,
        Status.REJECTED_OTHER, Status.NEEDS_REVIEW]
    # A foto vazia e a órfã nunca chegam à triagem; só a que passa chega à leitura
    assert triage.calls == ["poste.jpg", "medidor.jpg"]
    assert reading.calls == [("medidor.jpg", "digital")]


def test_corrupted_wins_over_orphan(tmp_path) -> None:
    # Vazia e órfã: o motivo gravado é o primeiro na ordem do pipeline
    row = process_images([_black_image(tmp_path / "x.jpg")], _batch("y.jpg"),
                         FakeTriage(), SpyReading())[0]
    assert row.reason is Reason.CORRUPTED


def test_unreadable_file_is_corrupted(tmp_path) -> None:
    bad = tmp_path / "quebrada.jpg"
    bad.write_bytes(b"isto nao e um jpeg")
    row = process_images([bad], _batch("quebrada.jpg"), FakeTriage(), SpyReading())[0]
    assert row.status is Status.REJECTED_CORRUPTED


@pytest.mark.parametrize("dest", ["indeterminado", "multiplos", "outros"])
def test_triage_rejection_row(tmp_path, dest) -> None:
    img = _noise_image(tmp_path / "a.jpg")
    row = process_images([img], _batch("a.jpg"), FakeTriage({"a.jpg": dest}), SpyReading())[0]
    assert row == ControlRow.rejected("a.jpg", TRIAGE_REJECTIONS[dest])


def test_meter_type_reaches_reading_stage(tmp_path) -> None:
    img = _noise_image(tmp_path / "a.jpg")
    reading = SpyReading()
    process_images([img], _batch("a.jpg"), FakeTriage(meter_type="ciclometrico"), reading)
    assert reading.calls == [("a.jpg", "ciclometrico")]


def test_provisional_never_approves(tmp_path) -> None:
    img = _noise_image(tmp_path / "a.jpg")
    row = process_images([img], _batch("a.jpg"), FakeTriage(), ProvisionalReading())[0]
    assert (row.status, row.reason, row.destination) == (
        Status.NEEDS_REVIEW, Reason.LOW_CONFIDENCE, ("Verificar",))


def test_approval_path_with_fake_reader(tmp_path) -> None:
    # O caminho de aprovação existe no contrato, mesmo sem uso no esqueleto
    class ApprovingReading:
        version = "falso"

        def read(self, path, row, meter_type):
            return ControlRow(path.name, row.expected_meter_number, "03", row.reading,
                              Status.APPROVED, None)

    img = _noise_image(tmp_path / "a.jpg")
    row = process_images([img], _batch("a.jpg"), FakeTriage(), ApprovingReading())[0]
    assert row.destination == ("Aprovadas",) and row.meter_number == "A1"


def test_reading_stage_cannot_return_another_photo(tmp_path) -> None:
    class WrongReading:
        version = "falso"

        def read(self, path, row, meter_type):
            return ControlRow("outra.jpg", None, None, None,
                              Status.NEEDS_REVIEW, Reason.LOW_CONFIDENCE)

    img = _noise_image(tmp_path / "a.jpg")
    with pytest.raises(ValueError, match="outra foto"):
        process_images([img], _batch("a.jpg"), FakeTriage(), WrongReading())
