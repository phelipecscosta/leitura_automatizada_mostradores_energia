"""Testes do orquestrador, com triagem e detector falsos e imagens sintéticas (sem dados do cliente)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from meter_reader.contracts import (REJECTION_NO_FIELD, TRIAGE_REJECTIONS, ControlRow,
                                    ExtractionBatch, ExtractionRow, Reason, Status)
from meter_reader.detector import DetectionResult
from meter_reader.fields import AMBIGUOUS_SAME_CLASS, FieldSelection
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


def detection(passes=True, family="digital", ambiguity=None) -> DetectionResult:
    """Resultado de detector falso: uma caixa de display, com a família e a ambiguidade dadas."""
    box = np.array([[10, 10, 40, 20]], np.float32)
    fields = None
    if passes:
        consumption = None if ambiguity or family is None else box[0]
        fields = FieldSelection(family, consumption, None, ambiguity)
    return DetectionResult(box, np.array([0.9 if passes else 0.001], np.float32),
                           np.array([1 if family == "digital" else 2]), 0.9 if passes else 0.001,
                           passes, family if passes else None, fields)


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


class FakeDetector:
    """Detector falso: devolve o resultado indicado por foto (padrão: passa, digital)."""

    def __init__(self, results: dict[str, DetectionResult] | None = None):
        self.results = results or {}
        self.calls: list[str] = []

    def predict(self, path: Path) -> DetectionResult:
        self.calls.append(path.name)
        return self.results.get(path.name, detection())


class SpyReading(ProvisionalReading):
    """Provisório que registra a foto, o tipo e a detecção recebidos."""

    def __init__(self):
        self.calls: list[tuple[str, str]] = []
        self.detections: list[DetectionResult] = []

    def read(self, path, row, meter_type, detection):
        self.calls.append((path.name, meter_type))
        self.detections.append(detection)
        return super().read(path, row, meter_type, detection)


def _one(tmp_path, triage=None, detector=None, reader=None) -> ControlRow:
    img = _noise_image(tmp_path / "a.jpg")
    return process_images([img], _batch("a.jpg"), triage or FakeTriage(),
                          detector or FakeDetector(), reader or SpyReading())[0]


def test_each_stage_rejects_and_stops(tmp_path) -> None:
    black = _black_image(tmp_path / "preta.jpg")      # listada, mas vazia
    orphan = _noise_image(tmp_path / "orfa.jpg")      # íntegra, sem linha
    other = _noise_image(tmp_path / "poste.jpg")      # listada, triagem rejeita
    blank = _noise_image(tmp_path / "sem_campo.jpg")  # listada, segunda porta rejeita
    meter = _noise_image(tmp_path / "medidor.jpg")    # listada, segue
    triage, reading = FakeTriage({"poste.jpg": "outros"}), SpyReading()
    detector = FakeDetector({"sem_campo.jpg": detection(passes=False)})

    rows = process_images([black, orphan, other, blank, meter],
                          _batch("preta.jpg", "poste.jpg", "sem_campo.jpg", "medidor.jpg"),
                          triage, detector, reading)

    assert [r.status for r in rows] == [
        Status.REJECTED_CORRUPTED, Status.REJECTED_NOT_LISTED, Status.REJECTED_OTHER,
        Status.REJECTED_LOW_QUALITY, Status.NEEDS_REVIEW]
    # Cada etapa só vê as fotos que passaram pelas anteriores
    assert triage.calls == ["poste.jpg", "sem_campo.jpg", "medidor.jpg"]
    assert detector.calls == ["sem_campo.jpg", "medidor.jpg"]
    assert reading.calls == [("medidor.jpg", "digital")]


def test_corrupted_wins_over_orphan(tmp_path) -> None:
    # Vazia e órfã: o motivo gravado é o primeiro na ordem do pipeline
    row = process_images([_black_image(tmp_path / "x.jpg")], _batch("y.jpg"),
                         FakeTriage(), FakeDetector(), SpyReading())[0]
    assert row.reason is Reason.CORRUPTED


def test_unreadable_file_is_corrupted(tmp_path) -> None:
    bad = tmp_path / "quebrada.jpg"
    bad.write_bytes(b"isto nao e um jpeg")
    row = process_images([bad], _batch("quebrada.jpg"), FakeTriage(), FakeDetector(),
                         SpyReading())[0]
    assert row.status is Status.REJECTED_CORRUPTED


@pytest.mark.parametrize("dest", ["indeterminado", "multiplos", "outros"])
def test_triage_rejection_row(tmp_path, dest) -> None:
    row = _one(tmp_path, triage=FakeTriage({"a.jpg": dest}))
    assert row == ControlRow.rejected("a.jpg", TRIAGE_REJECTIONS[dest])


def test_second_gate_rejection_row(tmp_path) -> None:
    row = _one(tmp_path, detector=FakeDetector({"a.jpg": detection(passes=False)}))
    assert row == ControlRow.rejected("a.jpg", REJECTION_NO_FIELD)
    assert row.destination == ("Rejeitadas", "Baixa qualidade")


def test_type_mismatch_goes_to_review(tmp_path) -> None:
    row = _one(tmp_path, triage=FakeTriage(meter_type="ciclometrico"))  # detector: digital
    assert (row.status, row.reason) == (Status.NEEDS_REVIEW, Reason.TYPE_MISMATCH)


def test_type_mismatch_wins_over_kvarh_ambiguity(tmp_path) -> None:
    det = detection(family="ciclometrico", ambiguity=AMBIGUOUS_SAME_CLASS)
    row = _one(tmp_path, triage=FakeTriage(meter_type="digital"),
               detector=FakeDetector({"a.jpg": det}))
    assert row.reason is Reason.TYPE_MISMATCH


def test_kvarh_ambiguity_goes_to_review(tmp_path) -> None:
    reader = SpyReading()
    det = detection(family="ciclometrico", ambiguity=AMBIGUOUS_SAME_CLASS)
    row = _one(tmp_path, triage=FakeTriage(meter_type="ciclometrico"),
               detector=FakeDetector({"a.jpg": det}), reader=reader)
    assert (row.status, row.reason) == (Status.NEEDS_REVIEW, Reason.CONSUMPTION_NOT_IDENTIFIED)
    assert reader.calls == []  # o leitor não é chamado


def test_only_serial_number_reaches_the_reader(tmp_path) -> None:
    # Só número de série: sem comparação de tipo (B1); o caminho é do leitor (ET6)
    reader = SpyReading()
    row = _one(tmp_path, triage=FakeTriage(meter_type="ciclometrico"),
               detector=FakeDetector({"a.jpg": detection(family=None)}), reader=reader)
    assert reader.calls == [("a.jpg", "ciclometrico")] and row.reason is Reason.LOW_CONFIDENCE


def test_reader_receives_the_detection(tmp_path) -> None:
    reader, det = SpyReading(), detection(family="ciclometrico")
    _one(tmp_path, triage=FakeTriage(meter_type="ciclometrico"),
         detector=FakeDetector({"a.jpg": det}), reader=reader)
    assert reader.detections == [det]


def test_provisional_never_approves(tmp_path) -> None:
    row = _one(tmp_path, reader=ProvisionalReading())
    assert (row.status, row.reason, row.destination) == (
        Status.NEEDS_REVIEW, Reason.LOW_CONFIDENCE, ("Verificar",))


def test_approval_path_with_fake_reader(tmp_path) -> None:
    # O caminho de aprovação existe no contrato, mesmo sem uso no esqueleto
    class ApprovingReading:
        version = "falso"

        def read(self, path, row, meter_type, detection):
            return ControlRow(path.name, row.expected_meter_number, "03", row.reading,
                              Status.APPROVED, None)

    row = _one(tmp_path, reader=ApprovingReading())
    assert row.destination == ("Aprovadas",) and row.meter_number == "A1"


def test_reading_stage_cannot_return_another_photo(tmp_path) -> None:
    class WrongReading:
        version = "falso"

        def read(self, path, row, meter_type, detection):
            return ControlRow("outra.jpg", None, None, None,
                              Status.NEEDS_REVIEW, Reason.LOW_CONFIDENCE)

    with pytest.raises(ValueError, match="outra foto"):
        _one(tmp_path, reader=WrongReading())
