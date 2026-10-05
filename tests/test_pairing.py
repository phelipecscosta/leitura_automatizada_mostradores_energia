"""Testes do pareamento (dados sintéticos, sem acesso a disco)."""

from __future__ import annotations

from datetime import date

import pytest

from meter_reader.contracts import REJECTION_NOT_LISTED, ExtractionBatch, ExtractionRow
from meter_reader.pairing import index_by_photo, pair_images, rows_without_image

PHOTO_A = "LOTE_TESTE_00000000000000000001_000.jpg"
PHOTO_B = "LOTE_TESTE_00000000000000000002_000.jpg"
ORPHAN = "LOTE_TESTE_00000000000000000003_001.jpg"


def _row(photo: str | None, meter: str = "B12345") -> ExtractionRow:
    return ExtractionRow(meter_numbers=(meter,), reading=1, note=None, photo_name=photo)


def _batch(*rows: ExtractionRow) -> ExtractionBatch:
    return ExtractionBatch(batch_date=date(2026, 7, 3), rows=rows)


def test_listed_image_is_paired_with_its_row() -> None:
    batch = _batch(_row(PHOTO_A, meter="A11111"), _row(PHOTO_B, meter="B22222"))
    result = pair_images([PHOTO_B], batch)[0]
    assert result.row.expected_meter_number == "B22222"
    assert result.rejection is None


def test_orphan_image_is_rejected_as_not_listed() -> None:
    result = pair_images([ORPHAN], _batch(_row(PHOTO_A)))[0]
    assert result.row is None
    assert result.rejection == REJECTION_NOT_LISTED


def test_results_keep_input_order() -> None:
    batch = _batch(_row(PHOTO_A), _row(PHOTO_B))
    names = [ORPHAN, PHOTO_B, PHOTO_A]
    assert [r.photo_name for r in pair_images(names, batch)] == names


def test_rows_without_photo_do_not_take_part() -> None:
    # Linhas "NA" na coluna Foto não entram no índice
    assert index_by_photo(_batch(_row(None), _row(PHOTO_A))).keys() == {PHOTO_A}


def test_match_is_case_sensitive() -> None:
    # Comparação exata (fato F7): diferença de maiúsculas não pareia
    result = pair_images([PHOTO_A.upper()], _batch(_row(PHOTO_A)))[0]
    assert result.row is None


def test_repeated_photo_in_batch_is_rejected() -> None:
    with pytest.raises(ValueError):
        index_by_photo(_batch(_row(PHOTO_A), _row(PHOTO_A, meter="C33333")))


def test_rows_without_image_are_reported() -> None:
    batch = _batch(_row(PHOTO_A), _row(PHOTO_B), _row(None))
    missing = rows_without_image([PHOTO_A], batch)
    assert [r.photo_name for r in missing] == [PHOTO_B]
