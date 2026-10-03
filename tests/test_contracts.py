"""Testes dos contratos: o vocabulário precisa ser idêntico ao da spec."""

from __future__ import annotations

import pytest

from meter_reader.contracts import Reason, Status, first_reason

# Cópia literal da spec (2.5, itens 1.5 e 1.6). É o oráculo independente:
# se alguém alterar um texto ou a ordem no código, o teste falha.
SPEC_STATUS = [
    "Processada com sucesso",
    "Necessita de verificação",
    "Rejeitada - Outros",
    "Rejeitada - Baixa qualidade",
    "Rejeitada - Corrompida",
    "Rejeitada - Não listada em BaseExtracao",
    "Não processada",
]
SPEC_REASONS_IN_PIPELINE_ORDER = [
    "Arquivo corrompido ou vazio",
    "Sem registro na tabela BaseExtracao",
    "Não é medidor",
    "Ilegível",
    "Não listado na tabela fornecida com os parâmetros",
    "Tipo divergente entre triagem e detector",
    "Confiança baixa",
    "Número do medidor divergente do registro",
    "Leitura de consumo divergente do registro",
    "Função incorreta",
    "Cliente de alta criticidade",
    "Leitura fora dos Parâmetros",
]


def test_status_texts_match_spec() -> None:
    assert [s.value for s in Status] == SPEC_STATUS


def test_reason_texts_and_order_match_spec() -> None:
    assert [r.value for r in Reason] == SPEC_REASONS_IN_PIPELINE_ORDER


def test_enum_members_compare_equal_to_text() -> None:
    # StrEnum: o membro vale como texto, pronto para gravar na planilha
    assert Status.REJECTED_CORRUPTED == "Rejeitada - Corrompida"


def test_first_reason_follows_pipeline_order() -> None:
    # A ordem de entrada não importa; vale a ordem do pipeline
    reasons = [Reason.OUT_OF_PARAMETERS, Reason.LOW_CONFIDENCE, Reason.WRONG_FUNCTION]
    assert first_reason(reasons) is Reason.LOW_CONFIDENCE


def test_first_reason_requires_at_least_one() -> None:
    with pytest.raises(ValueError):
        first_reason([])


# ---------------------------------------------------------------------
# Registros do BaseExtracao (dados sintéticos, nunca do cliente)
# ---------------------------------------------------------------------
import dataclasses
from datetime import date

from meter_reader.contracts import ExtractionBatch, ExtractionRow


def _row(**overrides) -> ExtractionRow:
    """Linha sintética válida; cada teste altera só o que precisa."""
    fields = dict(
        meter_numbers=("B12345", "99887766"),
        reading=123,
        note=None,
        photo_name="LOTE_TESTE_00000000000000000001_000.jpg",
    )
    fields.update(overrides)
    return ExtractionRow(**fields)


def test_expected_meter_is_leftmost_part() -> None:
    assert _row().expected_meter_number == "B12345"


def test_row_is_immutable() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        _row().reading = 0


def test_row_without_photo_is_valid() -> None:
    # Linhas "NA" na coluna Foto são preservadas para a análise da ET1
    assert _row(photo_name=None).photo_name is None


@pytest.mark.parametrize("parts", [(), ("B 12345",), ("",), ("B-123",)])
def test_invalid_meter_numbers_are_rejected(parts) -> None:
    with pytest.raises(ValueError):
        _row(meter_numbers=parts)


@pytest.mark.parametrize("value", ["9999", "A1"])
def test_short_meter_numbers_are_accepted(value) -> None:
    # Opção B: sem tamanho fixo (fato F5: há números reais com 4 e 5 caracteres)
    assert _row(meter_numbers=(value,)).expected_meter_number == value


@pytest.mark.parametrize(
    ("reading", "error"),
    [(-1, ValueError), ("123", TypeError), (True, TypeError), (12.5, TypeError)],
)
def test_invalid_readings_are_rejected(reading, error) -> None:
    with pytest.raises(error):
        _row(reading=reading)


def test_zero_reading_is_valid() -> None:
    # Fato F6: há 347 leituras iguais a 0 no CSV real
    assert _row(reading=0).reading == 0


@pytest.mark.parametrize("field", ["note", "photo_name"])
@pytest.mark.parametrize("value", ["NA", ""])
def test_missing_marker_must_be_none(field, value) -> None:
    with pytest.raises(ValueError):
        _row(**{field: value})


def test_batch_keeps_date_and_rows() -> None:
    batch = ExtractionBatch(batch_date=date(2000, 1, 1), rows=(_row(),))
    assert batch.batch_date.year == 2000 and len(batch.rows) == 1
