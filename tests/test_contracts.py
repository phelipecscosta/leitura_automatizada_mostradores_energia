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
    "Rejeitada - Múltiplos medidores",
    "Rejeitada - Corrompida",
    "Rejeitada - Não listada em BaseExtracao",
    "Não processada",
]
SPEC_REASONS_IN_PIPELINE_ORDER = [
    "Arquivo corrompido ou vazio",
    "Sem registro na tabela BaseExtracao",
    "Não é medidor",
    "Indeterminação do tipo de medidor",
    "Múltiplos medidores",
    "Não listado na tabela fornecida com os parâmetros",
    "Tipo divergente entre triagem e detector",
    "Mostrador de leitura de consumo não identificado",
    "Falha na extração",
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

# ---------------------------------------------------------------------
# Resultados da integridade e do pareamento
# ---------------------------------------------------------------------
from meter_reader.contracts import (
    REJECTION_CORRUPTED,
    REJECTION_NOT_LISTED,
    IntegrityIssue,
    IntegrityResult,
    PairingResult,
    Reason,
    Status,
)


def test_rejection_pairs_match_spec() -> None:
    assert REJECTION_CORRUPTED.status is Status.REJECTED_CORRUPTED
    assert REJECTION_CORRUPTED.reason is Reason.CORRUPTED
    assert REJECTION_NOT_LISTED.status is Status.REJECTED_NOT_LISTED
    assert REJECTION_NOT_LISTED.reason is Reason.NOT_IN_EXTRACTION


def test_intact_image_has_no_rejection() -> None:
    assert IntegrityResult("a.jpg", issue=None).rejection is None


@pytest.mark.parametrize("issue", list(IntegrityIssue))
def test_every_integrity_issue_maps_to_corrupted(issue) -> None:
    # Toda causa técnica vira o mesmo par na planilha
    assert IntegrityResult("a.jpg", issue=issue).rejection == REJECTION_CORRUPTED


def test_paired_image_has_no_rejection() -> None:
    row = _row()
    assert PairingResult(row.photo_name, row=row).rejection is None


def test_orphan_image_is_rejected_as_not_listed() -> None:
    assert PairingResult("orfa.jpg", row=None).rejection == REJECTION_NOT_LISTED


def test_pairing_rejects_row_of_another_photo() -> None:
    with pytest.raises(ValueError):
        PairingResult("outra.jpg", row=_row())


def test_triage_rejections_match_spec() -> None:
    from meter_reader.contracts import PRODUCT_CLASSES, TRIAGE_REJECTIONS

    # As três classes rejeitáveis são as últimas do contrato (spec v12, 2.5, item 5)
    assert tuple(TRIAGE_REJECTIONS) == PRODUCT_CLASSES[2:]
    expected = {
        "indeterminado": ("Rejeitada - Baixa qualidade", "Indeterminação do tipo de medidor"),
        "multiplos": ("Rejeitada - Múltiplos medidores", "Múltiplos medidores"),
        "outros": ("Rejeitada - Outros", "Não é medidor"),
    }
    for product_class, (status, reason) in expected.items():
        rejection = TRIAGE_REJECTIONS[product_class]
        assert (rejection.status, rejection.reason) == (status, reason)



def test_triage_groups_map_to_every_product_class_in_order():
    from meter_reader.contracts import PRODUCT_CLASSES, TRIAGE_GROUP_CLASS, TRIAGE_GROUPS
    assert len(TRIAGE_GROUPS) == len(TRIAGE_GROUP_CLASS) == 12
    assert len(set(TRIAGE_GROUPS)) == 12
    assert list(TRIAGE_GROUP_CLASS) == sorted(TRIAGE_GROUP_CLASS)  # agrupadas por classe
    assert set(TRIAGE_GROUP_CLASS) == set(range(len(PRODUCT_CLASSES)))


# ---------------------------------------------------------------------
# Linha da planilha de controle
# ---------------------------------------------------------------------

from meter_reader.contracts import (  # noqa: E402
    DESTINATION_FOLDERS, REJECTION_CORRUPTED, STATUS_REASONS, TRIAGE_REJECTIONS,
    ControlRow, wrong_function,
)


def test_every_status_and_reason_is_covered() -> None:
    # Cada status tem motivos e destino definidos, e cada motivo pertence a um só status
    assert set(STATUS_REASONS) == set(Status) == set(DESTINATION_FOLDERS)
    owners = [s for s, reasons in STATUS_REASONS.items() for _ in reasons]
    assert len(owners) == len(Reason)
    assert set().union(*STATUS_REASONS.values()) == set(Reason)


def test_only_not_processed_stays_in_origin() -> None:
    assert [s for s, d in DESTINATION_FOLDERS.items() if d is None] == [Status.NOT_PROCESSED]


def test_rejected_row_has_empty_fields_and_destination() -> None:
    row = ControlRow.rejected("a.jpg", REJECTION_CORRUPTED)
    assert (row.meter_number, row.function, row.reading) == (None, None, None)
    assert row.destination == ("Rejeitadas", "Corrompidas")


@pytest.mark.parametrize("rejection", list(TRIAGE_REJECTIONS.values()))
def test_triage_rejections_build_valid_rows(rejection) -> None:
    assert ControlRow.rejected("a.jpg", rejection).status is rejection.status


def test_reason_must_match_status() -> None:
    with pytest.raises(ValueError, match="incompatível"):
        ControlRow("a.jpg", None, None, None, Status.REJECTED_OTHER, Reason.EXTRACTION_FAILURE)
    with pytest.raises(ValueError, match="exige um motivo"):
        ControlRow("a.jpg", None, None, None, Status.NEEDS_REVIEW, None)
    with pytest.raises(ValueError, match="incompatível"):
        ControlRow("a.jpg", "A1", "03", 10, Status.APPROVED, Reason.LOW_CONFIDENCE)


def test_rejection_cannot_carry_extracted_fields() -> None:
    with pytest.raises(ValueError, match="ficam vazios"):
        ControlRow("a.jpg", "A1", None, None, Status.REJECTED_OTHER, Reason.NOT_A_METER)


def test_approved_row_rules() -> None:
    ok = ControlRow("a.jpg", "A1", "03", 12, Status.APPROVED, None)
    assert ok.destination == ("Aprovadas",)
    ControlRow("b.jpg", "A1", None, 12, Status.APPROVED, None)  # ciclométrico: função NAN
    with pytest.raises(ValueError, match="exige número"):
        ControlRow("c.jpg", None, "03", 12, Status.APPROVED, None)
    with pytest.raises(ValueError, match="função incorreta"):
        ControlRow("d.jpg", "A1", wrong_function("103"), 12, Status.APPROVED, None)


def test_review_row_accepts_wrong_function_and_missing_fields() -> None:
    ControlRow("a.jpg", "A1", wrong_function("103"), 12, Status.NEEDS_REVIEW, Reason.WRONG_FUNCTION)
    row = ControlRow("b.jpg", None, None, None, Status.NEEDS_REVIEW, Reason.LOW_CONFIDENCE)
    assert row.destination == ("Verificar",)


@pytest.mark.parametrize("value", ["3", "Função incorreta - ", "Função incorreta - 1a", "03 "])
def test_function_vocabulary_is_closed(value) -> None:
    with pytest.raises(ValueError, match="vocabulário"):
        ControlRow("a.jpg", "A1", value, 1, Status.NEEDS_REVIEW, Reason.WRONG_FUNCTION)


@pytest.mark.parametrize("reading", [-1, 1.5, True])
def test_reading_must_be_non_negative_int(reading) -> None:
    with pytest.raises(ValueError, match="inteiro"):
        ControlRow("a.jpg", "A1", "03", reading, Status.NEEDS_REVIEW, Reason.LOW_CONFIDENCE)


def test_wrong_function_rejects_non_digits() -> None:
    with pytest.raises(ValueError):
        wrong_function("1O3")

def test_new_reason_goes_to_review_after_type_mismatch() -> None:
    from meter_reader.contracts import STATUS_REASONS, REJECTION_NO_FIELD
    order = list(Reason)
    assert order.index(Reason.CONSUMPTION_NOT_IDENTIFIED) == order.index(Reason.TYPE_MISMATCH) + 1
    assert Reason.CONSUMPTION_NOT_IDENTIFIED in STATUS_REASONS[Status.NEEDS_REVIEW]
    assert (REJECTION_NO_FIELD.status, REJECTION_NO_FIELD.reason) == (
        Status.REJECTED_LOW_QUALITY, Reason.INDETERMINATE_TYPE)
