"""Testes da planilha de controle (só linhas sintéticas, em pasta temporária)."""

from __future__ import annotations

from datetime import date, datetime

import pytest
from openpyxl import load_workbook

from meter_reader import control_sheet as cs
from meter_reader.contracts import (REJECTION_CORRUPTED, TRIAGE_REJECTIONS, ControlRow,
                                    Reason, Status, wrong_function)

REVIEW = ControlRow("b.jpg", None, None, None, Status.NEEDS_REVIEW, Reason.LOW_CONFIDENCE)
APPROVED = ControlRow("a.jpg", "00123A", "03", 4567, Status.APPROVED, None)
WRONG_FN = ControlRow("c.jpg", "A1", wrong_function("103"), 12,
                      Status.NEEDS_REVIEW, Reason.WRONG_FUNCTION)
CORRUPTED = ControlRow.rejected("d.jpg", REJECTION_CORRUPTED)
OTHER = ControlRow.rejected("e.jpg", TRIAGE_REJECTIONS["outros"])
ROWS = [APPROVED, REVIEW, WRONG_FN, CORRUPTED, OTHER]


def _record(kind=cs.RunKind.FULL, hour=10) -> cs.ExecutionRecord:
    return cs.ExecutionRecord(datetime(2026, 11, 10, hour, 0, 0), kind, "CPU",
                              "triagem C4 ee4fc0309027; leitura provisorio",
                              cs.PARAMETERS_DISABLED)


def test_file_name_uses_batch_date() -> None:
    assert cs.control_file_name(date(2026, 5, 20)) == "Controle_20052026.xlsx"


def test_round_trip_keeps_every_field(tmp_path) -> None:
    path = tmp_path / "Controle_20052026.xlsx"
    cs.write_control(path, ROWS, [_record()])
    rows, executions = cs.read_control(path)
    assert rows == ROWS
    assert executions == [_record()]


def test_cells_follow_spec_text(tmp_path) -> None:
    path = tmp_path / "c.xlsx"
    cs.write_control(path, ROWS, [_record()])
    wb = load_workbook(path)
    assert wb.sheetnames == ["Controle", "Execução"]
    control = list(wb["Controle"].iter_rows(values_only=True))
    assert control[0] == cs.CONTROL_COLUMNS
    # Aprovada: motivo vazio; número como texto, com zeros à esquerda
    assert control[1] == ("a.jpg", "00123A", "03", 4567, "Processada com sucesso", None)
    # Rejeitada: campos extraídos com o texto NAN
    assert control[4] == ("d.jpg", "NAN", "NAN", "NAN", "Rejeitada - Corrompida",
                          "Arquivo corrompido ou vazio")
    execution = list(wb["Execução"].iter_rows(values_only=True))
    assert execution[0] == cs.EXECUTION_COLUMNS
    assert execution[1][1:] == ("completo", "CPU", "triagem C4 ee4fc0309027; leitura provisorio",
                                "desativada")


def test_merge_replaces_in_place_and_appends() -> None:
    new_b = ControlRow.rejected("b.jpg", TRIAGE_REJECTIONS["multiplos"])
    new_f = ControlRow.rejected("f.jpg", TRIAGE_REJECTIONS["indeterminado"])
    merged = cs.merge_rows(ROWS, [new_f, new_b])
    assert [r.photo_name for r in merged] == ["a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg", "f.jpg"]
    assert merged[1] == new_b and merged[5] == new_f
    assert merged[0] is APPROVED  # linhas não reprocessadas ficam intactas


def test_merge_refuses_repeated_photo() -> None:
    with pytest.raises(ValueError, match="repetida"):
        cs.merge_rows([], [REVIEW, REVIEW])


def test_save_full_then_complementary(tmp_path) -> None:
    path = tmp_path / "c.xlsx"
    cs.save_control(path, ROWS, _record())
    new_b = ControlRow.rejected("b.jpg", TRIAGE_REJECTIONS["outros"])
    cs.save_control(path, [new_b], _record(cs.RunKind.COMPLEMENTARY, hour=11))
    rows, executions = cs.read_control(path)
    assert rows == [APPROVED, new_b, WRONG_FN, CORRUPTED, OTHER]
    assert [e.kind for e in executions] == [cs.RunKind.FULL, cs.RunKind.COMPLEMENTARY]


def test_kind_must_match_existing_file(tmp_path) -> None:
    path = tmp_path / "c.xlsx"
    with pytest.raises(cs.ControlFileError, match="complementar"):
        cs.save_control(path, ROWS, _record(cs.RunKind.COMPLEMENTARY))
    cs.save_control(path, ROWS, _record())
    with pytest.raises(cs.ControlFileError, match="complementar"):
        cs.save_control(path, ROWS, _record())


def test_excel_edits_are_accepted(tmp_path) -> None:
    # Leitura digitada como 12.0 e linha em branco deixada pelo Excel
    path = tmp_path / "c.xlsx"
    cs.write_control(path, [WRONG_FN], [_record()])
    wb = load_workbook(path)
    wb["Controle"]["D2"] = 12.0
    wb["Controle"].append([None] * 6)
    wb.save(path)
    assert cs.read_control(path)[0] == [WRONG_FN]


@pytest.mark.parametrize("cell, value", [("E3", "Aprovada"), ("D3", "doze"),
                                         ("F2", "Confiança baixa"), ("A3", "a.jpg")])
def test_invalid_cell_points_to_line_without_value(tmp_path, cell, value) -> None:
    path = tmp_path / "c.xlsx"
    cs.write_control(path, [APPROVED, WRONG_FN], [_record()])
    wb = load_workbook(path)
    wb["Controle"][cell] = value
    wb.save(path)
    with pytest.raises(cs.ControlFileError) as info:
        cs.read_control(path)
    message = str(info.value)
    assert value not in message
    assert "linha" in message or "repetido" in message


def test_changed_header_is_refused(tmp_path) -> None:
    path = tmp_path / "c.xlsx"
    cs.write_control(path, ROWS, [_record()])
    wb = load_workbook(path)
    wb["Controle"]["B1"] = "Medidor"
    wb.save(path)
    with pytest.raises(cs.ControlFileError, match="cabeçalho"):
        cs.read_control(path)


def test_locked_file_keeps_original(tmp_path, monkeypatch) -> None:
    # Simula a planilha aberta no Excel: a troca final falha
    path = tmp_path / "c.xlsx"
    cs.write_control(path, ROWS, [_record()])
    before = path.read_bytes()

    def locked(src, dst):
        raise PermissionError

    monkeypatch.setattr(cs.os, "replace", locked)
    with pytest.raises(cs.ControlFileError, match="aberta"):
        cs.write_control(path, [REVIEW], [_record()])
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]  # nenhum temporário deixado


@pytest.mark.parametrize("field, value", [("profile", "TPU"), ("model_versions", ""),
                                          ("parameters", "ativa"), ("parameters", "sim")])
def test_execution_record_vocabulary(field, value) -> None:
    kwargs = dict(timestamp=datetime(2026, 1, 1), kind=cs.RunKind.FULL, profile="CPU",
                  model_versions="v", parameters="desativada")
    kwargs[field] = value
    with pytest.raises(ValueError):
        cs.ExecutionRecord(**kwargs)


def test_parameters_active_with_file_name() -> None:
    cs.ExecutionRecord(datetime(2026, 1, 1), cs.RunKind.FULL, "GPU", "v",
                       "ativa (Parametros.xlsx)")
