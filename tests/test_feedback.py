"""Testes do resumo em texto (linhas sintéticas)."""

from __future__ import annotations

from pathlib import Path

from meter_reader.contracts import (REJECTION_CORRUPTED, TRIAGE_REJECTIONS, ControlRow,
                                    Reason, Status)
from meter_reader.feedback import SUMMARY_LINES, Summary, format_feedback

REVIEW = [ControlRow(f"r{i}.jpg", None, None, None, Status.NEEDS_REVIEW, Reason.LOW_CONFIDENCE)
          for i in range(3)]
ROWS = REVIEW + [ControlRow.rejected("o.jpg", TRIAGE_REJECTIONS["outros"]),
                 ControlRow.rejected("c.jpg", REJECTION_CORRUPTED)]
CONTROL = Path("saida") / "20052026" / "Controle_20052026.xlsx"


def test_summary_lines_cover_every_status_once() -> None:
    assert sorted(s for s, _ in SUMMARY_LINES) == sorted(Status)


def test_counts_and_percentages() -> None:
    summary = Summary.from_rows(ROWS)
    assert summary.total == 5 and summary.by_status[Status.NEEDS_REVIEW] == 3
    text = format_feedback(summary, CONTROL, updated=False, parameters="desativada")
    assert "Processamento concluído." in text
    assert 'Planilha "Controle_20052026.xlsx" gerada' in text
    assert "100,0%" in text and "60,0%" in text and "20,0%" in text
    assert text.endswith("Validação por Parâmetros: desativada")


def test_spec_order_of_lines() -> None:
    text = format_feedback(Summary.from_rows(ROWS), CONTROL, False, "desativada")
    positions = [text.index(label) for label in
                 ["Imagens lidas"] + [label for _, label in SUMMARY_LINES]]
    assert positions == sorted(positions)


def test_updated_provisional_and_stale() -> None:
    text = format_feedback(Summary.from_rows(ROWS), CONTROL, updated=True,
                           parameters="desativada", stale_by_folder={"Verificar": 2},
                           stale_list="Copias_antigas_20052026.txt", provisional=True)
    assert "atualizada" in text and "Leitura: provisória" in text
    assert "Atenção: 2 imagens" in text and "Verificar: 2" in text
    assert "Copias_antigas_20052026.txt, na pasta do lote" in text


def test_no_photo_name_in_text() -> None:
    text = format_feedback(Summary.from_rows(ROWS), CONTROL, False, "desativada",
                           stale_by_folder={"Verificar": 1}, provisional=True)
    assert not any(r.photo_name in text for r in ROWS)


def test_empty_run_and_thousands() -> None:
    assert "0,0%" in format_feedback(Summary.from_rows([]), CONTROL, False, "desativada")
    many = [ControlRow.rejected(f"{i}.jpg", REJECTION_CORRUPTED) for i in range(1234)]
    assert "1.234" in format_feedback(Summary.from_rows(many), CONTROL, False, "desativada")




def test_stale_warning_agrees_in_number() -> None:
    one = format_feedback(Summary.from_rows(ROWS), CONTROL, False, "desativada",
                          stale_by_folder={"Verificar": 1})
    assert "Atenção: 1 imagem deste processamento tem uma cópia antiga" in one
    two = format_feedback(Summary.from_rows(ROWS), CONTROL, False, "desativada",
                          stale_by_folder={"Verificar": 2})
    assert "Atenção: 2 imagens deste processamento têm uma cópia antiga" in two
