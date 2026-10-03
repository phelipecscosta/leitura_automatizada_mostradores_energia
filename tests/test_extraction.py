"""Testes do leitor do BaseExtracao (somente dados sintéticos)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from meter_reader.extraction import parse_batch_date


@pytest.mark.parametrize(
    "name",
    [
        "BaseExtracao_03072026_Dia.csv",
        "BaseExtracao_03072026_Dia (1).csv",  # download repetido no Windows
        "BaseExtracao_03072026_Dia.CSV",
    ],
)
def test_batch_date_is_parsed(name) -> None:
    assert parse_batch_date(name) == date(2026, 7, 3)


def test_only_file_name_is_used(tmp_path: Path) -> None:
    # A data vem do nome, nunca das pastas do caminho
    path = tmp_path / "BaseExtracao_01012025_Dia" / "BaseExtracao_03072026_Dia.csv"
    assert parse_batch_date(path) == date(2026, 7, 3)


@pytest.mark.parametrize(
    "name",
    [
        "BaseExtracao_0307202_Dia.csv",  # 7 dígitos
        "Base_03072026_Dia.csv",  # prefixo errado
        "BaseExtracao_03072026_Dia.xlsx",  # extensão errada
        "BaseExtracao_32012026_Dia.csv",  # dia inexistente
        "BaseExtracao_29022026_Dia.csv",  # 2026 não é bissexto
    ],
)
def test_invalid_names_are_rejected(name) -> None:
    with pytest.raises(ValueError):
        parse_batch_date(name)


def test_error_message_does_not_expose_path(tmp_path: Path) -> None:
    # Decisão E15: mensagens sem caminhos
    path = tmp_path / "pasta_secreta" / "arquivo_errado.csv"
    with pytest.raises(ValueError) as error:
        parse_batch_date(path)
    assert "pasta_secreta" not in str(error.value)

# ---------------------------------------------------------------------
# Leitura do arquivo (CSV sintético gravado em pasta temporária)
# ---------------------------------------------------------------------
import logging

from meter_reader.extraction import ExtractionFileError, read_extraction

HEADER = "Numero do medidor;Posicao do medidor lida;Nota de Leitura Atual;Foto do medidor"
PHOTO_1 = "LOTE_TESTE_00000000000000000001_000.jpg"
PHOTO_2 = "LOTE_TESTE_00000000000000000002_001.jpg"


def _write_csv(tmp_path: Path, lines: list[str], encoding: str = "utf-8",
               name: str = "BaseExtracao_03072026_Dia.csv") -> Path:
    """Grava um CSV sintético com CRLF, como o arquivo real (fato F3)."""
    path = tmp_path / name
    path.write_bytes(("\r\n".join(lines) + "\r\n").encode(encoding))
    return path


def test_valid_file_is_read(tmp_path: Path) -> None:
    path = _write_csv(tmp_path, [
        HEADER,
        f"B12345/99887766;123;T181;{PHOTO_1}",  # número composto
        "123456789A;0;NA;NA",  # leitura zero, sem nota e sem foto
        f"9999;45;NA;{PHOTO_2}",  # número curto (opção B)
    ])
    batch = read_extraction(path)
    assert batch.batch_date == date(2026, 7, 3)
    assert len(batch.rows) == 3
    first, second, third = batch.rows
    assert first.meter_numbers == ("B12345", "99887766")
    assert first.expected_meter_number == "B12345"
    assert (first.reading, first.note, first.photo_name) == (123, "T181", PHOTO_1)
    assert (second.reading, second.note, second.photo_name) == (0, None, None)
    assert third.expected_meter_number == "9999"


def test_utf8_bom_is_accepted(tmp_path: Path) -> None:
    path = _write_csv(tmp_path, [HEADER, f"B12345;1;NA;{PHOTO_1}"], encoding="utf-8-sig")
    assert len(read_extraction(path).rows) == 1


def test_cp1252_falls_back_with_warning(tmp_path: Path, caplog) -> None:
    # Nota com acento gravada em Windows-1252: não é UTF-8 válido
    path = _write_csv(tmp_path, [HEADER, f"B12345;1;Leitura não;{PHOTO_1}"], encoding="cp1252")
    with caplog.at_level(logging.WARNING):
        batch = read_extraction(path)
    assert batch.rows[0].note == "Leitura não"
    assert "Windows-1252" in caplog.text


def test_wrong_header_is_rejected(tmp_path: Path) -> None:
    path = _write_csv(tmp_path, [HEADER.replace("Foto", "Imagem"), f"B12345;1;NA;{PHOTO_1}"])
    with pytest.raises(ExtractionFileError):
        read_extraction(path)


def test_wrong_field_count_reports_line(tmp_path: Path) -> None:
    path = _write_csv(tmp_path, [HEADER, f"B12345;1;NA;{PHOTO_1}", "B12345;1;NA"])
    with pytest.raises(ExtractionFileError, match="Linha 3"):
        read_extraction(path)


@pytest.mark.parametrize("reading", ["12,5", "", "abc", "-1"])
def test_invalid_reading_is_rejected(tmp_path: Path, reading) -> None:
    path = _write_csv(tmp_path, [HEADER, f"B12345;{reading};NA;{PHOTO_1}"])
    with pytest.raises(ExtractionFileError, match="Linha 2"):
        read_extraction(path)


def test_repeated_photo_is_rejected(tmp_path: Path) -> None:
    path = _write_csv(tmp_path, [HEADER, f"B12345;1;NA;{PHOTO_1}", f"C67890;2;NA;{PHOTO_1}"])
    with pytest.raises(ExtractionFileError, match="Linha 3"):
        read_extraction(path)


def test_header_only_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ExtractionFileError):
        read_extraction(_write_csv(tmp_path, [HEADER]))


def test_blank_lines_are_ignored(tmp_path: Path) -> None:
    path = _write_csv(tmp_path, [HEADER, f"B12345;1;NA;{PHOTO_1}", "", ""])
    assert len(read_extraction(path).rows) == 1


def test_error_message_does_not_expose_values(tmp_path: Path) -> None:
    # Número de medidor inválido: a mensagem cita a linha, nunca o valor
    path = _write_csv(tmp_path, [HEADER, f"SEGREDO 1;1;NA;{PHOTO_1}"])
    with pytest.raises(ExtractionFileError) as error:
        read_extraction(path)
    assert "SEGREDO" not in str(error.value)
    assert "Linha 2" in str(error.value)
