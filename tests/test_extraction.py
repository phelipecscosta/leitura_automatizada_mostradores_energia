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
