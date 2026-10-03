"""Leitor do arquivo BaseExtracao: fronteira de entrada do motor.

Converte o texto bruto do SAP em registros validados (contracts.py).
Qualquer linha inválida interrompe a leitura antes do processamento
(política de falhar cedo, T1.4). As mensagens indicam a linha, nunca o
caminho do arquivo nem valores do cliente (decisão E15).
"""

from __future__ import annotations

import csv
import io
import logging
import re
from datetime import date, datetime
from pathlib import Path

from meter_reader.contracts import MISSING_MARKER, ExtractionBatch, ExtractionRow

logger = logging.getLogger(__name__)

# A spec diz que o nome "sempre começa com" BaseExtracao_DDMMAAAA_Dia.csv
# (2.3). Aceita sufixos que o Windows acrescenta em downloads repetidos,
# como " (1)". Sem diferenciar maiúsculas, como o próprio Windows.
_FILE_NAME_RE = re.compile(r"^BaseExtracao_(\d{8})_Dia.*\.csv$", re.IGNORECASE)

# Formato observado nos 4 lotes reais (fatos F3 a F7 da ET1)
EXPECTED_HEADER = (
    "Numero do medidor",
    "Posicao do medidor lida",
    "Nota de Leitura Atual",
    "Foto do medidor",
)
_DELIMITER = ";"
_METER_SEPARATOR = "/"
_READING_RE = re.compile(r"^\d+$")


class ExtractionFileError(ValueError):
    """Arquivo BaseExtracao inválido. A mensagem é exibível ao utilizador."""


def parse_batch_date(path: Path | str) -> date:
    """Extrai a data do lote (DDMMAAAA) do nome do arquivo BaseExtracao."""
    match = _FILE_NAME_RE.fullmatch(Path(path).name)
    if match is None:
        raise ExtractionFileError(
            "Nome de arquivo fora do padrão BaseExtracao_DDMMAAAA_Dia.csv."
        )
    try:
        return datetime.strptime(match.group(1), "%d%m%Y").date()
    except ValueError as exc:
        # Oito dígitos que não formam uma data, como 32012026
        raise ExtractionFileError("Data inválida no nome do arquivo BaseExtracao.") from exc


def _decode(raw: bytes) -> str:
    """UTF-8 (com ou sem BOM); se falhar, Windows-1252 com aviso (T1.4)."""
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    try:
        text = raw.decode("cp1252")
    except UnicodeDecodeError as exc:
        # Alguns bytes não existem nem em Windows-1252
        raise ExtractionFileError("Codificação do BaseExtracao não reconhecida.") from exc
    logger.warning("O BaseExtracao não está em UTF-8; foi lido como Windows-1252.")
    return text


def _optional(value: str) -> str | None:
    """Converte o marcador de ausência do SAP (e o campo vazio) em None."""
    return None if value in ("", MISSING_MARKER) else value


def read_extraction(path: Path | str) -> ExtractionBatch:
    """Lê e valida um arquivo BaseExtracao inteiro."""
    path = Path(path)
    batch_date = parse_batch_date(path)
    text = _decode(path.read_bytes())

    # newline="": o módulo csv trata o CRLF corretamente
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=_DELIMITER)
    header = next(reader, None)
    if header is None or tuple(header) != EXPECTED_HEADER:
        raise ExtractionFileError("Cabeçalho do BaseExtracao diferente do esperado.")

    rows: list[ExtractionRow] = []
    seen_photos: set[str] = set()
    for line_number, fields in enumerate(reader, start=2):  # linha 1 = cabeçalho
        if not fields:
            continue  # linha totalmente vazia: não há dado a perder
        if len(fields) != len(EXPECTED_HEADER):
            raise ExtractionFileError(
                f"Linha {line_number}: esperados {len(EXPECTED_HEADER)} campos, "
                f"encontrados {len(fields)}."
            )
        meter_field, reading_field, note_field, photo_field = fields
        if not _READING_RE.fullmatch(reading_field):
            raise ExtractionFileError(
                f"Linha {line_number}: a leitura não é um inteiro sem sinal."
            )
        try:
            row = ExtractionRow(
                meter_numbers=tuple(meter_field.split(_METER_SEPARATOR)),
                reading=int(reading_field),
                note=_optional(note_field),
                photo_name=_optional(photo_field),
            )
        except (ValueError, TypeError) as exc:
            # As mensagens do contrato não contêm valores do cliente
            raise ExtractionFileError(f"Linha {line_number}: {exc}") from exc
        if row.photo_name is not None:
            if row.photo_name in seen_photos:
                raise ExtractionFileError(
                    f"Linha {line_number}: nome de foto repetido no BaseExtracao."
                )
            seen_photos.add(row.photo_name)
        rows.append(row)

    if not rows:
        raise ExtractionFileError("O BaseExtracao não contém linhas de dados.")
    return ExtractionBatch(batch_date=batch_date, rows=tuple(rows))
