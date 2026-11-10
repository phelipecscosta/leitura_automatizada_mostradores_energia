"""Planilha de controle Controle_DDMMAAAA.xlsx (spec v14, 2.5, itens 1 e 2.2).

Duas abas: "Controle", uma linha por imagem, com o nome do arquivo como
chave; e "Execução", uma linha por processamento. Células sem valor recebem
o texto NAN, como na spec; o Motivo fica vazio em "Processada com sucesso".

A leitura é estrita: uma linha fora do vocabulário interrompe tudo, com a
linha indicada e sem exibir valores (mesma política do BaseExtracao, E16).
A gravação é atômica: a planilha nova é escrita num arquivo temporário da
mesma pasta e só então substitui a antiga.
"""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path

from openpyxl import Workbook, load_workbook

from meter_reader.contracts import ControlRow, Reason, Status

CONTROL_SHEET = "Controle"
EXECUTION_SHEET = "Execução"
NAN = "NAN"

# Nomes das colunas exatamente como na spec (2.5, itens 1.1 a 1.6)
CONTROL_COLUMNS = (
    "Foto do medidor",
    "Numero do medidor",
    "Funcao",
    "Posicao do medidor lida",
    "Status",
    "Motivo",
)
# Spec 2.5, item 1.8
EXECUTION_COLUMNS = (
    "Data e hora",
    "Tipo de processamento",
    "Perfil de execução",
    "Versão dos modelos",
    "Validação por Parâmetros",
)
_DATETIME_FORMAT = "dd/mm/yyyy hh:mm:ss"

PROFILES = ("CPU", "GPU")
PARAMETERS_DISABLED = "desativada"
_PARAMETERS_RE = re.compile(r"^(desativada|ativa \(.+\))$")


class ControlFileError(ValueError):
    """Planilha de controle inválida ou inacessível. A mensagem é exibível ao utilizador."""


class RunKind(StrEnum):
    """Tipo de processamento (spec 2.5, itens 1.8 e 2.2)."""

    FULL = "completo"
    COMPLEMENTARY = "complementar"


@dataclass(frozen=True)
class ExecutionRecord:
    """Uma linha da aba Execução."""

    timestamp: datetime
    kind: RunKind
    profile: str  # "CPU" ou "GPU" (E06)
    model_versions: str
    parameters: str  # "desativada" ou "ativa (nome do arquivo)"

    def __post_init__(self) -> None:
        if self.profile not in PROFILES:
            raise ValueError("Perfil de execução precisa ser CPU ou GPU.")
        if not self.model_versions:
            raise ValueError("A versão dos modelos não pode ficar vazia.")
        if not _PARAMETERS_RE.fullmatch(self.parameters):
            raise ValueError("Estado da validação por Parâmetros fora do padrão.")


def control_file_name(batch_date: date) -> str:
    """Controle_DDMMAAAA.xlsx, com a data do BaseExtracao (spec 2.4, ação 5)."""
    return f"Controle_{batch_date:%d%m%Y}.xlsx"


# ---------------------------------------------------------------------
# Mesclagem do reprocessamento complementar (spec 2.5, item 2.2)
# ---------------------------------------------------------------------


def merge_rows(existing: Iterable[ControlRow], new: Iterable[ControlRow]) -> list[ControlRow]:
    """Substitui as linhas das fotos reprocessadas, no lugar, e acrescenta as novas ao fim.

    As demais linhas permanecem intactas e na mesma ordem.
    """
    new = list(new)
    pending = {r.photo_name: r for r in new}
    if len(pending) != len(new):
        raise ValueError("Foto repetida entre as linhas novas.")
    merged = [pending.pop(r.photo_name, r) for r in existing]
    merged.extend(r for r in new if r.photo_name in pending)
    return merged


# ---------------------------------------------------------------------
# Conversão entre linhas e células
# ---------------------------------------------------------------------


def _cell(value):
    return NAN if value is None else value


def _row_to_cells(row: ControlRow) -> list:
    # O número do medidor vai como texto, para preservar zeros à esquerda
    return [
        row.photo_name,
        _cell(row.meter_number),
        _cell(row.function),
        _cell(row.reading),
        str(row.status),
        "" if row.reason is None else str(row.reason),
    ]


def _optional(value):
    return None if value in (None, "", NAN) else value


def _reading_from_cell(value) -> int | None:
    value = _optional(value)
    if value is None:
        return None
    # O Excel pode devolver 12.0 para um número editado à mão
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise ValueError("A leitura precisa ser um número inteiro.")


def _cells_to_row(values: tuple) -> ControlRow:
    photo, meter, function, reading, status, reason = values
    reason = _optional(reason)
    meter = _optional(meter)
    return ControlRow(
        photo_name=str(photo) if photo is not None else "",
        meter_number=None if meter is None else str(meter),
        function=_optional(function),
        reading=_reading_from_cell(reading),
        status=Status(status),
        reason=None if reason is None else Reason(reason),
    )


def _cells_to_execution(values: tuple) -> ExecutionRecord:
    timestamp, kind, profile, versions, parameters = values
    if not isinstance(timestamp, datetime):
        raise ValueError("Data e hora inválida.")
    return ExecutionRecord(timestamp, RunKind(kind), profile, versions, parameters)


# ---------------------------------------------------------------------
# Leitura e gravação
# ---------------------------------------------------------------------


def _read_sheet(workbook, name: str, columns: tuple[str, ...], convert) -> list:
    if name not in workbook.sheetnames:
        raise ControlFileError(f"A planilha de controle não tem a aba {name}.")
    rows = workbook[name].iter_rows(values_only=True)
    header = next(rows, None)
    if header is None or tuple(header[: len(columns)]) != columns:
        raise ControlFileError(f"Aba {name}: cabeçalho diferente do esperado.")
    out = []
    for line, values in enumerate(rows, start=2):  # linha 1 = cabeçalho
        values = tuple(values[: len(columns)])
        if all(v in (None, "") for v in values):
            continue  # linha vazia deixada pelo Excel
        try:
            out.append(convert(values))
        except (ValueError, TypeError) as exc:
            # As mensagens dos contratos não exibem valores do cliente
            raise ControlFileError(f"Aba {name}, linha {line}: valor inválido.") from exc
    return out


def read_control(path: Path | str) -> tuple[list[ControlRow], list[ExecutionRecord]]:
    """Lê e valida a planilha de controle inteira."""
    try:
        workbook = load_workbook(path, read_only=True, data_only=True)
    except (OSError, ValueError, KeyError) as exc:
        raise ControlFileError("Não foi possível abrir a planilha de controle.") from exc
    try:
        rows = _read_sheet(workbook, CONTROL_SHEET, CONTROL_COLUMNS, _cells_to_row)
        executions = _read_sheet(workbook, EXECUTION_SHEET, EXECUTION_COLUMNS,
                                 _cells_to_execution)
    finally:
        workbook.close()
    names = [r.photo_name for r in rows]
    if len(set(names)) != len(names):
        raise ControlFileError(f"Aba {CONTROL_SHEET}: nome de foto repetido.")
    return rows, executions


def write_control(path: Path | str, rows: Iterable[ControlRow],
                  executions: Iterable[ExecutionRecord]) -> None:
    """Grava a planilha inteira, de forma atômica."""
    path = Path(path)
    workbook = Workbook()
    control = workbook.active
    control.title = CONTROL_SHEET
    control.append(CONTROL_COLUMNS)
    for row in rows:
        control.append(_row_to_cells(row))
    control.freeze_panes = "A2"
    control.auto_filter.ref = control.dimensions  # filtro por status para o analista

    execution = workbook.create_sheet(EXECUTION_SHEET)
    execution.append(EXECUTION_COLUMNS)
    for record in executions:
        execution.append([record.timestamp, str(record.kind), record.profile,
                          record.model_versions, record.parameters])
        execution.cell(execution.max_row, 1).number_format = _DATETIME_FORMAT

    # Arquivo temporário na mesma pasta: a troca final é uma única operação
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix="~tmp_", suffix=".xlsx")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        workbook.save(tmp)
        os.replace(tmp, path)
    except PermissionError as exc:
        # No Windows, o Excel bloqueia o arquivo aberto
        raise ControlFileError(
            "A planilha de controle está aberta em outro programa. "
            "Feche-a e tente novamente."
        ) from exc
    finally:
        tmp.unlink(missing_ok=True)


def save_control(path: Path | str, rows: Iterable[ControlRow],
                 record: ExecutionRecord) -> list[ControlRow]:
    """Cria a planilha (processamento completo) ou mescla nela (complementar).

    Devolve as linhas gravadas. O tipo do processamento precisa corresponder à
    existência da planilha, para que um lote nunca seja sobrescrito por engano.
    """
    path = Path(path)
    rows = list(rows)
    if path.exists():
        if record.kind is not RunKind.COMPLEMENTARY:
            raise ControlFileError("Este lote já tem planilha de controle: "
                                   "use o processamento complementar.")
        existing, executions = read_control(path)
        rows = merge_rows(existing, rows)
        executions.append(record)
    else:
        if record.kind is not RunKind.FULL:
            raise ControlFileError("Não há planilha de controle para complementar.")
        merge_rows([], rows)  # recusa fotos repetidas
        executions = [record]
    write_control(path, rows, executions)
    return rows
