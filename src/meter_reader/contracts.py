"""Contratos do motor: vocabulário fixo e registros trocados entre módulos.

Os textos de Status e Motivo são exatamente os da spec (2.5, item 1), porque
vão para a planilha de controle e são lidos pelo utilizador. Usar estas
enumerações, e nunca o texto digitado, impede status inválidos na planilha.

Os registros são dataclasses imutáveis. A validação aqui garante invariantes
simples; a conversão de texto bruto é feita na fronteira (leitor do CSV).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from enum import StrEnum

# ---------------------------------------------------------------------
# Vocabulário da planilha de controle
# ---------------------------------------------------------------------


class Status(StrEnum):
    """Status de uma imagem na planilha de controle (spec 2.5, item 1.5)."""

    APPROVED = "Processada com sucesso"
    NEEDS_REVIEW = "Necessita de verificação"
    REJECTED_OTHER = "Rejeitada - Outros"
    REJECTED_LOW_QUALITY = "Rejeitada - Baixa qualidade"
    REJECTED_CORRUPTED = "Rejeitada - Corrompida"
    REJECTED_NOT_LISTED = "Rejeitada - Não listada em BaseExtracao"
    NOT_PROCESSED = "Não processada"


class Reason(StrEnum):
    """Motivo de rejeição ou verificação (spec 2.5, item 1.6).

    A ordem de declaração é a ordem do pipeline e decide qual motivo
    prevalece quando mais de um se aplica (spec 2.5, item 1.7). Não reordene.
    """

    CORRUPTED = "Arquivo corrompido ou vazio"
    NOT_IN_EXTRACTION = "Sem registro na tabela BaseExtracao"
    NOT_A_METER = "Não é medidor"
    ILLEGIBLE = "Ilegível"
    NOT_IN_PARAMETERS = "Não listado na tabela fornecida com os parâmetros"
    TYPE_MISMATCH = "Tipo divergente entre triagem e detector"
    LOW_CONFIDENCE = "Confiança baixa"
    METER_NUMBER_MISMATCH = "Número do medidor divergente do registro"
    READING_MISMATCH = "Leitura de consumo divergente do registro"
    WRONG_FUNCTION = "Função incorreta"
    HIGH_CRITICALITY = "Cliente de alta criticidade"
    OUT_OF_PARAMETERS = "Leitura fora dos Parâmetros"


def first_reason(reasons: Iterable[Reason]) -> Reason:
    """Retorna o motivo que prevalece: o primeiro na ordem do pipeline."""
    order = list(Reason)
    candidates = list(reasons)
    if not candidates:
        raise ValueError("É preciso informar ao menos um motivo.")
    return min(candidates, key=order.index)


@dataclass(frozen=True)
class Rejection:
    """Par fixo de status e motivo de uma rejeição."""

    status: Status
    reason: Reason


# Pares definidos pela spec (2.5, itens 2.3 e 2.4)
REJECTION_CORRUPTED = Rejection(Status.REJECTED_CORRUPTED, Reason.CORRUPTED)
REJECTION_NOT_LISTED = Rejection(Status.REJECTED_NOT_LISTED, Reason.NOT_IN_EXTRACTION)

# ---------------------------------------------------------------------
# Entrada: BaseExtracao
# ---------------------------------------------------------------------

# Marcador de ausência usado pelo SAP no CSV (fato F4 da ET1).
# Precisa virar None no leitor; nunca pode chegar a um registro.
MISSING_MARKER = "NA"

# Opção B do item 5 da T1.3: letras e dígitos, sem tamanho fixo
_METER_PART_RE = re.compile(r"^[A-Za-z0-9]+$")


@dataclass(frozen=True)
class ExtractionRow:
    """Uma linha do BaseExtracao (spec 2.5, item 2.4)."""

    meter_numbers: tuple[str, ...]  # todas as partes, na ordem do CSV
    reading: int  # "Posicao do medidor lida" (inteiro, fato F6)
    note: str | None  # "Nota de Leitura Atual"; None quando ausente
    photo_name: str | None  # nome exato do arquivo; None quando sem foto

    def __post_init__(self) -> None:
        if not self.meter_numbers:
            raise ValueError("A linha precisa ter ao menos um número de medidor.")
        for part in self.meter_numbers:
            if not _METER_PART_RE.fullmatch(part):
                raise ValueError("Número de medidor com caracteres inválidos.")
        # bool é subclasse de int em Python; por isso é excluído à parte
        if isinstance(self.reading, bool) or not isinstance(self.reading, int):
            raise TypeError("A leitura precisa ser um inteiro.")
        if self.reading < 0:
            raise ValueError("A leitura não pode ser negativa.")
        for field_value in (self.note, self.photo_name):
            if field_value is not None and field_value in ("", MISSING_MARKER):
                raise ValueError("Ausência deve ser representada por None.")

    @property
    def expected_meter_number(self) -> str:
        """Número esperado na foto: a parte mais à esquerda (spec 2.5, item 2.4)."""
        return self.meter_numbers[0]


@dataclass(frozen=True)
class ExtractionBatch:
    """Conteúdo de um arquivo BaseExtracao: a data do lote e as suas linhas."""

    batch_date: date  # DDMMAAAA do nome do arquivo (fato F1)
    rows: tuple[ExtractionRow, ...]


# ---------------------------------------------------------------------
# Saída: integridade e pareamento
# ---------------------------------------------------------------------


class IntegrityIssue(StrEnum):
    """Causa técnica de uma falha de integridade (spec 2.5, item 2.3).

    Vai para o manifesto e para o log. Na planilha, todas aparecem com o
    mesmo motivo: "Arquivo corrompido ou vazio".
    """

    UNREADABLE = "não abre ou não decodifica"
    EMPTY_DARK = "vazia: preta"
    EMPTY_BRIGHT = "vazia: branca"


@dataclass(frozen=True)
class IntegrityResult:
    """Resultado da verificação de integridade de uma imagem."""

    photo_name: str
    issue: IntegrityIssue | None  # None: imagem íntegra

    @property
    def rejection(self) -> Rejection | None:
        return None if self.issue is None else REJECTION_CORRUPTED


@dataclass(frozen=True)
class PairingResult:
    """Resultado do pareamento de uma imagem com o BaseExtracao."""

    photo_name: str
    row: ExtractionRow | None  # None: imagem sem linha (órfã)

    def __post_init__(self) -> None:
        if self.row is not None and self.row.photo_name != self.photo_name:
            raise ValueError("A linha pareada precisa apontar para a mesma foto.")

    @property
    def rejection(self) -> Rejection | None:
        return None if self.row is not None else REJECTION_NOT_LISTED
