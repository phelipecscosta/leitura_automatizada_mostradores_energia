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
    """Status de uma imagem na planilha de controle (spec v14, 2.5, item 1.5)."""

    APPROVED = "Processada com sucesso"
    NEEDS_REVIEW = "Necessita de verificação"
    REJECTED_OTHER = "Rejeitada - Outros"
    REJECTED_LOW_QUALITY = "Rejeitada - Baixa qualidade"
    REJECTED_MULTIPLE = "Rejeitada - Múltiplos medidores"
    REJECTED_CORRUPTED = "Rejeitada - Corrompida"
    REJECTED_NOT_LISTED = "Rejeitada - Não listada em BaseExtracao"
    NOT_PROCESSED = "Não processada"


class Reason(StrEnum):
    """Motivo de rejeição ou verificação (spec v14, 2.5, item 1.6).

    A ordem de declaração é a ordem do pipeline e decide qual motivo
    prevalece quando mais de um se aplica (spec 2.5, item 1.7). Não reordene.
    """

    CORRUPTED = "Arquivo corrompido ou vazio"
    NOT_IN_EXTRACTION = "Sem registro na tabela BaseExtracao"
    NOT_A_METER = "Não é medidor"
    INDETERMINATE_TYPE = "Indeterminação do tipo de medidor"
    MULTIPLE_METERS = "Múltiplos medidores"
    NOT_IN_PARAMETERS = "Não listado na tabela fornecida com os parâmetros"
    TYPE_MISMATCH = "Tipo divergente entre triagem e detector"
    EXTRACTION_FAILURE = "Falha na extração"
    LOW_CONFIDENCE = "Confiança baixa"
    METER_NUMBER_MISMATCH = "Número do medidor divergente do registro"
    READING_MISMATCH = "Leitura de consumo divergente do registro"
    WRONG_FUNCTION = "Função incorreta"
    HIGH_CRITICALITY = "Cliente de alta criticidade"
    OUT_OF_PARAMETERS = "Leitura fora dos Parâmetros"

# Classes de saída da triagem, na ordem do contrato (spec v11, 2.5, item 5; E49).
# No colapso binário, "medidor" soma as quatro primeiras.
PRODUCT_CLASSES = ("digital", "ciclometrico", "indeterminado", "multiplos", "outros")


# Saídas do modelo da triagem: os 12 grupos de treino, na ordem fixa das saídas
# (protocolo da triagem, seção 1.1, fusão U-b; revisão R6). Mudar esta ordem
# invalida qualquer modelo treinado. O pacote training confere, ao ser
# importado, que ela é a mesma que ele deriva dos grupos rotulados.
TRIAGE_GROUPS = (
    "dig1_quadrado_branco", "dig2_redondo_branco", "dig3_vidro_redondo", "dig4_verde_arduino",
    "cic_disco", "cic_duas_leituras", "cic_uma_leitura",
    "indeterminado",
    "multiplos",
    "out_acesso", "out_sem_acesso", "out_sem_contexto",
)

# Classe do produto (índice em PRODUCT_CLASSES) de cada saída: o mapa que soma
# as probabilidades dos grupos por classe
TRIAGE_GROUP_CLASS = (0, 0, 0, 0, 1, 1, 1, 2, 3, 4, 4, 4)


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

# Rejeições da triagem por classe do produto (spec v12, 2.5, itens 6 e 7).
# Digital e ciclométrico não são rejeitados: seguem para o detector.
TRIAGE_REJECTIONS = {
    "indeterminado": Rejection(Status.REJECTED_LOW_QUALITY, Reason.INDETERMINATE_TYPE),
    "multiplos": Rejection(Status.REJECTED_MULTIPLE, Reason.MULTIPLE_METERS),
    "outros": Rejection(Status.REJECTED_OTHER, Reason.NOT_A_METER),
}

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


# ---------------------------------------------------------------------
# Saída: linha da planilha de controle (spec v14, 2.5, item 1)
# ---------------------------------------------------------------------

# Motivos permitidos para cada status. Fecha o vocabulário em pares:
# nenhum módulo consegue gravar um status com o motivo de outro.
# "Tipo divergente entre triagem e detector" vai para verificação
# (lado seguro; a spec não fixa o status desse motivo).
STATUS_REASONS: dict[Status, frozenset[Reason]] = {
    Status.APPROVED: frozenset(),
    Status.NEEDS_REVIEW: frozenset({
        Reason.TYPE_MISMATCH, Reason.LOW_CONFIDENCE, Reason.METER_NUMBER_MISMATCH,
        Reason.READING_MISMATCH, Reason.WRONG_FUNCTION, Reason.HIGH_CRITICALITY,
        Reason.OUT_OF_PARAMETERS,
    }),
    Status.REJECTED_OTHER: frozenset({Reason.NOT_A_METER}),
    Status.REJECTED_LOW_QUALITY: frozenset({Reason.INDETERMINATE_TYPE, Reason.EXTRACTION_FAILURE}),
    Status.REJECTED_MULTIPLE: frozenset({Reason.MULTIPLE_METERS}),
    Status.REJECTED_CORRUPTED: frozenset({Reason.CORRUPTED}),
    Status.REJECTED_NOT_LISTED: frozenset({Reason.NOT_IN_EXTRACTION}),
    Status.NOT_PROCESSED: frozenset({Reason.NOT_IN_PARAMETERS}),
}

# Pasta de destino de cada status, relativa à pasta do lote (spec 2.3).
# None: a imagem não é movimentada e fica na pasta de origem.
DESTINATION_FOLDERS: dict[Status, tuple[str, ...] | None] = {
    Status.APPROVED: ("Aprovadas",),
    Status.NEEDS_REVIEW: ("Verificar",),
    Status.REJECTED_OTHER: ("Rejeitadas", "Outros"),
    Status.REJECTED_LOW_QUALITY: ("Rejeitadas", "Baixa qualidade"),
    Status.REJECTED_MULTIPLE: ("Rejeitadas", "Múltiplos"),
    Status.REJECTED_CORRUPTED: ("Rejeitadas", "Corrompidas"),
    Status.REJECTED_NOT_LISTED: ("Rejeitadas", "Não listada"),
    Status.NOT_PROCESSED: None,
}

# Status em que todos os campos extraídos ficam NAN (spec 2.5, itens 2.3,
# 2.4, 6, 7, 9, 10.2 e 11.1)
_EMPTY_FIELD_STATUSES = frozenset(STATUS_REASONS) - {Status.APPROVED, Status.NEEDS_REVIEW}

FUNCTION_OK = "03"
_WRONG_FUNCTION_RE = re.compile(r"^Função incorreta - \d+$")


def wrong_function(code: str) -> str:
    """Texto da coluna Funcao para uma função lida diferente de 03 (spec 2.5, item 13.3)."""
    if not code.isdigit():
        raise ValueError("O código da função precisa ter só dígitos.")
    return f"Função incorreta - {code}"


@dataclass(frozen=True)
class ControlRow:
    """Uma linha da planilha de controle. None representa NAN."""

    photo_name: str  # chave da planilha
    meter_number: str | None
    function: str | None  # "03", "Função incorreta - XX" ou None
    reading: int | None
    status: Status
    reason: Reason | None  # None só em "Processada com sucesso"

    def __post_init__(self) -> None:
        if not self.photo_name:
            raise ValueError("A linha precisa do nome da foto.")
        allowed = STATUS_REASONS[self.status]
        if self.reason is None:
            if allowed:
                raise ValueError("Este status exige um motivo.")
        elif self.reason not in allowed:
            raise ValueError("Motivo incompatível com o status.")
        if self.meter_number is not None and not _METER_PART_RE.fullmatch(self.meter_number):
            raise ValueError("Número de medidor com caracteres inválidos.")
        if self.function is not None and self.function != FUNCTION_OK \
                and not _WRONG_FUNCTION_RE.fullmatch(self.function):
            raise ValueError("Valor de função fora do vocabulário.")
        if self.reading is not None and (isinstance(self.reading, bool)
                                         or not isinstance(self.reading, int)
                                         or self.reading < 0):
            raise ValueError("A leitura precisa ser um inteiro não negativo.")
        extracted = (self.meter_number, self.function, self.reading)
        if self.status in _EMPTY_FIELD_STATUSES and any(v is not None for v in extracted):
            raise ValueError("Neste status, os campos extraídos ficam vazios.")
        if self.status is Status.APPROVED:
            if self.meter_number is None or self.reading is None:
                raise ValueError("Uma aprovação exige número do medidor e leitura.")
            if self.function not in (None, FUNCTION_OK):
                raise ValueError("Uma aprovação não pode ter função incorreta.")

    @classmethod
    def rejected(cls, photo_name: str, rejection: Rejection) -> ControlRow:
        """Linha de uma rejeição (ou de "Não processada"), com os campos NAN."""
        return cls(photo_name, None, None, None, rejection.status, rejection.reason)

    @property
    def destination(self) -> tuple[str, ...] | None:
        """Pasta de destino relativa à pasta do lote; None: fica na origem."""
        return DESTINATION_FOLDERS[self.status]
