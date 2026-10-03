"""Contratos do motor: vocabulário fixo e registros trocados entre módulos.

Os textos de Status e Motivo são exatamente os da spec (2.5, item 1), porque
vão para a planilha de controle e são lidos pelo utilizador. Usar estas
enumerações, e nunca o texto digitado, impede status inválidos na planilha.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum


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
