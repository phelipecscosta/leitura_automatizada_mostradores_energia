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
    "Rejeitada - Corrompida",
    "Rejeitada - Não listada em BaseExtracao",
    "Não processada",
]
SPEC_REASONS_IN_PIPELINE_ORDER = [
    "Arquivo corrompido ou vazio",
    "Sem registro na tabela BaseExtracao",
    "Não é medidor",
    "Ilegível",
    "Não listado na tabela fornecida com os parâmetros",
    "Tipo divergente entre triagem e detector",
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
