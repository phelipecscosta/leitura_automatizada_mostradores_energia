"""Tabela de decisão do leitor: faixas e validadores 13.1 a 13.3 (spec 2.5, itens 11
a 14; decisões da ET6c, T6c.1).

Função pura: recebe as leituras livres do comitê e a decisão da função, e devolve a
linha da planilha. Não lê disco nem carrega modelo. Regras:

- Faixa muito baixa: sem leitura do display, ou confiança abaixo do limiar inferior.
  Vai para "Falha na extração", exceto quando o número de série lido é igual ao
  esperado: aí vai para "Verificar" com "Confiança baixa" (M-b). Vale também para a
  foto só com número de série.
- Fora da faixa muito baixa, cada falha gera um motivo, e vale o primeiro na ordem
  do pipeline (spec 2.5, item 1.7):
  - 13.1: sem número de série, "Número do medidor não localizado" (S-c); lido e
    diferente do esperado, "Número do medidor divergente do registro". O limiar alto
    é a igualdade (E103).
  - 13.2: leitura livre diferente do CSV, "Leitura de consumo divergente do registro",
    com o valor lido. A comparação é numérica (Z-a). No ciclométrico, diferença de
    exatamente 1 conta como igual: tambor em transição (B10-b).
  - 13.3 (digital): `function_decision`, pelo limiar da R17 e o motivo da R-b.
- Sem falha: "Processada com sucesso", com o número e a leitura do CSV (na transição,
  vale o valor do CSV, como na T-a).
O 13.4 (Parâmetros) fica para a ET8 (E08).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from meter_reader.contracts import (FUNCTION_OK, ControlRow, ExtractionRow, Reason, Rejection,
                                    Status, first_reason, wrong_function)
from meter_reader.reader_text import CONSUMPTION_CLASSES, FUNCTION_CLASSES

CYCLOMETRIC_TOLERANCE = 1   # B10-b: tambor em transição
REJECTION_EXTRACTION = Rejection(Status.REJECTED_LOW_QUALITY, Reason.EXTRACTION_FAILURE)
_FUNCTION_REASONS = frozenset({Reason.WRONG_FUNCTION, Reason.LOW_CONFIDENCE})


@dataclass(frozen=True)
class FieldReading:
    """Leitura livre de um campo pelo comitê: texto ("" sem leitura) e log da confiança."""

    text: str
    log_conf: float


@dataclass(frozen=True)
class FunctionDecision:
    """Resultado do validador 13.3 num digital: se a função permite aprovar, o valor da
    coluna Funcao ("03", "Função incorreta - XX" ou None) e o motivo quando não permite."""

    accepted: bool
    column: str | None
    reason: Reason | None = None

    def __post_init__(self) -> None:
        if self.accepted:
            if self.reason is not None or self.column not in (None, FUNCTION_OK):
                raise ValueError("Uma função aceita não tem motivo e só vale 03 ou NAN.")
        elif self.reason not in _FUNCTION_REASONS:
            raise ValueError("Motivo da função fora do vocabulário do validador 13.3.")

def function_decision(probs: Mapping[str, float], threshold: float) -> FunctionDecision:
    """13.3 num digital, pelas probabilidades médias do comitê por classe da cabeça
    (R17 e R-b). Aceita quando P(consumo) = P(vazio) + P(03) + P(003) >= limiar; a
    coluna Funcao fica "03" se, entre as classes de consumo, a mais provável é 03 ou
    003, e NAN se é vazio. Retido: se a classe mais provável não é de consumo, "Função
    incorreta", com "Função incorreta - XX" (NAN em "outro"); senão, "Confiança
    baixa", com a coluna como na aceitação. Empate: a primeira na ordem das classes."""
    if set(probs) != set(FUNCTION_CLASSES):
        raise ValueError("As probabilidades precisam ter as classes da função.")
    p_consumption = sum(probs[c] for c in CONSUMPTION_CLASSES)
    best_consumption = max(CONSUMPTION_CLASSES,
                           key=lambda c: (probs[c], -CONSUMPTION_CLASSES.index(c)))
    consumption_column = None if best_consumption == "vazio" else FUNCTION_OK
    if p_consumption >= threshold:
        return FunctionDecision(True, consumption_column)
    best = max(FUNCTION_CLASSES, key=lambda c: (probs[c], -FUNCTION_CLASSES.index(c)))
    if best in CONSUMPTION_CLASSES:
        return FunctionDecision(False, consumption_column, Reason.LOW_CONFIDENCE)
    column = None if best == "outro" else wrong_function(best)
    return FunctionDecision(False, column, Reason.WRONG_FUNCTION)


def reading_matches(text: str, expected: int, family: str | None) -> bool:
    """13.2: comparação numérica; no ciclométrico, diferença de até 1 (B10-b)."""
    if not text.isdigit():
        return False
    tolerance = CYCLOMETRIC_TOLERANCE if family == "ciclometrico" else 0
    return abs(int(text) - expected) <= tolerance


def _very_low(display: FieldReading | None, log_low: float) -> bool:
    return display is None or not display.text or math.isnan(display.log_conf) \
        or display.log_conf < log_low


def decide(name: str, row: ExtractionRow, family: str | None, display: FieldReading | None,
           serial: FieldReading | None, function: FunctionDecision | None,
           log_low: float) -> ControlRow:
    """Linha da planilha de uma foto que chegou ao leitor."""
    if family == "digital" and display is not None and function is None:
        raise ValueError("Um display digital precisa da decisão da função.")
    if family != "digital" and function is not None:
        raise ValueError("A decisão da função só existe no digital.")

    expected_meter = row.expected_meter_number
    serial_text = serial.text if serial is not None and serial.text else None
    serial_ok = serial_text == expected_meter

    if _very_low(display, log_low):           # item 11.1
        if serial_ok:                          # M-b: o número de série tem valor
            return ControlRow(name, serial_text, None, None, Status.NEEDS_REVIEW,
                              Reason.LOW_CONFIDENCE)
        return ControlRow.rejected(name, REJECTION_EXTRACTION)

    reasons = []
    if serial_text is None:                    # 13.1
        reasons.append(Reason.METER_NUMBER_NOT_FOUND)
    elif not serial_ok:
        reasons.append(Reason.METER_NUMBER_MISMATCH)
    if not reading_matches(display.text, row.reading, family):   # 13.2
        reasons.append(Reason.READING_MISMATCH)
    if function is not None and not function.accepted:          # 13.3
        reasons.append(function.reason)

    column = function.column if function is not None else None
    if not reasons:                            # item 14
        return ControlRow(name, expected_meter, column, row.reading, Status.APPROVED, None)
    read_value = int(display.text) if display.text.isdigit() else None
    return ControlRow(name, serial_text, column, read_value, Status.NEEDS_REVIEW,
                      first_reason(reasons))
