"""Regras de aceitação dos rótulos do professor (T6a.1: R1 a R3 e fila F-2) e medidas
do piloto (T6a.3): status de cada recorte, suspeita de tambor decimal (D2, item 2) e
estabilidade entre repetições.
"""

from __future__ import annotations

import itertools
import re

ACCEPTED, QUEUE_F2, DISAGREE = "aceito", "fila_f2", "discorda"
PARTIAL, ILLEGIBLE, FORMAT_ERROR = "parcial", "ilegivel", "erro_formato"
STATUSES = (ACCEPTED, QUEUE_F2, DISAGREE, PARTIAL, ILLEGIBLE, FORMAT_ERROR)

FUNCTIONS_OK = {"03", ""}  # R1: função 03, ou display sem código de função
_TOKEN_RE = re.compile(r"\[[0-9]{2}\]|[0-9?]")


def reading_values(text: str) -> set[int] | None:
    """Valores numéricos possíveis de uma leitura transcrita: cada transição [ab] vale a
    ou b. None se houver caractere ilegível (?)."""
    if "?" in text:
        return None
    options = [list(t[1:3]) if t.startswith("[") else [t] for t in _TOKEN_RE.findall(text)]
    return {int("".join(c)) for c in itertools.product(*options)}


def label_status(field_class: str, record: dict, expected_number: str,
                 expected_reading: int) -> str:
    """Status do recorte pelas regras da T6a.1. `expected_number` é o bloco mais à
    esquerda do número do medidor no CSV; `expected_reading`, a leitura do CSV."""
    resp = record.get("resposta")
    if record.get("erro") is not None or resp is None:
        return FORMAT_ERROR
    if not resp["legivel"]:
        return ILLEGIBLE
    if field_class == "numero_serie":
        if "?" in resp["numero"]:
            return PARTIAL
        return ACCEPTED if resp["numero"] == expected_number.upper() else DISAGREE
    values = reading_values(resp["leitura"])
    if values is None:
        return PARTIAL
    agrees = expected_reading in values  # comparação numérica: sem os zeros à esquerda
    if field_class == "display_digital":
        return ACCEPTED if agrees and resp["funcao"] in FUNCTIONS_OK else QUEUE_F2
    return ACCEPTED if agrees else DISAGREE


def decimal_drum_suspect(record: dict, expected_reading: int) -> bool:
    """D2, item 2: a leitura sem o último tambor bate com o CSV, e a leitura inteira não."""
    resp = record.get("resposta")
    values = reading_values(resp["leitura"]) if resp and resp.get("legivel") else None
    return bool(values) and expected_reading not in values and any(
        v // 10 == expected_reading for v in values)


def stable(first: dict, second: dict) -> bool:
    """Duas respostas ao mesmo pedido são idênticas (resposta validada e erro)."""
    return (first.get("resposta"), first.get("erro")) == (second.get("resposta"), second.get("erro"))
