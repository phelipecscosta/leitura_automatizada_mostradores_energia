"""Alfabetos e codificação de texto do leitor (protocolo do leitor, seções 1.3 e 1.6).

Duas cabeças de CTC (F-c): o texto principal (leitura ou número de série; dígitos e
letras maiúsculas, V-a) e a função (só dígitos). O índice 0 é o branco da CTC; o
caractere de posição i no alfabeto recebe o índice i + 1.
"""

from __future__ import annotations

from collections.abc import Iterable

TEXT_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
FUNCTION_ALPHABET = "0123456789"
BLANK = 0
MAX_READING_LENGTH = 8   # maior leitura digital observada (T6b.1)
# Classes da cabeça da função (R11) e as que contam como consumo (R15, R18)
FUNCTION_CLASSES = ("vazio", "03", "003", "103", "24", "55", "outro")
CONSUMPTION_CLASSES = ("vazio", "03", "003")


def encode(text: str, alphabet: str) -> list[int]:
    """Texto -> índices da CTC (1 em diante). Recusa caractere fora do alfabeto."""
    out = []
    for char in text:
        position = alphabet.find(char)
        if position < 0:
            raise ValueError("Caractere fora do alfabeto do leitor.")
        out.append(position + 1)
    return out


def greedy_decode(steps: Iterable[int], alphabet: str) -> str:
    """Melhor caminho da CTC: colapsa repetições e remove os brancos."""
    chars, previous = [], BLANK
    for k in steps:
        k = int(k)
        if k != previous and k != BLANK:
            chars.append(alphabet[k - 1])
        previous = k
    return "".join(chars)


def reading_variants(value: int, max_length: int = MAX_READING_LENGTH) -> list[str]:
    """Formas que o display pode mostrar para uma leitura do CSV (Z-a): o número com 0
    a k zeros à esquerda, até `max_length` caracteres."""
    if value < 0:
        raise ValueError("Leitura negativa.")
    base = str(value)
    return [base.zfill(n) for n in range(len(base), max(len(base), max_length) + 1)]
