"""Pseudonimização de identificadores do cliente (T2.3; instruções 6.2).

A chave secreta fica na pasta de trabalho, junto da tabela de correspondência
(opção 1B). Perder ou sobrescrever a chave desliga os rótulos das fotos; por
isso ela só é criada por comando explícito e nunca é sobrescrita.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import sys
from pathlib import Path

from meter_reader.config import get_work_dir

KEY_NAME = "chave_pseudonimos.key"
KEY_BYTES = 32  # 256 bits, o tamanho da saída do SHA-256


class PseudonymError(RuntimeError):
    """Chave ausente, inválida ou já existente."""

# Prefixo de cada tipo de identificador (item 4 da T2.3)
PREFIXES = {"arquivo": "F-", "medidor": "M-"}
CODE_HEX_CHARS = 12  # 48 bits; colisões são verificadas, não supostas


def pseudonym(key: bytes, kind: str, value: str) -> str:
    """Código pseudônimo determinístico de um identificador.

    O tipo entra na mensagem do HMAC: o mesmo texto como arquivo e como
    medidor gera códigos diferentes, e os dois espaços não se cruzam.
    O valor é usado exatamente como está, com maiúsculas (regra do pareamento).
    """
    if kind not in PREFIXES:
        raise ValueError("Tipo de identificador desconhecido.")
    if not value:
        raise ValueError("Identificador vazio.")
    message = f"{kind}:{value}".encode("utf-8")
    digest = hmac.new(key, message, hashlib.sha256).hexdigest()
    return PREFIXES[kind] + digest[:CODE_HEX_CHARS]


def build_mapping(key: bytes, kind: str, values) -> dict[str, str]:
    """Tabela valor real -> pseudônimo. Falha se dois valores colidirem."""
    mapping: dict[str, str] = {}
    owners: set[str] = set()
    for value in sorted(set(values)):
        code = pseudonym(key, kind, value)
        if code in owners:
            # A mensagem não mostra os valores: eles são dados do cliente
            raise PseudonymError("Colisão de pseudônimos: aumente CODE_HEX_CHARS.")
        owners.add(code)
        mapping[value] = code
    return mapping

def create_key(path: Path) -> None:
    """Cria a chave como texto hexadecimal. Nunca sobrescreve uma existente."""
    try:
        # Modo "x": a criação falha se o arquivo já existir (verificação atômica)
        with path.open("x", encoding="ascii") as file:
            file.write(secrets.token_hex(KEY_BYTES))
    except FileExistsError:
        raise PseudonymError("A chave já existe e não será sobrescrita.") from None


def load_key(path: Path) -> bytes:
    """Lê e valida a chave. Falha cedo, sem exibir caminhos (E15)."""
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        raise PseudonymError(
            "Chave ausente. Crie-a com: python -m training.pseudonym criar-chave"
        ) from None
    try:
        # UnicodeDecodeError também é ValueError: cobre texto não ASCII
        key = bytes.fromhex(raw.decode("ascii").strip())
    except ValueError:
        raise PseudonymError("Chave com formato inválido.") from None
    if len(key) < KEY_BYTES:
        raise PseudonymError("Chave curta demais.")
    return key


def main(argv: list[str]) -> None:
    if argv != ["criar-chave"]:
        print("Uso: python -m training.pseudonym criar-chave")
        raise SystemExit(2)
    create_key(get_work_dir() / KEY_NAME)
    print("Chave criada na pasta de trabalho. Inclua-a no backup dessa pasta.")


if __name__ == "__main__":
    main(sys.argv[1:])
