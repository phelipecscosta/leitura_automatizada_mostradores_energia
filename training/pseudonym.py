"""Pseudonimização de identificadores do cliente (T2.3; instruções 6.2).

A chave secreta fica na pasta de trabalho, junto da tabela de correspondência
(opção 1B). Perder ou sobrescrever a chave desliga os rótulos das fotos; por
isso ela só é criada por comando explícito e nunca é sobrescrita.
"""

from __future__ import annotations

import secrets
import sys
from pathlib import Path

from meter_reader.config import get_work_dir

KEY_NAME = "chave_pseudonimos.key"
KEY_BYTES = 32  # 256 bits, o tamanho da saída do SHA-256


class PseudonymError(RuntimeError):
    """Chave ausente, inválida ou já existente."""


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
