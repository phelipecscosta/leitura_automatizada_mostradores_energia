"""Pseudonimização de identificadores do cliente (T2.3; instruções 6.2).

A chave secreta fica na pasta de trabalho, junto da tabela de correspondência
(opção 1B). Perder ou sobrescrever a chave desliga os rótulos das fotos; por
isso ela só é criada por comando explícito e nunca é sobrescrita.
"""

from __future__ import annotations

import csv
import hashlib
import hmac
import secrets
import sys
from pathlib import Path

from meter_reader.config import get_work_dir
from training.build_manifest import MANIFEST_NAME, file_sha256

KEY_NAME = "chave_pseudonimos.key"
KEY_BYTES = 32  # 256 bits, o tamanho da saída do SHA-256
TABLE_NAME = "tabela_pseudonimos.csv"


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


def write_table(key: bytes, manifest: Path, output: Path) -> tuple[int, int]:
    """Grava a tabela arquivo -> pseudônimo de todas as imagens do manifesto.

    Retorna (linhas do manifesto, entradas da tabela). Os dois números devem
    ser iguais; se não forem, há nomes de arquivo repetidos entre lotes.
    """
    with manifest.open(encoding="utf-8", newline="") as file:
        names = [row["arquivo"] for row in csv.DictReader(file)]
    mapping = build_mapping(key, "arquivo", names)
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("arquivo", "pseudonimo"))
        writer.writerows(sorted(mapping.items()))
    return len(names), len(mapping)

def lookup(table: Path, codes: list[str]) -> dict[str, str | None]:
    """Pseudônimo -> nome real, a partir da tabela. None quando não existe."""
    with table.open(encoding="utf-8", newline="") as file:
        inverse = {row["pseudonimo"]: row["arquivo"] for row in csv.DictReader(file)}
    return {code: inverse.get(code) for code in codes}

USAGE = "Uso: python -m training.pseudonym [criar-chave | tabela | consultar CÓDIGO...]"


def main(argv: list[str]) -> None:
    work_dir = get_work_dir()
    if argv == ["criar-chave"]:
        create_key(work_dir / KEY_NAME)
        print("Chave criada na pasta de trabalho. Inclua-a no backup dessa pasta.")
    elif argv == ["tabela"]:
        output = work_dir / TABLE_NAME
        rows, entries = write_table(
            load_key(work_dir / KEY_NAME), work_dir / MANIFEST_NAME, output
        )
        # Só contagens e impressões digitais; nunca nomes nem caminhos (E15)
        print(f"Imagens no manifesto: {rows} | entradas na tabela: {entries}")
        print(f"SHA-256 da tabela: {file_sha256(output)}")
    elif argv[:1] == ["consultar"] and len(argv) > 1:
        # Mostra nomes reais: uso só local, na auditoria presencial (opção D)
        for code, name in lookup(work_dir / TABLE_NAME, argv[1:]).items():
            print(f"{code}  {name or 'não encontrado'}")
    else:
        print(USAGE)
        raise SystemExit(2)


if __name__ == "__main__":
    main(sys.argv[1:])
