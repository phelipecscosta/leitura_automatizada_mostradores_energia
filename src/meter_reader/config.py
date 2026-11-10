"""Leitura da configuração do ambiente.

A configuração vem de variáveis de ambiente. Em desenvolvimento, elas são
preenchidas pelo arquivo .env na raiz do projeto, que nunca é versionado.
Variáveis já definidas no ambiente têm prioridade sobre o .env.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

DATA_DIR_VAR = "METER_READER_DATA_DIR"
WORK_DIR_VAR = "METER_READER_WORK_DIR"
OUTPUT_DIR_VAR = "METER_READER_OUTPUT_DIR"
TRIAGE_ARTIFACT_VAR = "METER_READER_TRIAGE_ARTIFACT"


class ConfigError(RuntimeError):
    """Configuração ausente ou inválida."""


def _read_path(var: str) -> Path:
    """Lê uma variável de caminho, do ambiente ou do .env."""
    # usecwd=True: procura o .env a partir da pasta onde o programa foi
    # iniciado, e não a partir deste arquivo. Assim, cada repositório
    # (produto ou pesquisa) usa o seu próprio .env.
    load_dotenv(find_dotenv(usecwd=True))

    value = os.getenv(var)
    if not value:
        raise ConfigError(
            f"Variável {var} não definida. "
            "Copie .env.example para .env e preencha o caminho."
        )
    return Path(value).expanduser().resolve()


def _read_dir(var: str) -> Path:
    """Lê uma variável de pasta e confirma que a pasta existe."""
    path = _read_path(var)
    if not path.is_dir():
        # A mensagem não mostra o caminho de propósito: erros exibidos em
        # notebooks ficam gravados nas saídas, e o caminho pode identificar
        # o cliente.
        raise ConfigError(f"{var} aponta para uma pasta inexistente.")
    return path


def _inside_git(path: Path) -> bool:
    return any((folder / ".git").exists() for folder in (path, *path.parents))


def _inside_data_dir(path: Path) -> bool:
    """Verdadeiro se o caminho estiver na pasta de dados (somente leitura), quando definida."""
    data_value = os.getenv(DATA_DIR_VAR)
    if not data_value:
        return False
    data_dir = Path(data_value).expanduser().resolve()
    return path == data_dir or data_dir in path.parents


def get_data_dir() -> Path:
    """Retorna a pasta com os lotes originais, depois de validá-la.

    Regra do projeto: a pasta é somente leitura. Nenhum código pode mover,
    renomear ou alterar arquivos dentro dela.
    """
    return _read_dir(DATA_DIR_VAR)


def get_work_dir() -> Path:
    """Retorna a pasta de trabalho local, depois de validá-la.

    Guarda artefatos derivados dos dados do cliente (o manifesto e, a partir
    da ET2, a chave de pseudonimização). Por isso fica fora de qualquer
    repositório Git e fora da pasta de dados, que é somente leitura.
    """
    path = _read_dir(WORK_DIR_VAR)
    if _inside_git(path):
        raise ConfigError(f"{WORK_DIR_VAR} não pode ficar dentro de um repositório Git.")
    if _inside_data_dir(path):
        raise ConfigError(
            f"{WORK_DIR_VAR} não pode ficar dentro da pasta de dados (somente leitura)."
        )
    return path


def get_output_dir() -> Path:
    """Retorna a raiz "Imagens processadas", depois de validá-la (spec 2.3; D1 b).

    Recebe as fotos movidas, as planilhas de controle e as listas de cópias
    antigas, todas com nomes reais de fotos. Por isso fica fora de qualquer
    repositório Git e fora da pasta de dados, que é somente leitura.
    """
    path = _read_dir(OUTPUT_DIR_VAR)
    if _inside_git(path):
        raise ConfigError(f"{OUTPUT_DIR_VAR} não pode ficar dentro de um repositório Git.")
    if _inside_data_dir(path):
        raise ConfigError(
            f"{OUTPUT_DIR_VAR} não pode ficar dentro da pasta de dados (somente leitura)."
        )
    return path


def get_triage_artifact() -> Path:
    """Retorna o caminho do artefato da triagem (o SHA-256 é conferido ao carregar)."""
    path = _read_path(TRIAGE_ARTIFACT_VAR)
    if not path.is_file():
        raise ConfigError(f"{TRIAGE_ARTIFACT_VAR} aponta para um arquivo inexistente.")
    return path
