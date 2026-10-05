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


class ConfigError(RuntimeError):
    """Configuração ausente ou inválida."""


def _read_dir(var: str) -> Path:
    """Lê uma variável de pasta e confirma que a pasta existe."""
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

    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        # A mensagem não mostra o caminho de propósito: erros exibidos em
        # notebooks ficam gravados nas saídas, e o caminho pode identificar
        # o cliente.
        raise ConfigError(f"{var} aponta para uma pasta inexistente.")
    return path


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

    if any((folder / ".git").exists() for folder in (path, *path.parents)):
        raise ConfigError(f"{WORK_DIR_VAR} não pode ficar dentro de um repositório Git.")

    data_value = os.getenv(DATA_DIR_VAR)
    if data_value:
        data_dir = Path(data_value).expanduser().resolve()
        if path == data_dir or data_dir in path.parents:
            raise ConfigError(
                f"{WORK_DIR_VAR} não pode ficar dentro da pasta de dados (somente leitura)."
            )
    return path
