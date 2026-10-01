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


class ConfigError(RuntimeError):
    """Configuração ausente ou inválida."""


def get_data_dir() -> Path:
    """Retorna a pasta com os lotes originais, depois de validá-la.

    Regra do projeto: a pasta é somente leitura. Nenhum código pode mover,
    renomear ou alterar arquivos dentro dela.
    """
    # usecwd=True: procura o .env a partir da pasta onde o programa foi
    # iniciado, e não a partir deste arquivo. Assim, cada repositório
    # (produto ou pesquisa) usa o seu próprio .env.
    load_dotenv(find_dotenv(usecwd=True))

    value = os.getenv(DATA_DIR_VAR)
    if not value:
        raise ConfigError(
            f"Variável {DATA_DIR_VAR} não definida. "
            "Copie .env.example para .env e preencha o caminho."
        )

    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        # A mensagem não mostra o caminho de propósito: erros exibidos em
        # notebooks ficam gravados nas saídas, e o caminho pode identificar
        # o cliente.
        raise ConfigError(f"{DATA_DIR_VAR} aponta para uma pasta inexistente.")

    return path