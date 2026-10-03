"""Leitor do arquivo BaseExtracao: fronteira de entrada do motor.

Converte o texto bruto do SAP em registros validados (contracts.py).
As mensagens de erro não exibem caminhos (decisão E15).
"""

from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path

# A spec diz que o nome "sempre começa com" BaseExtracao_DDMMAAAA_Dia.csv
# (2.3). Aceita sufixos que o Windows acrescenta em downloads repetidos,
# como " (1)". Sem diferenciar maiúsculas, como o próprio Windows.
_FILE_NAME_RE = re.compile(r"^BaseExtracao_(\d{8})_Dia.*\.csv$", re.IGNORECASE)


def parse_batch_date(path: Path | str) -> date:
    """Extrai a data do lote (DDMMAAAA) do nome do arquivo BaseExtracao."""
    match = _FILE_NAME_RE.fullmatch(Path(path).name)
    if match is None:
        raise ValueError("Nome de arquivo fora do padrão BaseExtracao_DDMMAAAA_Dia.csv.")
    try:
        return datetime.strptime(match.group(1), "%d%m%Y").date()
    except ValueError as exc:
        # Oito dígitos que não formam uma data, como 32012026
        raise ValueError("Data inválida no nome do arquivo BaseExtracao.") from exc
