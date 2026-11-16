"""Configuração comum da suíte do produto (ET5b, T5b.0).

Os testes marcados com @pytest.mark.lento (as exportações reais para ONNX) só
rodam com a opção --lentos. Regra A2: a suíte com --lentos roda antes de cada
push e em todo commit que altera um dos módulos cobertos por eles:
training/onnx_export.py, src/meter_reader/triage.py, training/finetune.py e
training/lab02_triage.py. Sem a opção, eles aparecem como "skipped" no resumo,
para que fique visível que não rodaram.
"""
from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--lentos", action="store_true", default=False,
                     help="Roda também os testes marcados como lentos.")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Sem --lentos, marca como pulados os testes com a etiqueta lento."""
    if config.getoption("--lentos"):
        return
    skip = pytest.mark.skip(reason="Teste lento: rode com --lentos (regra A2).")
    for item in items:
        if "lento" in item.keywords:
            item.add_marker(skip)
