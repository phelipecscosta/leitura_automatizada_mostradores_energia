"""Testes de fumaça: o pacote instala, importa e encontra o ambiente.

Não dependem dos dados do cliente. O único teste com a base real é pulado
quando ela não está configurada (por exemplo, em um clone limpo).
"""

from __future__ import annotations

import importlib.metadata
import tomllib
from pathlib import Path

import pytest
import torch

from meter_reader.config import DATA_DIR_VAR, ConfigError, get_data_dir

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_version_has_single_source() -> None:
    # A versão instalada precisa ser a mesma declarada no pyproject.toml.
    # Se divergirem, o pacote foi alterado sem ser reinstalado.
    with (REPO_ROOT / "pyproject.toml").open("rb") as f:
        declared = tomllib.load(f)["project"]["version"]
    assert importlib.metadata.version("meter-reader") == declared


def test_torch_runs_on_cpu() -> None:
    # Perfil CPU: o piso do cliente (E04) e o retorno seguro (E06)
    x = torch.rand(64, 64)
    assert (x @ x).shape == (64, 64)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="sem GPU NVIDIA com CUDA")
def test_torch_runs_on_gpu() -> None:
    # Perfil GPU (E06): só executa onde há GPU compatível
    x = torch.rand(64, 64, device="cuda")
    assert (x @ x).shape == (64, 64)


def test_data_dir_from_environment(tmp_path, monkeypatch) -> None:
    # Uma variável de ambiente válida é lida e devolvida como caminho absoluto
    monkeypatch.setenv(DATA_DIR_VAR, str(tmp_path))
    assert get_data_dir() == tmp_path.resolve()


def test_data_dir_missing_variable(tmp_path, monkeypatch) -> None:
    # Sem variável e sem .env, a função falha com uma mensagem clara.
    # chdir para uma pasta temporária: impede que o .env do projeto seja lido.
    monkeypatch.delenv(DATA_DIR_VAR, raising=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ConfigError, match="não definida"):
        get_data_dir()


def test_data_dir_nonexistent_folder(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(DATA_DIR_VAR, str(tmp_path / "pasta_inexistente"))
    with pytest.raises(ConfigError, match="inexistente"):
        get_data_dir()


def test_real_data_dir_has_four_lots() -> None:
    # Integração com a base real: só roda onde ela está configurada.
    # Mostra apenas a contagem, nunca nomes ou caminhos.
    try:
        data_dir = get_data_dir()
    except ConfigError:
        pytest.skip("base de dados não configurada nesta máquina")
    lots = [p for p in data_dir.iterdir() if p.is_dir()]
    assert len(lots) == 4