"""Testes da pasta de trabalho (somente pastas temporárias)."""

from __future__ import annotations

from pathlib import Path

import pytest

from meter_reader.config import DATA_DIR_VAR, WORK_DIR_VAR, ConfigError, get_work_dir


@pytest.fixture
def clean_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isola o teste: sem as variáveis reais e sem achar o .env do repositório."""
    monkeypatch.delenv(WORK_DIR_VAR, raising=False)
    monkeypatch.delenv(DATA_DIR_VAR, raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_work_dir_from_environment(clean_env: Path, monkeypatch) -> None:
    work = clean_env / "trabalho"
    work.mkdir()
    monkeypatch.setenv(WORK_DIR_VAR, str(work))
    assert get_work_dir() == work.resolve()


def test_work_dir_missing_variable(clean_env: Path) -> None:
    with pytest.raises(ConfigError):
        get_work_dir()


def test_work_dir_nonexistent_folder(clean_env: Path, monkeypatch) -> None:
    monkeypatch.setenv(WORK_DIR_VAR, str(clean_env / "inexistente"))
    with pytest.raises(ConfigError):
        get_work_dir()


def test_work_dir_inside_git_repository_is_rejected(clean_env: Path, monkeypatch) -> None:
    # Decisão 1 da T1.7: o manifesto nunca pode cair dentro de um repositório
    repo = clean_env / "repositorio"
    (repo / ".git").mkdir(parents=True)
    (repo / "trabalho").mkdir()
    monkeypatch.setenv(WORK_DIR_VAR, str(repo / "trabalho"))
    with pytest.raises(ConfigError, match="repositório"):
        get_work_dir()


def test_work_dir_inside_data_dir_is_rejected(clean_env: Path, monkeypatch) -> None:
    data = clean_env / "dados"
    (data / "trabalho").mkdir(parents=True)
    monkeypatch.setenv(DATA_DIR_VAR, str(data))
    monkeypatch.setenv(WORK_DIR_VAR, str(data / "trabalho"))
    with pytest.raises(ConfigError, match="dados"):
        get_work_dir()


def test_real_work_dir_is_valid() -> None:
    # Integração: pulado quando a pasta de trabalho não está configurada
    try:
        work = get_work_dir()
    except ConfigError:
        pytest.skip("Pasta de trabalho não configurada neste ambiente.")
    assert work.is_dir()



# ---------------------------------------------------------------------
# Raiz de saída e artefato da triagem (ET4d)
# ---------------------------------------------------------------------

from meter_reader.config import (OUTPUT_DIR_VAR, TRIAGE_ARTIFACT_VAR,  # noqa: E402
                                 get_output_dir, get_triage_artifact)


@pytest.fixture
def clean_output_env(clean_env: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.delenv(OUTPUT_DIR_VAR, raising=False)
    monkeypatch.delenv(TRIAGE_ARTIFACT_VAR, raising=False)
    return clean_env


def test_output_dir_from_environment(clean_output_env: Path, monkeypatch) -> None:
    out = clean_output_env / "Imagens processadas"
    out.mkdir()
    monkeypatch.setenv(OUTPUT_DIR_VAR, str(out))
    assert get_output_dir() == out.resolve()


def test_output_dir_missing_or_nonexistent(clean_output_env: Path, monkeypatch) -> None:
    with pytest.raises(ConfigError, match="não definida"):
        get_output_dir()
    monkeypatch.setenv(OUTPUT_DIR_VAR, str(clean_output_env / "inexistente"))
    with pytest.raises(ConfigError, match="inexistente"):
        get_output_dir()


def test_output_dir_inside_git_repository_is_rejected(clean_output_env: Path,
                                                      monkeypatch) -> None:
    # Fotos, planilhas e listas com nomes reais nunca podem cair num repositório
    repo = clean_output_env / "repositorio"
    (repo / ".git").mkdir(parents=True)
    (repo / "saida").mkdir()
    monkeypatch.setenv(OUTPUT_DIR_VAR, str(repo / "saida"))
    with pytest.raises(ConfigError, match="repositório"):
        get_output_dir()


def test_output_dir_inside_data_dir_is_rejected(clean_output_env: Path, monkeypatch) -> None:
    data = clean_output_env / "dados"
    (data / "saida").mkdir(parents=True)
    monkeypatch.setenv(DATA_DIR_VAR, str(data))
    monkeypatch.setenv(OUTPUT_DIR_VAR, str(data / "saida"))
    with pytest.raises(ConfigError, match="dados"):
        get_output_dir()


def test_triage_artifact_must_be_a_file(clean_output_env: Path, monkeypatch) -> None:
    artifact = clean_output_env / "triagem.pt"
    monkeypatch.setenv(TRIAGE_ARTIFACT_VAR, str(artifact))
    with pytest.raises(ConfigError, match="inexistente"):
        get_triage_artifact()
    artifact.write_bytes(b"x")
    assert get_triage_artifact() == artifact.resolve()
