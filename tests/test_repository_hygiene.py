"""Testes de higiene do repositório: proteção dos dados do cliente.

Verificam que o .gitignore bloqueia o que deve bloquear, que nenhum arquivo
proibido está versionado e que pastas com conteúdo não mantêm o .gitkeep.
"""

from __future__ import annotations

import subprocess
from pathlib import Path, PurePosixPath

import pytest

# Raiz do repositório: a pasta acima de tests/
REPO_ROOT = Path(__file__).resolve().parents[1]

# Casos do .gitignore: caminho relativo e se ele deve ser ignorado
GITIGNORE_CASES = [
    ("foto_medidor.jpg", True),                   # imagem na raiz
    ("FOTO_CELULAR.JPG", True),                   # extensão em maiúsculas
    ("notebooks/recorte_debug.png", True),        # imagem fora das pastas de dados
    ("tests/Controle_03072026.xlsx", True),       # planilha gerada por teste
    ("manifesto.csv", True),                      # tabela com nomes do cliente
    ("src/meter_reader/models/pesos.pt", True),   # peso salvo dentro do código
    ("features_dinov2.npy", True),                # embedding derivado das fotos
    ("relatorio_sr1.pdf", True),                  # documento
    ("data/qualquer_coisa.txt", True),            # pasta de dados na raiz
    (".env", True),                               # segredos
    (".env.example", False),                      # modelo de configuração
    ("src/meter_reader/models/rede.py", False),   # código em subpacote "models"
    ("lab-02/lab02.ipynb", False),                # entrega obrigatória da rubrica
]

# Extensões que nunca podem estar versionadas, nem com "git add -f"
FORBIDDEN_SUFFIXES = {
    ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff", ".heic",
    ".csv", ".tsv", ".xls", ".xlsx", ".xlsm", ".parquet", ".feather",
    ".pdf", ".doc", ".docx", ".pptx", ".html", ".htm",
    ".zip", ".7z", ".rar", ".tar", ".gz",
    ".pt", ".pth", ".ckpt", ".onnx", ".safetensors", ".bin", ".h5",
    ".npy", ".npz", ".pkl", ".pickle", ".joblib",
    ".key", ".pem", ".pfx",
}


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    """Executa um comando git na raiz do repositório."""
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )


def _tracked_files() -> list[PurePosixPath]:
    """Lista os arquivos versionados (inclusive os já preparados para commit)."""
    result = _git("ls-files")
    assert result.returncode == 0, result.stderr
    return [PurePosixPath(line) for line in result.stdout.splitlines() if line]


def _is_forbidden(path: PurePosixPath) -> bool:
    """Indica se um arquivo não pode estar no repositório."""
    if path.suffix.lower() in FORBIDDEN_SUFFIXES:
        return True
    # .env e variações com segredos; só o modelo .env.example é permitido
    return path.name == ".env" or (
        path.name.startswith(".env.") and path.name != ".env.example"
    )


@pytest.mark.parametrize(("path", "should_ignore"), GITIGNORE_CASES)
def test_gitignore_rules(path: str, should_ignore: bool) -> None:
    # check-ignore avalia o caminho pelas regras, sem exigir que o arquivo exista.
    # Código de saída: 0 = ignorado, 1 = não ignorado, outro = erro do git.
    result = _git("check-ignore", "-q", "--no-index", path)
    assert result.returncode in (0, 1), result.stderr
    assert (result.returncode == 0) == should_ignore


def test_no_forbidden_files_tracked() -> None:
    offenders = [str(p) for p in _tracked_files() if _is_forbidden(p)]
    assert not offenders, f"Arquivos proibidos no repositório: {offenders}"


def test_gitkeep_only_in_empty_folders() -> None:
    tracked = _tracked_files()
    stale = [
        str(keep)
        for keep in tracked
        if keep.name == ".gitkeep"
        # a pasta tem outro arquivo versionado, em qualquer nível abaixo dela
        and any(keep.parent in p.parents for p in tracked if p != keep)
    ]
    assert not stale, f"Apague estes .gitkeep, pois a pasta já tem conteúdo: {stale}"
    