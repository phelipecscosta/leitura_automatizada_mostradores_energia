"""Testes do gerador do manifesto (lote sintético em pasta temporária)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from training.build_manifest import COLUMNS, file_sha256, manifest_rows, write_manifest

HEADER = "Numero do medidor;Posicao do medidor lida;Nota de Leitura Atual;Foto do medidor"
LISTED = "LOTE_00000000000000000001_000.jpg"
ORPHAN_BLACK = "LOTE_00000000000000000002_001.jpg"
BROKEN = "LOTE_00000000000000000003_000.jpg"


@pytest.fixture
def lot_dir(tmp_path: Path) -> Path:
    """Lote com uma foto listada, uma órfã preta e um arquivo corrompido."""
    lot = tmp_path / "LOTE"
    lot.mkdir()
    lines = [HEADER, f"B12345/777;321;T181;{LISTED}", "C67890;0;NA;NA"]
    (lot / "BaseExtracao_03072026_Dia.csv").write_bytes(("\r\n".join(lines) + "\r\n").encode())
    scene = np.random.default_rng(0).integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
    Image.fromarray(scene).save(lot / LISTED)
    Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8)).save(lot / ORPHAN_BLACK)
    (lot / BROKEN).write_bytes(b"")
    return lot


def test_one_row_per_image_with_rules_applied(lot_dir: Path) -> None:
    rows = {r["arquivo"]: r for r in manifest_rows(lot_dir)}
    assert set(rows) == {LISTED, ORPHAN_BLACK, BROKEN}

    listed = rows[LISTED]
    assert listed["lote"] == "2026-07-03" and listed["pareada"] is True
    assert (listed["numero_esperado"], listed["partes_numero"]) == ("B12345", 2)
    assert (listed["leitura"], listed["nota"], listed["integridade"]) == (321, "T181", "")
    assert len(listed["dhash"]) == 16

    black = rows[ORPHAN_BLACK]
    assert black["pareada"] is False and black["integridade"] == "vazia: preta"
    assert black["sufixo"] == "001"

    broken = rows[BROKEN]
    assert broken["integridade"] == "não abre ou não decodifica"
    assert broken["brilho"] == "" and broken["dhash"] == ""


def test_lot_without_single_extraction_file_is_rejected(lot_dir: Path) -> None:
    (lot_dir / "BaseExtracao_04072026_Dia.csv").write_text(HEADER)
    with pytest.raises(ValueError):
        manifest_rows(lot_dir)


def test_written_manifest_has_expected_columns(lot_dir: Path, tmp_path: Path) -> None:
    output = tmp_path / "manifesto.csv"
    write_manifest(manifest_rows(lot_dir), output)
    with output.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        assert tuple(reader.fieldnames) == COLUMNS
        assert len(list(reader)) == 3


def test_sha256_is_deterministic(lot_dir: Path, tmp_path: Path) -> None:
    # O mesmo lote gera o mesmo manifesto: a impressão digital se repete
    first, second = tmp_path / "a.csv", tmp_path / "b.csv"
    write_manifest(manifest_rows(lot_dir), first)
    write_manifest(manifest_rows(lot_dir), second)
    assert file_sha256(first) == file_sha256(second)
