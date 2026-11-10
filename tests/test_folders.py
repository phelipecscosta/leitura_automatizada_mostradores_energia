"""Testes das pastas do lote e da movimentação (arquivos sintéticos em pasta temporária)."""

from __future__ import annotations

import errno
from datetime import date

import pytest

from meter_reader import folders
from meter_reader.contracts import (REJECTION_CORRUPTED, TRIAGE_REJECTIONS, ControlRow,
                                    Reason, Rejection, Status)

REVIEW = ControlRow("b.jpg", None, None, None, Status.NEEDS_REVIEW, Reason.LOW_CONFIDENCE)
OTHER = ControlRow.rejected("c.jpg", TRIAGE_REJECTIONS["outros"])
CORRUPTED = ControlRow.rejected("d.jpg", REJECTION_CORRUPTED)
NOT_PROCESSED = ControlRow.rejected(
    "e.jpg", Rejection(Status.NOT_PROCESSED, Reason.NOT_IN_PARAMETERS))


def _source(tmp_path, *names):
    src = tmp_path / "origem"
    src.mkdir()
    for n in names:
        (src / n).write_bytes(f"conteudo {n}".encode())
    return src


def test_lot_dir_uses_batch_date(tmp_path) -> None:
    assert folders.lot_dir(tmp_path, date(2026, 5, 20)) == tmp_path / "20052026"


def test_structure_matches_spec(tmp_path) -> None:
    folders.create_structure(tmp_path / "lote")
    folders.create_structure(tmp_path / "lote")  # repetir não falha
    found = sorted(p.relative_to(tmp_path / "lote").as_posix()
                   for p in (tmp_path / "lote").rglob("*"))
    assert found == ["Aprovadas", "Rejeitadas", "Rejeitadas/Baixa qualidade",
                     "Rejeitadas/Corrompidas", "Rejeitadas/Múltiplos",
                     "Rejeitadas/Não listada", "Rejeitadas/Outros", "Verificar"]


def test_each_status_goes_to_its_folder(tmp_path) -> None:
    src = _source(tmp_path, "b.jpg", "c.jpg", "d.jpg", "e.jpg")
    lot = tmp_path / "lote"
    moved = folders.move_images(src, [REVIEW, OTHER, CORRUPTED, NOT_PROCESSED], lot)
    assert moved == [lot / "Verificar" / "b.jpg", lot / "Rejeitadas" / "Outros" / "c.jpg",
                     lot / "Rejeitadas" / "Corrompidas" / "d.jpg"]
    assert all(p.exists() for p in moved)
    # Só a "Não processada" fica na origem
    assert [p.name for p in src.iterdir()] == ["e.jpg"]


def test_existing_file_is_overwritten(tmp_path) -> None:
    src = _source(tmp_path, "b.jpg")
    lot = tmp_path / "lote"
    folders.create_structure(lot)
    (lot / "Verificar" / "b.jpg").write_bytes(b"antigo")
    folders.move_images(src, [REVIEW], lot)
    assert (lot / "Verificar" / "b.jpg").read_bytes() == b"conteudo b.jpg"


def test_cross_device_move_copies_then_deletes(tmp_path, monkeypatch) -> None:
    src = _source(tmp_path, "b.jpg")
    lot = tmp_path / "lote"
    real_replace = folders.os.replace
    calls = []

    def replace(a, b):
        calls.append((a, b))
        if len(calls) == 1:  # primeira tentativa: discos diferentes
            raise OSError(errno.EXDEV, "outro disco")
        return real_replace(a, b)

    monkeypatch.setattr(folders.os, "replace", replace)
    folders.move_images(src, [REVIEW], lot)
    assert (lot / "Verificar" / "b.jpg").read_bytes() == b"conteudo b.jpg"
    assert not (src / "b.jpg").exists()
    assert list((lot / "Verificar").iterdir()) == [lot / "Verificar" / "b.jpg"]  # sem temporário


def test_failed_copy_keeps_original(tmp_path, monkeypatch) -> None:
    src = _source(tmp_path, "b.jpg")

    def replace(a, b):
        raise OSError(errno.EXDEV, "outro disco")

    def broken_copy(a, b):
        raise OSError("disco cheio")

    monkeypatch.setattr(folders.os, "replace", replace)
    monkeypatch.setattr(folders.shutil, "copy2", broken_copy)
    with pytest.raises(OSError, match="disco cheio"):
        folders.move_images(src, [REVIEW], tmp_path / "lote")
    assert (src / "b.jpg").exists()  # a foto nunca é perdida


def test_missing_source_raises(tmp_path) -> None:
    src = _source(tmp_path)
    with pytest.raises(FileNotFoundError):
        folders.move_images(src, [REVIEW], tmp_path / "lote")


def test_stale_copies_are_reported_not_deleted(tmp_path) -> None:
    lot = tmp_path / "lote"
    folders.create_structure(lot)
    old = lot / "Verificar" / "c.jpg"
    old.write_bytes(b"antigo")
    (lot / "Rejeitadas" / "Outros" / "c.jpg").write_bytes(b"novo")
    assert folders.stale_copies([OTHER, REVIEW], lot) == [old]
    assert old.exists()
