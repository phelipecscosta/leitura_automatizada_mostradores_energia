"""Testes do universo do agrupamento da triagem de ML, com dados sintéticos."""

from pathlib import Path

from training.unlabeled import development_photos

LOTS = ("2026-05-20", "2026-05-21", "2026-05-22", "2026-07-03")  # o último é o teste


def _row(lot: str, name: str, integrity: str = "", paired: str = "True") -> dict:
    return {"lote": lot, "arquivo": name, "integridade": integrity, "pareada": paired}


def _universe(rows: list[dict]):
    names = [r["arquivo"] for r in rows]
    mapping = {n: f"F-{9 - i:03d}" for i, n in enumerate(names)}  # ordem inversa à do manifesto
    paths = {n: Path("dados") / n for n in names}
    return development_photos(rows, mapping, paths)


def test_keeps_intact_development_photos_including_orphans() -> None:
    rows = [
        _row(LOTS[0], "a.jpg"),
        _row(LOTS[2], "b.jpg", paired="False"),       # órfã da validação: entra
        _row(LOTS[1], "c.jpg", integrity="vazia"),     # não íntegra: sai
        _row(LOTS[3], "d.jpg"),                        # lote de teste: sai
    ]
    photos = _universe(rows)
    assert [p.path.name for p in photos] == ["b.jpg", "a.jpg"]  # ordenadas pelo pseudônimo
    assert {p.lot for p in photos} == {LOTS[0], LOTS[2]}


def test_never_includes_the_test_lot() -> None:
    rows = [_row(lot, f"{i}.jpg") for i, lot in enumerate(LOTS)]
    assert all(p.lot != LOTS[3] for p in _universe(rows))
