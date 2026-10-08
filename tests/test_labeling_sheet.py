"""Testes da folha de rotulagem do Lab02 (training/labeling_sheet.py)."""

from training.build_sample import ROLE_BLIND, ROLE_LABEL, ROLE_ZERO
from training.labeling_sheet import SHEET_COLUMNS, build_sheet


def _sample(lot, position, name, role, stress=""):
    return {"lote": lot, "conjunto": "treino", "posicao": position,
            "arquivo": name, "papel": role, "posicao_estresse": stress}


SAMPLE = [
    _sample("2000-01-02", "2", "b.jpg", ROLE_LABEL),
    _sample("2000-01-01", "1", "a.jpg", ROLE_BLIND),
    _sample("2000-01-01", "90", "z.jpg", ROLE_ZERO, stress="1"),
    _sample("2000-01-01", "3", "n.jpg", ""),          # não sorteada: fica de fora
    _sample("2000-01-02", "1", "c.jpg", ROLE_BLIND),
]
NAMES = [s["arquivo"] for s in SAMPLE]
MAPPING = {n: f"F-{n[0] * 12}" for n in NAMES}       # pseudônimos fictícios
READINGS = {n: "0" for n in NAMES}
PATHS = {n: f"lote/{n}" for n in NAMES}


def test_order_and_exclusion():
    rows = build_sheet(SAMPLE, READINGS, MAPPING, PATHS)
    # Às cegas primeiro, alternando lotes; depois rotulagem; depois estresse
    assert [r["pseudonimo"] for r in rows] == [
        MAPPING["a.jpg"], MAPPING["c.jpg"], MAPPING["b.jpg"], MAPPING["z.jpg"],
    ]


def test_row_content():
    row = build_sheet(SAMPLE, READINGS, MAPPING, PATHS)[0]
    assert tuple(row) == SHEET_COLUMNS
    assert row["pseudonimo"] == MAPPING["a.jpg"]
    assert row["caminho"] == "lote/a.jpg"
    assert "a.jpg" not in (row["pseudonimo"] + row["lote"] + row["papel"])
