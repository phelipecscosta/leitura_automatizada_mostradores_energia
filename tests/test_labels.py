"""Testes dos rótulos do Lab02 prontos para o treino (training/labels.py)."""

import csv

import pytest

from training import labels as lb

HEADER = (
    "nome_arquivo", "lote", "rotulador", "cena_n1", "cena_n2", "legibilidade",
    "incerto", "campos_visiveis", "leitura_confere", "observacao",
)


def write_labels(path, rows):
    """Grava um arquivo de rótulos sintético; colunas ausentes ficam vazias."""
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(HEADER)
        for row in rows:
            writer.writerow(row + ("",) * (len(HEADER) - len(row)))
    return path


def label(code, rater, n1, n2="", legibility=""):
    return (code, "2000-01-01", rater, n1, n2, legibility)


def test_adj_overrides_r1_and_r2_is_ignored(tmp_path):
    path = write_labels(tmp_path / "rotulos.csv", [
        label("F-1", "R1", "medidor", "digital", "legivel"),
        label("F-1", "R2", "medidor", "digital", "ilegivel"),
        label("F-1", "ADJ", "medidor", "ciclometrico", "legivel"),
        label("F-2", "R1", "outros"),
    ])
    final = lb.read_final_labels(path)
    assert set(final) == {"F-1", "F-2"}
    assert final["F-1"]["rotulador"] == "ADJ"
    assert final["F-1"]["cena_n2"] == "ciclometrico"


def test_adj_without_r1_fails(tmp_path):
    path = write_labels(tmp_path / "rotulos.csv", [
        label("F-1", "ADJ", "medidor", "digital", "legivel"),
    ])
    with pytest.raises(lb.LabelError):
        lb.read_final_labels(path)


def test_repeated_rater_fails(tmp_path):
    path = write_labels(tmp_path / "rotulos.csv", [
        label("F-1", "R1", "outros"),
        label("F-1", "R1", "outros"),
    ])
    with pytest.raises(lb.LabelError):
        lb.read_final_labels(path)


def test_unknown_rater_fails(tmp_path):
    path = write_labels(tmp_path / "rotulos.csv", [label("F-1", "R3", "outros")])
    with pytest.raises(lb.LabelError):
        lb.read_final_labels(path)


@pytest.mark.parametrize(("n1", "n2", "legibility", "expected"), [
    ("medidor", "digital", "legivel", (0, 0)),
    ("medidor", "ciclometrico", "ilegivel", (1, 1)),
    ("medidor", "indeterminado", "ilegivel", (lb.SCENE_METER_ONLY, 1)),
    ("outros", "", "", (2, lb.MASKED)),
])
def test_encode(n1, n2, legibility, expected):
    row = {"cena_n1": n1, "cena_n2": n2, "legibilidade": legibility}
    assert lb.encode(row) == expected


@pytest.mark.parametrize(("n1", "n2", "legibility"), [
    ("medidor", "", "legivel"),         # medidor sem tipo
    ("medidor", "digital", ""),         # medidor sem legibilidade
    ("outros", "digital", ""),          # outros com tipo
    ("outros", "", "ilegivel"),         # outros com legibilidade
    ("animal", "", ""),                 # cena fora do vocabulário
])
def test_encode_rejects_invalid(n1, n2, legibility):
    row = {"cena_n1": n1, "cena_n2": n2, "legibilidade": legibility}
    with pytest.raises(lb.LabelError):
        lb.encode(row)
