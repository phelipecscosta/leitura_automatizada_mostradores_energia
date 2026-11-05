"""Testes do carregador dos rótulos da triagem de cinco classes (T4f.1)."""

import csv

import pytest

from meter_reader.contracts import PRODUCT_CLASSES
from training.build_group_labels import COLUMNS
from training.build_manifest import file_sha256
from training.group_dataset import GroupLabelFileError, read_group_labels
from training.group_labels import TRAINING_GROUPS

ROWS = [
    ("F-00000000000a", "2026-05-20", "treino", "True", "dig5_pequenino", "raiz"),
    ("F-00000000000b", "2026-05-21", "treino", "False", "multiplos", "raiz"),
    ("F-00000000000c", "2026-05-22", "validacao", "True", "out_poste", "raiz"),
]


def write(tmp_path, rows, columns=COLUMNS):
    path = tmp_path / "rotulos.csv"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        writer.writerows(rows)
    return path


def test_reads_rows_and_applies_merge(tmp_path):
    photos = read_group_labels(write(tmp_path, ROWS))
    assert [p.paired for p in photos] == [True, False, True]
    assert TRAINING_GROUPS[photos[0].target] == "dig1_quadrado_branco"
    assert PRODUCT_CLASSES[photos[0].product_class] == "digital"
    assert PRODUCT_CLASSES[photos[1].product_class] == "multiplos"
    assert TRAINING_GROUPS[photos[2].target] == "out_sem_contexto"


def test_checks_fingerprint(tmp_path):
    path = write(tmp_path, ROWS)
    assert len(read_group_labels(path, file_sha256(path))) == 3
    with pytest.raises(GroupLabelFileError, match="Impressão digital"):
        read_group_labels(path, "0" * 64)


def test_refuses_wrong_header(tmp_path):
    with pytest.raises(GroupLabelFileError, match="Cabeçalho"):
        read_group_labels(write(tmp_path, ROWS, columns=COLUMNS[::-1]))


@pytest.mark.parametrize("column, bad", [
    (0, "arquivo.jpg"), (1, "20/05/2026"), (2, "outro"),
    (3, "sim"), (4, "dig6"), (5, "longe"),
])
def test_refuses_invalid_value_without_showing_it(tmp_path, column, bad):
    rows = [list(r) for r in ROWS]
    rows[1][column] = bad
    with pytest.raises(GroupLabelFileError) as error:
        read_group_labels(write(tmp_path, rows))
    assert "Linha 3" in str(error.value) and bad not in str(error.value)


def test_refuses_repeated_pseudonym(tmp_path):
    with pytest.raises(GroupLabelFileError, match="repetido"):
        read_group_labels(write(tmp_path, [*ROWS, ROWS[0]]))
