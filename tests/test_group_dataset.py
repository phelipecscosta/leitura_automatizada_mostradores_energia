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


from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from training.group_dataset import attach_paths, fold_indices  # noqa: E402

MANIFEST = [
    {"arquivo": "a.jpg", "lote": "2026-05-20", "pareada": "True"},
    {"arquivo": "b.jpg", "lote": "2026-05-21", "pareada": "False"},
    {"arquivo": "c.jpg", "lote": "2026-05-22", "pareada": "True"},
]
MAPPING = {"a.jpg": "F-00000000000a", "b.jpg": "F-00000000000b", "c.jpg": "F-00000000000c"}
PATHS = {name: Path("dados") / name for name in MAPPING}


def test_attach_paths_fills_paths(tmp_path):
    photos = attach_paths(read_group_labels(write(tmp_path, ROWS)), MANIFEST, MAPPING, PATHS)
    assert [p.path.name for p in photos] == ["a.jpg", "b.jpg", "c.jpg"]


@pytest.mark.parametrize("field, value, message", [
    ("lote", "2026-05-22", "1 de lote"),
    ("pareada", "True", "1 de pareamento"),
])
def test_attach_paths_refuses_manifest_divergence(tmp_path, field, value, message):
    manifest = [dict(r) for r in MANIFEST]
    manifest[1][field] = value
    with pytest.raises(GroupLabelFileError, match=message):
        attach_paths(read_group_labels(write(tmp_path, ROWS)), manifest, MAPPING, PATHS)


def test_attach_paths_refuses_missing_file_and_unknown_code(tmp_path):
    photos = read_group_labels(write(tmp_path, ROWS))
    with pytest.raises(GroupLabelFileError, match="1 sem arquivo"):
        attach_paths(photos, MANIFEST, MAPPING, {k: v for k, v in PATHS.items() if k != "c.jpg"})
    with pytest.raises(GroupLabelFileError, match="fora do manifesto"):
        attach_paths(photos, MANIFEST[:2], MAPPING, PATHS)


def test_fold_trains_with_orphans_and_evaluates_only_paired(tmp_path):
    photos = read_group_labels(write(tmp_path, ROWS))
    train, held = fold_indices(photos, "2026-05-22")
    assert train.tolist() == [0, 1]           # a órfã (índice 1) entra no treino
    assert held.tolist() == [2]
    train, held = fold_indices(photos, "2026-05-21")
    assert train.tolist() == [0, 2]
    assert held.size == 0                     # a única foto do lote é órfã
    with pytest.raises(ValueError):
        fold_indices(photos, "2026-07-03")


from training.group_dataset import check_group_sizes, internal_split  # noqa: E402
from training.group_labels import TRAINING_GROUPS as GROUPS12  # noqa: E402


def synthetic(n_per_group, orphans_per_group=2):
    """Fotos sintéticas: n pareadas e algumas órfãs por grupo de treino."""
    from training.group_dataset import GroupPhoto
    photos, k = [], 0
    for group in GROUPS12:
        for i in range(n_per_group + orphans_per_group):
            photos.append(GroupPhoto(f"F-{k:012x}", "2026-05-20", "treino",
                                     i < n_per_group, group, "raiz"))
            k += 1
    return photos


def test_group_sizes_pass_and_fail():
    assert set(check_group_sizes(synthetic(30, 0)).values()) == {30}
    with pytest.raises(GroupLabelFileError, match="menos de 30"):
        check_group_sizes(synthetic(29, 0))


def test_internal_split_is_stratified_fixed_and_without_orphans():
    photos = synthetic(40)
    train = np.arange(len(photos))
    fit, internal = internal_split(photos, train)
    # 10% de 40 pareadas = 4 por grupo; nenhuma órfã na parte interna
    assert len(internal) == 4 * len(GROUPS12)
    assert all(photos[i].paired for i in internal)
    assert sorted(np.concatenate([fit, internal]).tolist()) == train.tolist()
    # Mesma semente, mesma divisão; semente diferente, outra divisão
    assert np.array_equal(internal, internal_split(photos, train)[1])
    assert not np.array_equal(internal, internal_split(photos, train, seed=1)[1])
