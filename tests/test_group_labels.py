"""Testes da tabela de tradução das pastas de cluster_Rotulado (T4e.1)."""

import re
import unicodedata
from collections import Counter

import pytest

from meter_reader.contracts import PRODUCT_CLASSES
from training.group_labels import (
    CONDITIONS, FOLDERS, GROUPS, LabelFolderError, folder_label, product_class,
)

_CODE = re.compile(r"^[a-z0-9_]+$")


def test_codes_are_short_ascii():
    for code in [*GROUPS, *CONDITIONS]:
        assert _CODE.match(code)


def test_every_folder_points_to_known_codes():
    assert len(FOLDERS) == 29  # pastas com fotos no inventário da T4e.1
    for group, condition in FOLDERS.values():
        assert group in GROUPS and condition in CONDITIONS


def test_every_group_has_a_root_folder_and_no_pair_repeats():
    pairs = list(FOLDERS.values())
    assert len(pairs) == len(set(pairs))
    assert {g for g, c in pairs if c == "raiz"} == set(GROUPS)


def test_groups_per_product_class():
    per_class = Counter(product_class(g) for g in GROUPS)
    assert per_class == {"ciclometrico": 3, "digital": 5, "indeterminado": 1,
                         "multiplos": 1, "outros": 4}
    assert set(per_class) == set(PRODUCT_CLASSES)


def test_folder_label_accepts_decomposed_accents():
    nfd = unicodedata.normalize("NFD", "Ciclométrico - disco")
    assert folder_label(nfd) == ("cic_disco", "raiz")


def test_unknown_folder_is_refused():
    with pytest.raises(LabelFolderError):
        folder_label("Digital 6 - Inexistente")

from training.group_labels import (  # noqa: E402
    MERGES, TRAINING_GROUP_CLASS, TRAINING_GROUPS, training_group,
)


def test_twelve_training_groups_after_merge():
    assert len(TRAINING_GROUPS) == 12
    assert not set(MERGES) & set(TRAINING_GROUPS)


def test_merge_never_crosses_product_class():
    for source, target in MERGES.items():
        assert target in TRAINING_GROUPS  # destino existe e não é fundido de novo
        assert product_class(source) == product_class(target)


def test_training_groups_ordered_by_product_class():
    # Classes em ordem não decrescente: os grupos de cada classe ficam contíguos
    assert list(TRAINING_GROUP_CLASS) == sorted(TRAINING_GROUP_CLASS)
    assert set(TRAINING_GROUP_CLASS) == set(range(len(PRODUCT_CLASSES)))


def test_training_group_applies_merge_and_refuses_unknown():
    assert training_group("dig5_pequenino") == "dig1_quadrado_branco"
    assert training_group("multiplos") == "multiplos"
    with pytest.raises(ValueError):
        training_group("dig6_inexistente")



def test_training_groups_come_from_the_product_and_match_the_derivation():
    from meter_reader import contracts
    from training import group_labels as gl
    assert gl.TRAINING_GROUPS is contracts.TRIAGE_GROUPS
    assert gl.TRAINING_GROUP_CLASS is contracts.TRIAGE_GROUP_CLASS
    assert all(PRODUCT_CLASSES[c] == gl.product_class(g)
               for g, c in zip(gl.TRAINING_GROUPS, gl.TRAINING_GROUP_CLASS))
