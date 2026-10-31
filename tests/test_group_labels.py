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
