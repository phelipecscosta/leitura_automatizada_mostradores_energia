"""Testes do conversor das pastas de cluster_Rotulado (T4e.1)."""

import pytest

from training.build_group_labels import (
    GroupLabelError, build_group_labels, unlabeled_counts,
)

LOTS = ("2026-05-20", "2026-05-21", "2026-05-22", "2026-07-03")  # treino, treino, validação, teste


def _manifest():
    rows = []
    for lot in LOTS:
        for i in range(3):
            rows.append({"lote": lot, "arquivo": f"{lot}_{i}.jpg",
                         "integridade": "", "pareada": "True" if i else "False"})
    rows[1]["integridade"] = "vazia: preta"  # 2026-05-20_1 não é íntegra
    return rows


def _mapping(manifest):
    return {r["arquivo"]: f"F-{k:012x}" for k, r in enumerate(manifest)}


def test_rows_carry_set_pairing_and_codes_without_real_names():
    manifest = _manifest()
    found = [("Digital 1 - Quadrado - Branco", "2026-05-22_0.jpg"),
             ("Múltiplos", "2026-05-20_2.jpg")]
    rows = build_group_labels(found, manifest, _mapping(manifest))
    assert [r["conjunto"] for r in rows] == ["treino", "validacao"]
    by_group = {r["grupo"]: r for r in rows}
    assert by_group["dig1_quadrado_branco"]["pareada"] == "False"  # órfã da validação
    assert by_group["multiplos"]["condicao"] == "raiz"
    assert not any(".jpg" in str(v) for r in rows for v in r.values())


@pytest.mark.parametrize("found, message", [
    ([("Múltiplos", "2026-05-20_0.jpg"), ("Digital 3 - Vidro redondo", "2026-05-20_0.jpg")],
     "mais de uma pasta"),
    ([("Múltiplos", "inexistente.jpg")], "fora do manifesto"),
    ([("Múltiplos", "2026-07-03_0.jpg")], "lote de teste"),
    ([("Múltiplos", "2026-05-20_1.jpg")], "não íntegra"),
])
def test_inconsistent_folders_are_refused_without_names(found, message):
    manifest = _manifest()
    with pytest.raises(GroupLabelError, match=message) as error:
        build_group_labels(found, manifest, _mapping(manifest))
    assert ".jpg" not in str(error.value)


def test_unlabeled_counts_ignore_test_and_broken_photos():
    manifest = _manifest()
    mapping = _mapping(manifest)
    rows = build_group_labels([("Múltiplos", "2026-05-20_0.jpg")], manifest, mapping)
    # treino: 6 fotos, 1 não íntegra, 1 rotulada -> 4; validação: 3; teste: fora
    assert unlabeled_counts(rows, manifest, mapping) == {"treino": 4, "validacao": 3}

from training.labels import TEST_SUBSET


def test_rank_prefix_is_removed_only_when_the_rest_is_known():
    manifest = _manifest()
    rows = build_group_labels([("Múltiplos", "0007_2026-05-20_0.jpg")], manifest, _mapping(manifest))
    assert rows[0]["pseudonimo"] == _mapping(manifest)["2026-05-20_0.jpg"]
    # Nome truncado (como os 908 da T4e.1): continua recusado
    with pytest.raises(GroupLabelError, match="fora do manifesto"):
        build_group_labels([("Múltiplos", "0007_026-05-20_0.jpg")], manifest, _mapping(manifest))


def test_same_photo_with_two_rank_prefixes_is_refused():
    manifest = _manifest()
    found = [("Múltiplos", "0001_2026-05-20_0.jpg"), ("Digital 3 - Vidro redondo", "0002_2026-05-20_0.jpg")]
    with pytest.raises(GroupLabelError, match="mais de uma pasta"):
        build_group_labels(found, manifest, _mapping(manifest))


def test_test_part_accepts_only_paired_photos_of_the_test_lot():
    manifest = _manifest()
    mapping = _mapping(manifest)
    rows = build_group_labels([("Múltiplos", "2026-07-03_1.jpg")], manifest, mapping, TEST_SUBSET)
    assert rows[0]["conjunto"] == TEST_SUBSET
    with pytest.raises(GroupLabelError, match="fora do lote de teste"):
        build_group_labels([("Múltiplos", "2026-05-22_1.jpg")], manifest, mapping, TEST_SUBSET)
    with pytest.raises(GroupLabelError, match="órfã"):
        build_group_labels([("Múltiplos", "2026-07-03_0.jpg")], manifest, mapping, TEST_SUBSET)


def test_unlabeled_counts_in_test_part_use_only_paired_test_photos():
    manifest = _manifest()
    mapping = _mapping(manifest)
    rows = build_group_labels([("Múltiplos", "2026-07-03_1.jpg")], manifest, mapping, TEST_SUBSET)
    # teste: 2 pareadas íntegras (_1 e _2), 1 rotulada -> 1; a órfã (_0) fica fora
    assert unlabeled_counts(rows, manifest, mapping, TEST_SUBSET) == {TEST_SUBSET: 1}
