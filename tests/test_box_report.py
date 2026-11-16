"""Testes da conferência das caixas (T5.4), com dados sintéticos."""
import pandas as pd

from training import box_report as br
from training.box_labels import Box, PhotoRecord

A, B, C, D = ("F-" + ch * 12 for ch in "abcd")
WHEN = "2026-11-13 12:00:00"


def _sample():
    return pd.DataFrame({
        "pseudonimo": [A, B, C, D],
        "conjunto": ["desenvolvimento", "desenvolvimento", "teste", "teste"],
        "grupo": ["dig1_quadrado_branco", "cic_duas_leituras", "indeterminado", "cic_disco"],
        "condicao": ["raiz"] * 4,
        "determinado": [True, True, False, True],
    })


def _labels():
    photos = [PhotoRecord(A, 2, 9.0, WHEN), PhotoRecord(B, 2, 9.0, WHEN), PhotoRecord(C, 0, 3.0, WHEN)]
    boxes = [Box(A, "display_digital", 1, 1, 9, 9), Box(A, "numero_serie", 1, 1, 9, 9),
             Box(B, "display_ciclometrico", 1, 1, 9, 9), Box(B, "numero_serie", 1, 1, 9, 9)]
    return photos, boxes


def test_flags_rules():
    assert br.flags("dig1_quadrado_branco", True, {"display_digital", "numero_serie"}) == []
    assert br.flags("dig2_redondo_branco", True, {"display_kvarh"}) == [
        "digital_com_ciclometrico", "kvarh_fora_duas_leituras"]
    assert br.flags("cic_disco", True, {"display_digital"}) == ["ciclometrico_com_digital"]
    assert br.flags("cic_duas_leituras", True, {"display_ciclometrico"}) == ["duas_leituras_sem_kvarh"]
    assert br.flags("multiplos", False, {"display_kvarh"}) == []  # vizinho de duas leituras
    assert br.flags("cic_uma_leitura", True, set()) == ["determinado_sem_caixas"]
    assert br.flags("out_acesso", False, set()) == []
    assert br.flags("out_acesso", False, {"numero_serie"}) == ["outros_com_caixas"]
    assert br.flags("indeterminado", False, {"display_digital"}) == ["indeterminado_com_caixas"]
    assert br.flags("indeterminado", False, set()) == []
    assert br.flags("multiplos", False, set()) == []  # múltiplos ficam fora da revisão


def test_per_photo_counts_and_missing():
    frame = br.per_photo(_sample(), *_labels())
    row = frame.set_index("pseudonimo")
    assert row.loc[A, "caixas"] == 2 and row.loc[A, "display_digital"] == 1
    assert row.loc[C, "anotada"] and row.loc[C, "caixas"] == 0
    assert not row.loc[D, "anotada"]  # foto da amostra ainda sem registro


def test_summaries():
    frame = br.per_photo(_sample(), *_labels())
    classes = br.class_summary(frame)
    assert classes.loc[("desenvolvimento", "numero_serie"), "caixas"] == 2
    assert classes.loc[("desenvolvimento", "numero_serie"), "fotos"] == 2
    assert classes.loc[("teste", "display_digital"), "caixas"] == 0
    empty = br.empty_summary(frame)
    assert empty.loc[("teste", "indeterminado"), "sem_caixas"] == 1


def test_flagged_one_row_per_photo_with_rules():
    photos, boxes = _labels()
    boxes.append(Box(C, "display_digital", 1, 1, 9, 9))       # indeterminado com caixa
    photos[2] = PhotoRecord(C, 1, 3.0, WHEN)
    marks = br.flagged(br.per_photo(_sample(), photos, boxes)).set_index("pseudonimo")
    assert marks.loc[B, "regras"] == "duas_leituras_sem_kvarh"
    assert marks.loc[C, "regras"] == "indeterminado_com_caixas"
    assert A not in marks.index                                # sem divergência
    assert D not in marks.index                                # não anotada: não entra
    assert list(marks.reset_index().columns) == ["pseudonimo", "conjunto", "grupo", "condicao", "regras"]


def test_rules_have_descriptions():
    for group, classes in [("out_acesso", {"numero_serie"}), ("indeterminado", {"display_digital"}),
                           ("cic_disco", {"display_digital"}), ("cic_uma_leitura", set())]:
        assert set(br.flags(group, group.startswith("cic"), classes)) <= set(br.RULES)
