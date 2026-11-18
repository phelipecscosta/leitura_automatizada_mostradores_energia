"""Testes da medida da seleção dos campos (revisão R4, K2)."""
import numpy as np

from meter_reader.fields import AMBIGUOUS_MANY, FieldSelection
from training import detector_fields_eval as fe

TOP, MID = [100, 50, 200, 80], [100, 100, 200, 130]
GT_BOXES, GT_LABELS = [TOP, MID], [2, 3]  # consumo em cima, kvarh no meio


def test_outcome_hit_and_kvarh_error():
    hit = FieldSelection("ciclometrico", np.array(TOP, float), None)
    wrong = FieldSelection("ciclometrico", np.array(MID, float), None)
    assert fe.photo_outcome(hit, GT_BOXES, GT_LABELS) == "acerto"
    assert fe.photo_outcome(wrong, GT_BOXES, GT_LABELS) == "erro_kvarh"


def test_outcome_ambiguous_and_other():
    amb = FieldSelection("ciclometrico", None, None, AMBIGUOUS_MANY)
    none = FieldSelection(None, None, None)
    off = FieldSelection("ciclometrico", np.array([0, 0, 10, 10], float), None)
    assert fe.photo_outcome(amb, GT_BOXES, GT_LABELS) == AMBIGUOUS_MANY
    assert fe.photo_outcome(none, GT_BOXES, GT_LABELS) == "outro"
    assert fe.photo_outcome(off, GT_BOXES, GT_LABELS) == "outro"


def test_choose_cut_breaks_ties_with_the_larger_cut():
    curve = {0.1: {"acerto": 0.8}, 0.2: {"acerto": 0.9}, 0.3: {"acerto": 0.9}, 0.4: {"acerto": 0.7}}
    assert fe.choose_cut(curve) == 0.3


def test_grid_is_fixed():
    assert fe.CUT_GRID[0] == 0.01 and fe.CUT_GRID[-1] == 0.95 and len(fe.CUT_GRID) == 95

def _pred(rows):
    return {"boxes": np.array([r[0] for r in rows], float).reshape(-1, 4),
            "scores": np.array([r[1] for r in rows], float),
            "labels": np.array([r[2] for r in rows], int)}


def test_other_cause_digital_family():
    sel = FieldSelection("digital", np.array(TOP, float), None)
    assert fe.other_cause(sel, _pred([(TOP, 0.9, 1)]), GT_BOXES, GT_LABELS)[0] == "o1_familia_digital"


def test_other_cause_below_cut_and_not_detected():
    none = FieldSelection(None, None, None)
    assert fe.other_cause(none, _pred([(TOP, 0.05, 2)]), GT_BOXES, GT_LABELS)[0] == "o2_abaixo_do_corte"
    assert fe.other_cause(none, _pred([([0, 0, 5, 5], 0.05, 2)]), GT_BOXES,
                          GT_LABELS)[0] == "o3_nao_detectado"


def test_other_cause_shifted_and_elsewhere():
    shifted = np.array([100, 60, 200, 95], float)  # IoU com TOP entre 0,3 e 0,5
    cause, frac = fe.other_cause(FieldSelection("ciclometrico", shifted, None), _pred([]),
                                 GT_BOXES, GT_LABELS)
    assert cause == "o4_deslocada" and 0.0 < frac < 1.0
    far = FieldSelection("ciclometrico", np.array([300, 300, 400, 330], float), None)
    assert fe.other_cause(far, _pred([]), GT_BOXES, GT_LABELS)[0] == "o5_outro_lugar"
