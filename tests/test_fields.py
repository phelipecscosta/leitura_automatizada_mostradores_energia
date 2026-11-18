"""Testes da seleção dos campos (spec v17, 10.1; revisão R4 do detector)."""
import numpy as np

from meter_reader import fields as f

DIG, CYC, KVH, SER = 1, 2, 3, 4
TOP = [100, 50, 200, 80]      # display de cima (consumo)
MID = [100, 100, 200, 130]    # display do meio (kvarh)
SERIAL = [110, 200, 190, 215] # número de série, embaixo


def sel(rows, cut=0.1):
    boxes = [r[0] for r in rows]
    return f.select_fields(boxes, [r[1] for r in rows], [r[2] for r in rows], cut)


def test_digital_by_the_most_confident_region():
    out = sel([(TOP, 0.9, DIG), (MID, 0.5, CYC)])
    assert out.family == "digital" and np.array_equal(out.consumption, TOP)


def test_classes_on_the_same_region_are_merged():
    # caixa de consumo e de kvarh sobre o mesmo display: uma região só, com a classe da mais confiante
    out = sel([(TOP, 0.8, CYC), ([101, 51, 201, 81], 0.6, KVH)])
    assert out.ambiguity is None and np.array_equal(out.consumption, TOP)


def test_kvarh_is_the_display_closest_to_the_serial_number():
    # mesmo com as classes trocadas pelo detector, vale a posição
    out = sel([(TOP, 0.9, KVH), (MID, 0.8, CYC), (SERIAL, 0.7, SER)])
    assert out.family == "ciclometrico" and np.array_equal(out.consumption, TOP)


def test_without_serial_the_class_decides():
    out = sel([(TOP, 0.9, CYC), (MID, 0.8, KVH)])
    assert np.array_equal(out.consumption, TOP) and out.ambiguity is None


def test_two_of_the_same_class_without_serial_is_ambiguous():
    out = sel([(TOP, 0.9, CYC), (MID, 0.8, CYC)])
    assert out.consumption is None and out.ambiguity == f.AMBIGUOUS_SAME_CLASS


def test_more_than_two_displays_is_ambiguous():
    out = sel([(TOP, 0.9, CYC), (MID, 0.8, KVH), ([100, 300, 200, 330], 0.7, CYC),
               (SERIAL, 0.9, SER)])
    assert out.ambiguity == f.AMBIGUOUS_MANY


def test_single_kvarh_has_no_consumption_display():
    out = sel([(MID, 0.9, KVH)])
    assert out.consumption is None and out.ambiguity == f.AMBIGUOUS_NO_CONSUMPTION


def test_cut_removes_boxes_below_it():
    out = sel([(TOP, 0.9, CYC), (MID, 0.05, CYC)], cut=0.1)
    assert out.ambiguity is None and np.array_equal(out.consumption, TOP)


def test_only_serial_number_has_no_family():
    out = sel([(SERIAL, 0.9, SER), (TOP, 0.05, CYC)], cut=0.1)
    assert out.family is None and out.consumption is None and out.serial is not None


def test_most_confident_serial_is_used():
    out = sel([(SERIAL, 0.6, SER), ([10, 10, 50, 20], 0.9, SER), (TOP, 0.9, CYC)])
    assert np.array_equal(out.serial, [10, 10, 50, 20])


def test_no_boxes():
    out = f.select_fields(np.zeros((0, 4)), [], [], 0.1)
    assert out.family is None and out.consumption is None and out.serial is None
