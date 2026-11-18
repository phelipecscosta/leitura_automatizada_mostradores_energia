"""Testes da comparação de decisões da paridade do detector (T5c.4)."""
import numpy as np

from meter_reader.detector import DetectionResult
from meter_reader.fields import FieldSelection
from training.detector_parity import compare_decisions

EMPTY = np.zeros((0, 4))


def _res(passes=True, family="ciclometrico", consumption=(0, 0, 10, 10), ambiguity=None):
    fields = FieldSelection(family, None if consumption is None else np.array(consumption, float),
                            None, ambiguity) if passes else None
    return DetectionResult(EMPTY, np.zeros(0), np.zeros(0, int), 0.5, passes,
                           family if passes else None, fields)


def test_equal_decisions_within_tolerance():
    assert compare_decisions(_res(), _res(consumption=(0.2, 0, 10.3, 10))) == []


def test_each_divergence_is_reported():
    assert compare_decisions(_res(), _res(passes=False)) == ["porta"]
    assert compare_decisions(_res(), _res(family="digital")) == ["familia"]
    assert compare_decisions(_res(), _res(consumption=(5, 5, 20, 20))) == ["consumo"]
    assert compare_decisions(_res(), _res(consumption=None, ambiguity="x")) == ["ambiguidade",
                                                                                  "consumo"]
