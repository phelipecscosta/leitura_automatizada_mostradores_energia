"""Testes das métricas da abertura do teste, com dados sintéticos."""

import numpy as np
import pytest

from training.holdout import (
    OUTROS,
    clopper_pearson_interval,
    dl_decisions,
    dl_report,
    ml_report,
    proportion,
)
from training.labels import MASKED


def test_clopper_pearson_interval_known_values() -> None:
    low, high = clopper_pearson_interval(0, 74)
    assert low == 0.0 and high == pytest.approx(0.0486, abs=1e-3)   # 1 - 0,025^(1/74)
    low, high = clopper_pearson_interval(74, 74)
    assert high == 1.0 and low == pytest.approx(1 - 0.0486, abs=1e-3)
    low, high = clopper_pearson_interval(5, 10)
    assert low == pytest.approx(0.187, abs=1e-3) and high == pytest.approx(0.813, abs=1e-3)


def test_proportion_of_empty_set_is_nan() -> None:
    assert np.isnan(proportion(0, 0)["valor"])


def test_dl_decisions_threshold_and_destination() -> None:
    big = 20.0
    scene = np.array([[big, 0, 0], [0, 0, big], [big, 0, 0], [big, 0, 0]])  # outros na 2ª
    ill = np.array([-big, -big, big, -big])                                  # 3ª ilegível
    rejected, to_outros = dl_decisions(scene, ill, threshold=0.966)
    assert rejected.tolist() == [False, True, True, False]
    assert to_outros.tolist() == [False, True, False, False]   # ilegível vai a Baixa qualidade


def test_dl_report_counts() -> None:
    scene = np.array([0, 0, 1, OUTROS, OUTROS])
    illegible = np.array([0, 1, 0, MASKED, MASKED])
    rejected = np.array([True, True, False, True, False])     # rejeitou 1 legível (erro)
    to_outros = np.array([False, False, False, True, False])
    report = dl_report(rejected, to_outros, scene, illegible)
    assert report["cobertura_rejeicao"]["k"] == 2 and report["cobertura_rejeicao"]["n"] == 3
    assert (report["risco"]["k"], report["risco"]["n"]) == (1, 3)
    assert (report["perda"]["k"], report["perda"]["n"]) == (1, 2)
    assert (report["revocacao_medidor"]["k"], report["revocacao_medidor"]["n"]) == (3, 3)


def test_ml_report_counts() -> None:
    is_outros = np.array([0, 0, 0, 1])
    predicted = np.array([True, False, False, True])
    report = ml_report(predicted, is_outros)
    assert (report["revocacao_medidor"]["k"], report["revocacao_medidor"]["n"]) == (2, 3)
    assert (report["perda"]["k"], report["outros_rejeitados"]) == (1, 1)
