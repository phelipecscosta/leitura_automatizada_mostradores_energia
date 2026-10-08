"""Testes da curva cobertura × risco (training/risk_coverage.py)."""

import numpy as np

from training import risk_coverage as rc


def test_clopper_pearson_matches_closed_form_and_reference():
    # Zero erros: forma fechada 1 - alpha^(1/n)
    assert np.isclose(rc.clopper_pearson_upper(0, 60), 1 - 0.05 ** (1 / 60))
    # Valor de referência do scipy.stats.beta.ppf(0.95, 6, 95)
    assert np.isclose(rc.clopper_pearson_upper(5, 100), 0.1022533776, atol=1e-8)
    assert rc.clopper_pearson_upper(0, 0) == 1.0


def test_min_rejections_without_error():
    assert rc.min_rejections_without_error(0.10) == 29
    assert rc.min_rejections_without_error(0.05) == 59


def test_separable_scores_reach_full_coverage():
    # 40 rejeitáveis com escore alto, 100 medidores legíveis com escore baixo
    score = np.r_[np.full(40, 0.9), np.full(100, 0.1)]
    rejectable = np.r_[np.ones(40, bool), np.zeros(100, bool)]
    curve = rc.rejection_curve(score, rejectable, ~rejectable)
    row = rc.choose_threshold(curve, max_risk=0.10, max_loss=0.03)
    assert row is not None and row["coverage"] == 1.0 and row["losses"] == 0


def test_too_few_rejectables_make_the_goal_infeasible():
    score = np.r_[np.full(10, 0.9), np.full(100, 0.1)]
    rejectable = np.r_[np.ones(10, bool), np.zeros(100, bool)]
    curve = rc.rejection_curve(score, rejectable, ~rejectable)
    assert rc.choose_threshold(curve, max_risk=0.10, max_loss=0.03) is None


def test_curve_counts_are_consistent():
    rng = np.random.default_rng(0)
    score = rng.random(80)
    rejectable = score + rng.normal(0, 0.3, 80) > 0.6
    curve = rc.rejection_curve(score, rejectable, ~rejectable)
    assert curve["rejected"].is_monotonic_increasing
    assert curve["coverage"].is_monotonic_increasing
    assert curve.iloc[-1]["rejected"] == 80
