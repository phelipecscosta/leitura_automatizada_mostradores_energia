import numpy as np
import pytest

from training.crop_margin import expansion_needed, weighted_quantile


def test_contained_box_needs_nothing():
    # Anotada dentro da prevista: nenhum lado precisa crescer
    assert np.allclose(expansion_needed((0, 0, 100, 20), (10, 2, 90, 18)), 0.0)


def test_expansion_is_fraction_of_predicted_size():
    # Anotada passa 10 px à esquerda (largura 100) e 5 px abaixo (altura 20)
    need = expansion_needed((10, 0, 110, 20), (0, 0, 100, 25))
    assert np.allclose(need, [0.10, 0.0, 0.0, 0.25])


def test_zero_area_prediction_is_refused():
    with pytest.raises(ValueError):
        expansion_needed((5, 5, 5, 10), (0, 0, 10, 10))


def test_weighted_quantile_follows_weights():
    assert weighted_quantile([1, 2, 3], [1, 1, 1], 0.5) == 2
    assert weighted_quantile([1, 2, 3], [1, 1, 8], 0.5) == 3
    assert np.isnan(weighted_quantile([], [], 0.5))
