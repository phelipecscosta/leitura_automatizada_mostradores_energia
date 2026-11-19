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

from types import SimpleNamespace

from training.crop_margin import collect, expand, kvarh_touched, margins


def _photo(boxes, labels, weight=1.0):
    return SimpleNamespace(boxes=tuple(boxes), labels=tuple(labels), weight=weight)


def _pred(boxes, scores, labels):
    return {"boxes": np.array(boxes, float), "scores": np.array(scores, float),
            "labels": np.array(labels)}


def test_collect_measures_the_box_the_reader_receives():
    # Display digital anotado em x de 0 a 100; previsto de 5 a 100 (falta 5 px de 95)
    photo = _photo([(0, 0, 100, 20)], [1])
    pred = _pred([(5, 0, 100, 20)], [0.9], [1])
    needs, weights, cyc = collect([photo], [pred], cut=0.19)
    assert len(needs["display_digital"]) == 1
    assert np.allclose(needs["display_digital"][0], [5 / 95, 0, 0, 0])
    assert len(needs["numero_serie"]) == 0 and cyc == []


def test_collect_skips_choice_that_is_not_the_annotated_field():
    # A caixa escolhida está longe do display anotado (IoU < 0,5): não entra na medida
    photo = _photo([(0, 0, 100, 20)], [1])
    pred = _pred([(300, 300, 400, 320)], [0.9], [1])
    needs, _, _ = collect([photo], [pred], cut=0.19)
    assert len(needs["display_digital"]) == 0


def test_margins_and_kvarh_check():
    needs = {"display_ciclometrico": np.array([[0.1, 0.0, 0.0, 0.2], [0.0, 0.0, 0.0, 0.0]])}
    weights = {"display_ciclometrico": np.array([1.0, 1.0])}
    mh, mv = margins(needs, weights, q=0.99)["display_ciclometrico"]
    assert (mh, mv) == (0.1, 0.2)
    assert np.allclose(expand((0, 0, 100, 10), 0.1, 0.2), [-10, -2, 110, 12])
    box = np.array([0, 0, 100, 10], float)
    near = [(box, np.array([[0, 11, 100, 21]], float), 1.0)]  # kvarh 1 px abaixo
    far = [(box, np.array([[0, 40, 100, 50]], float), 1.0)]
    assert kvarh_touched(near, 0.1, 0.2) == 1.0
    assert kvarh_touched(far, 0.1, 0.2) == 0.0
