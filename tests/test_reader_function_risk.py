from training.reader_function_risk import clopper_pearson


def test_intervalo_exato():
    low, high = clopper_pearson(0, 50)
    assert low == 0.0 and 0.06 < high < 0.08          # cerca de 3,7/50 pela regra do 3
    low, high = clopper_pearson(5, 10)
    assert 0.18 < low < 0.19 and 0.81 < high < 0.82
