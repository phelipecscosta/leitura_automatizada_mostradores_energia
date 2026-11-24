import numpy as np

from training.reader_heads_train import FUNCTION_CLASSES
from training.reader_operating_point import gate_curve, lowest_safe


def test_curva_do_limiar_de_consumo():
    c03, c103 = FUNCTION_CLASSES.index("03"), FUNCTION_CLASSES.index("103")
    y = np.array([c03] * 4 + [c103] * 4)
    p = np.array([0.9, 0.8, 0.7, 0.3, 0.6, 0.2, 0.1, 0.05])
    curve = gate_curve(p, y, np.array([0.0, 0.5, 0.65]))
    assert curve["taxa_nao_consumo"].tolist() == [1.0, 0.25, 0.0]
    assert curve["consumo_mantido"].tolist() == [1.0, 0.75, 0.75]
    assert lowest_safe(curve, 0.02)["limiar"] == 0.65
