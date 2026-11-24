import math

import numpy as np

from meter_reader.reader_text import TEXT_ALPHABET
from training.reader_ensemble import ensemble_item, log_mean_exp


def test_media_das_probabilidades():
    assert math.isclose(log_mean_exp([math.log(0.2), math.log(0.6)]), math.log(0.4))
    assert log_mean_exp([-math.inf, -math.inf]) == -math.inf


def _out(text: str, steps: int = 10) -> np.ndarray:
    """Saída quase certa de um texto (um caractere por passo, brancos entre eles)."""
    lp = np.full((steps, len(TEXT_ALPHABET) + 1), -20.0)
    lp[:, 0] = 0.0
    for i, ch in enumerate(text):
        lp[2 * i + 1, 0], lp[2 * i + 1, TEXT_ALPHABET.index(ch) + 1] = -20.0, 0.0
    return lp


def test_comite_escolhe_a_leitura_da_maioria():
    lps = [_out("123"), _out("123"), _out("128")]
    out = ensemble_item(lps, "123", "display_digital", "chave")
    assert out["livre"] == "123" and out["acerto"]
    assert math.isclose(out["conf_livre"], math.log(2 / 3), rel_tol=1e-3)
