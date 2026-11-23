import numpy as np

from meter_reader.reader_text import FUNCTION_ALPHABET
from training.reader_calibration import ece, fit_temperature, free_confidence, free_readings


def test_ece():
    assert abs(ece(np.array([0.9] * 10), np.array([1] * 9 + [0])) - 0.0) < 1e-9
    assert abs(ece(np.array([0.8] * 4), np.array([0, 0, 0, 0])) - 0.8) < 1e-9


def _outputs(n=200, steps=6, seed=0):
    """Saídas sintéticas confiantes demais: o dígito certo sempre domina o passo 2."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 1, (n, steps, len(FUNCTION_ALPHABET) + 1))
    logits[:, :, 0] += 8                                   # branco nos demais passos
    right = rng.random(n) < 0.6                            # só 60% certos
    digit = np.where(right, 1, 2)                          # índice 1 = "0", 2 = "1"
    logits[np.arange(n), 2, digit] += 12                   # muito confiante
    lp = logits - np.log(np.exp(logits).sum(-1, keepdims=True))
    return lp, right


def test_temperatura_reduz_excesso_de_confianca():
    lp, right = _outputs()
    texts = free_readings(lp, FUNCTION_ALPHABET)
    conf1 = free_confidence(lp, texts, FUNCTION_ALPHABET, 1.0)
    t = fit_temperature(lp, texts, right, FUNCTION_ALPHABET)
    conf_t = free_confidence(lp, texts, FUNCTION_ALPHABET, t)
    assert t > 1 and ece(conf_t, right) < ece(conf1, right)
    assert free_readings(lp / t, FUNCTION_ALPHABET) == texts    # a leitura não muda
