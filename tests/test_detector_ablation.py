"""Testes da regra da ablação COCO (protocolo, 1.4)."""
import numpy as np

from training.detector_ablation import decide


def test_adopted_only_with_the_whole_interval_above_zero_and_viable():
    assert decide(np.linspace(0.01, 0.05, 2000), True)["adotada"]
    assert not decide(np.linspace(-0.01, 0.05, 2000), True)["adotada"]   # IC cruza o zero
    assert not decide(np.linspace(0.01, 0.05, 2000), False)["adotada"]   # não viável


def test_interval_and_mean_are_reported():
    out = decide(np.linspace(0.0, 1.0, 2001), True)
    assert np.isclose(out["diferenca_media"], 0.5)
    assert np.isclose(out["ic95"][0], 0.025) and np.isclose(out["ic95"][1], 0.975)
