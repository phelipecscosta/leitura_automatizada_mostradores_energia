"""Testes da triagem de cinco classes do produto (T4g.4)."""

import numpy as np
import pytest
import torch

from meter_reader.contracts import PRODUCT_CLASSES, TRIAGE_GROUPS
from meter_reader import triage as tr
from training.triage_eval import class_log_probs
from training.triage_eval import rejection_score as eval_score


def test_net_has_the_parameter_names_of_the_trained_models():
    # Os pesos da ET4f e da ET4g foram gravados com estes nomes
    names = {k.split(".")[0] for k in tr.MobileNetTriageNet().state_dict()}
    assert names == {"backbone", "head"}
    assert tr.MobileNetTriageNet()(torch.zeros(1, 3, 480, 360)).shape == (1, len(TRIAGE_GROUPS))


def test_ensemble_is_mean_of_logits_over_temperature():
    torch.manual_seed(0)
    nets = [torch.nn.Linear(4, 12) for _ in range(3)]
    x = torch.randn(5, 4)
    expected = torch.stack([n(x) for n in nets]).mean(0) / 2.0
    assert torch.allclose(tr.TriageEnsemble(nets, 2.0)(x), expected)
    with pytest.raises(ValueError):
        tr.TriageEnsemble(nets, 0.0)


def test_class_probs_and_score_match_the_evaluation_code():
    # Produto (torch, FP32) e avaliação do estudo (NumPy, FP64) dão o mesmo número
    rng = np.random.default_rng(0)
    logits = rng.normal(scale=3.0, size=(50, len(TRIAGE_GROUPS)))
    probs = tr.class_probs(torch.tensor(logits, dtype=torch.float32))
    expected = np.exp(class_log_probs(logits))
    assert torch.allclose(probs, torch.tensor(expected, dtype=torch.float32), atol=1e-6)
    assert np.allclose(tr.rejection_score(probs).numpy(), eval_score(expected), atol=1e-6)


def test_destination_is_the_largest_rejectable_class():
    probs = torch.zeros(2, len(PRODUCT_CLASSES))
    probs[0, PRODUCT_CLASSES.index("multiplos")] = 0.6
    probs[1, PRODUCT_CLASSES.index("outros")] = 0.3
    probs[1, PRODUCT_CLASSES.index("digital")] = 0.7  # classe não rejeitável é ignorada
    assert tr.rejection_destination(probs) == ["multiplos", "outros"]
