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



from training.build_triage import calibration, save_artifact  # noqa: E402


def tiny_ensemble():
    torch.manual_seed(0)
    return tr.TriageEnsemble([tr.MobileNetTriageNet() for _ in range(2)], 1.7).eval()


def test_artifact_roundtrip_gives_the_same_logits_and_threshold(tmp_path):
    ensemble = tiny_ensemble()
    path = tmp_path / "triagem_c4.pt"
    digest = save_artifact(path, ensemble, 0.83, {"candidato": "C4"})
    model = tr.load_triage(path, expected_sha256=digest)
    x = torch.randn(1, 3, 480, 360)
    with torch.no_grad():
        assert torch.equal(model.ensemble(x), ensemble(x))
    assert model.threshold == pytest.approx(0.83) and model.info["candidato"] == "C4"
    with pytest.raises(FileExistsError):
        save_artifact(path, ensemble, 0.83, {})


def test_load_refuses_altered_or_incompatible_artifacts(tmp_path):
    path = tmp_path / "triagem_c4.pt"
    save_artifact(path, tiny_ensemble(), 0.5, {})
    with pytest.raises(tr.TriageArtifactError):
        tr.load_triage(path, expected_sha256="0" * 64)   # SHA-256 diferente
    data = torch.load(path, weights_only=True)
    data["grupos"] = list(reversed(data["grupos"]))       # ordem das saídas trocada
    torch.save(data, path)
    with pytest.raises(tr.TriageArtifactError):
        tr.load_triage(path)


def test_calibration_returns_a_threshold_that_meets_the_target():
    # Logits sintéticos bem separados: há limiar que cumpre a meta com folga
    rng = np.random.default_rng(0)
    n = 3000
    labels = rng.choice([0, 1, 2, 3, 4], size=n, p=[0.8, 0.1, 0.04, 0.02, 0.04])
    groups = np.array([0, 4, 7, 8, 9])[labels]  # um grupo de cada classe
    logits = rng.normal(size=(n, len(TRIAGE_GROUPS)))
    logits[np.arange(n), groups] += 8.0
    lots = np.array(["a", "b", "c"])[np.arange(n) % 3]
    out = calibration(logits, labels, lots)
    assert out["perda_limite_superior"] <= 0.005 and 0 < out["cobertura"] <= 1
    assert out["temperatura"] > 0 and set(out["temperaturas_por_lote"]) == {"a", "b", "c"}




def test_artifact_with_numpy_values_in_info_loads_with_weights_only(tmp_path):
    # As temperaturas por lote saem da calibração como np.float64
    path = tmp_path / "triagem_c4.pt"
    save_artifact(path, tiny_ensemble(), 0.5, {"temperaturas_por_lote": {"a": np.float64(1.2)}})
    model = tr.load_triage(path)
    assert model.info["temperaturas_por_lote"]["a"] == pytest.approx(1.2)
