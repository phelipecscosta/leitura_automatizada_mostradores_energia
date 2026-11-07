"""Testes da exportação para ONNX (training/onnx_export.py)."""

import numpy as np
import torch

from training.lab02_triage import TriageModel
from training.onnx_export import INPUT_NAME, cpu_session, export_onnx, run_onnx


def test_export_matches_pytorch_in_both_orientations(tmp_path):
    torch.manual_seed(0)
    model = TriageModel(pretrained=False)
    model.freeze_backbone()
    model.eval()
    path = export_onnx(model, tmp_path / "triagem.onnx", torch.randn(1, 3, 299, 224))
    session = cpu_session(path, threads=2)
    assert session.get_inputs()[0].name == INPUT_NAME

    # Em pé e deitada, com lotes de tamanhos diferentes
    for shape in [(1, 3, 299, 224), (2, 3, 224, 299)]:
        x = torch.randn(shape)
        with torch.no_grad():
            scene, illegible = model(x)
        onnx_scene, onnx_illegible = run_onnx(session, x.numpy())
        assert onnx_scene.shape == (shape[0], 3) and onnx_illegible.shape == (shape[0],)
        assert np.abs(onnx_scene - scene.numpy()).max() < 1e-4
        assert np.abs(onnx_illegible - illegible.numpy()).max() < 1e-4




import pytest  # noqa: E402

from meter_reader.triage import MobileNetTriageNet, TriageEnsemble  # noqa: E402
from training.finetune import DinoClassifier  # noqa: E402
from training.onnx_export import TRIAGE5_OUTPUT_NAMES, run_onnx_triage5  # noqa: E402


@pytest.mark.parametrize("net, shapes", [
    (MobileNetTriageNet, [(1, 3, 480, 360), (2, 3, 360, 480)]),            # C4, resolução nativa
    (lambda: DinoClassifier(12, pretrained=False), [(1, 3, 294, 224), (1, 3, 224, 294)]),  # C3
])
def test_triage5_ensemble_export_matches_pytorch(tmp_path, net, shapes):
    torch.manual_seed(0)
    ensemble = TriageEnsemble([net() for _ in range(2)], 1.4).eval()
    path = export_onnx(ensemble, tmp_path / "triagem5.onnx", torch.randn(*shapes[0]),
                       TRIAGE5_OUTPUT_NAMES)
    session = cpu_session(path, threads=2)
    for shape in shapes:  # em pé e deitada
        x = torch.randn(shape)
        with torch.no_grad():
            expected = ensemble(x).numpy()
        out = run_onnx_triage5(session, x.numpy())
        assert out.shape == (shape[0], 12)
        assert np.abs(out - expected).max() < 1e-4
