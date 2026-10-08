"""Testes da exportação para ONNX (training/onnx_export.py)."""

import numpy as np
import torch

from meter_reader.triage import TriageModel
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
