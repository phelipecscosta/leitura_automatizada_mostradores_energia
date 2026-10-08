"""Exportação da triagem para ONNX e sessão do ONNX Runtime em CPU (N4).

O ONNX Runtime é medido como alternativa ao PyTorch; o motor de inferência do
produto é decidido na ET9 (M3), quando o detector e o leitor existirem.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch.export import Dim

INPUT_NAME = "imagem"
OUTPUT_NAMES = ("logits_cena", "logit_ilegivel")
OPSET = 18


def export_onnx(model: torch.nn.Module, path: Path, example: torch.Tensor) -> Path:
    """Exporta com lote, altura e largura variáveis (há fotos em pé e deitadas)."""
    model.eval()
    torch.onnx.export(
        model, (example,), str(path), dynamo=True,
        input_names=[INPUT_NAME], output_names=list(OUTPUT_NAMES),
        dynamic_shapes=({0: Dim.AUTO, 2: Dim.AUTO, 3: Dim.AUTO},),
        opset_version=OPSET, external_data=False, verbose=False,
    )
    return Path(path)


def cpu_session(path: Path, threads: int) -> ort.InferenceSession:
    """Sessão só em CPU, com o número de threads fixado (piso da E04: 4 núcleos)."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def run_onnx(session: ort.InferenceSession, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(logits de cena, logit de ilegível) para um lote em float32."""
    scene, illegible = session.run(None, {INPUT_NAME: np.asarray(x, dtype=np.float32)})
    return scene, illegible
