"""Paridade do detector: produto (CPU, foto a foto) × treino (GPU, em lote) (T5c.4; E66).

Roda o modelo final do S1_coco pelos dois caminhos nas fotos do desenvolvimento e
passa as saídas brutas pela mesma decisão do produto (DetectorModel.decide), para que
só o cálculo da rede seja comparado. Saídas: só contagens e medidas. Nenhuma foto do
teste é lida.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import torch

BOX_TOL = 0.5  # px: duas caixas de decisão são "a mesma" se nenhuma coordenada difere mais


def _same_box(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return bool(np.abs(np.asarray(a) - np.asarray(b)).max() <= BOX_TOL)


def compare_decisions(a, b) -> list[str]:
    """Itens de decisão em que dois DetectionResult divergem (lista vazia: iguais)."""
    out = []
    if a.passes_gate != b.passes_gate:
        return ["porta"]
    if a.family != b.family:
        out.append("familia")
    if a.fields is not None and b.fields is not None:
        if a.fields.ambiguity != b.fields.ambiguity:
            out.append("ambiguidade")
        if not _same_box(a.fields.consumption, b.fields.consumption):
            out.append("consumo")
        if not _same_box(a.fields.serial, b.fields.serial):
            out.append("numero_serie")
    return out


def main() -> None:
    from meter_reader.config import get_work_dir
    from meter_reader.detector import PRODUCT_DETECTOR_SHA256, load_detector
    from training.build_detector import ARTIFACT_NAME, FOLDER
    from training.detector_dataset import DetectorDataset, load_detector_photos
    from training.detector_train import DetectorTrainConfig, make_loader, predict
    from training.run_detector import FINAL_FOLDER, candidate, final_path

    work_dir = get_work_dir()
    product = load_detector(work_dir / FINAL_FOLDER / ARTIFACT_NAME, PRODUCT_DETECTOR_SHA256)
    photos = [p for p in load_detector_photos() if p.subset == "desenvolvimento"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Caminho do treino: o código de montagem do treino e os pesos finais gravados
    from training.run_detector import load_shapes
    shapes, _ = load_shapes(work_dir)
    model = candidate("S1").build(shapes["S1"]["formas"])
    model.load_state_dict(torch.load(final_path(work_dir, FOLDER), map_location="cpu",
                                     weights_only=True))
    model = model.to(device)
    cfg = DetectorTrainConfig()
    train_preds = predict(model, make_loader(DetectorDataset(photos, False), cfg, device, False),
                          device)
    print(f"Treino ({device}): {len(train_preds)} fotos preditas", flush=True)

    torch.set_num_threads(4)
    same_list, score_diff, box_diff, margin = 0, 0.0, 0.0, np.inf
    divergent = Counter()
    for photo, tp in zip(photos, train_preds):
        a = product.predict(photo.path)
        b = product.decide(tp["boxes"], tp["scores"], tp["labels"])
        if len(a.scores) == len(b.scores) and np.array_equal(a.labels, b.labels):
            same_list += 1
            if len(a.scores):
                score_diff = max(score_diff, float(np.abs(a.scores - b.scores).max()))
                box_diff = max(box_diff, float(np.abs(a.boxes - b.boxes).max()))
        margin = min(margin, abs(a.score - product.gate_threshold))
        for item in compare_decisions(a, b):
            divergent[item] += 1
    print(f"Fotos com a mesma lista de caixas e classes: {same_list} de {len(photos)}")
    print(f"Maior diferença de confiança: {score_diff:.2e}; de coordenada: {box_diff:.2e} px")
    print(f"Menor folga entre o escore da foto e o limiar da porta: {margin:.2e}")
    print("Decisões divergentes: " + (", ".join(f"{k} {v}" for k, v in divergent.items())
                                      if divergent else "nenhuma"))


if __name__ == "__main__":  # obrigatório no Windows: os processos auxiliares reimportam o módulo
    main()
