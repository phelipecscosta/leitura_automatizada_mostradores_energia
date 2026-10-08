"""Modelo final da triagem (ET3b, T3b.5).

As cabeças finais são uma única camada linear por saída, equivalente a:
média das sementes sobre a entrada padronizada, menos a correção dos pesos
de classe, dividida pela temperatura (protocolo do Lab02, seções 2.9 a 2.11).
Composição de funções lineares é linear: nada muda na inferência.
"""

from __future__ import annotations

import numpy as np
import torch
import json

from pathlib import Path
from training.build_manifest import file_sha256
from training.calibration import log_weights
from training.cross_validation import LinearHeads
from training.cross_validation import fit_heads
from training.labels import TEST_SUBSET, LabeledPhoto

HEADS_NAME = "triagem_cabecas.pt"
METADATA_NAME = "triagem_metadados.json"


def fold_heads(heads: list[LinearHeads], mean: np.ndarray, std: np.ndarray,
               scene_log_w: np.ndarray, leg_log_pw: float,
               t_scene: float, t_leg: float) -> LinearHeads:
    """Incorpora sementes, padronização, correção e temperatura em LinearHeads."""
    mu = torch.as_tensor(mean, dtype=torch.float64)
    sigma = torch.as_tensor(std, dtype=torch.float64)

    def fold(weights, biases, log_offset, temperature):
        w = torch.stack(weights).double().mean(0)   # média das sementes
        b = torch.stack(biases).double().mean(0)
        w = w / sigma                               # padronização: W(x - mu)/sigma
        b = b - w @ mu
        b = b - torch.as_tensor(log_offset, dtype=torch.float64)  # correção dos pesos
        return w / temperature, b / temperature                   # temperatura

    ws, bs = fold([h.scene_head.weight for h in heads], [h.scene_head.bias for h in heads],
                  scene_log_w, t_scene)
    wl, bl = fold([h.legibility_head.weight for h in heads],
                  [h.legibility_head.bias for h in heads], leg_log_pw, t_leg)

    folded = LinearHeads(len(mean))
    with torch.no_grad():
        folded.scene_head.weight.copy_(ws.float())
        folded.scene_head.bias.copy_(bs.float())
        folded.legibility_head.weight.copy_(wl.float())
        folded.legibility_head.bias.copy_(bl.float())
    return folded

def train_final(features: np.ndarray, photos: list[LabeledPhoto], seeds,
                t_scene: float, t_leg: float, device: str = "cpu") -> tuple[LinearHeads, dict]:
    """Cabeças finais treinadas em todos os lotes fora do teste, já incorporadas."""
    if any(p.subset == TEST_SUBSET for p in photos):
        raise ValueError("O lote de teste não entra no treino final (protocolo 2.1).")
    scene = np.array([p.scene for p in photos])
    illegible = np.array([p.illegible for p in photos])

    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std < 1e-6] = 1.0  # mesma regra da validação cruzada
    z = ((features - mean) / std).astype(np.float32)

    heads, final_losses = [], []
    for seed in seeds:
        h, history = fit_heads(z, scene, illegible, seed, device)
        heads.append(h.cpu())
        final_losses.append(history[-1])

    log_w, log_pw = log_weights(scene, illegible)
    folded = fold_heads(heads, mean, std, log_w, log_pw, t_scene, t_leg)
    info = {"fotos": len(photos), "lotes": sorted({p.lot for p in photos}),
            "sementes": list(seeds), "perda_final_treino": final_losses,
            "temperaturas": {"cena": t_scene, "legibilidade": t_leg}}
    return folded, info


def save_final(folder: Path, heads: LinearHeads, metadata: dict) -> Path:
    """Grava as cabeças e os metadados; devolve o caminho do JSON."""
    folder.mkdir(exist_ok=True)
    heads_path = folder / HEADS_NAME
    torch.save(heads.state_dict(), heads_path)
    record = {**metadata, "sha256_cabecas": file_sha256(heads_path)}
    json_path = folder / METADATA_NAME
    json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return json_path
