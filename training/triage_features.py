"""Características congeladas com as vistas da aumentação única (C0 e C1).

Protocolo da triagem, seção 1.3 e revisão R2. Cada foto gera a original e
`n_views` vistas; a matriz é (fotos, 1 + vistas, dimensão), com a original na
posição 0. Sem TF32 (E33, E37): o treino usa as mesmas características que a
inferência em CPU. Cache em .npz, conferido pelos pseudônimos.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from meter_reader.preprocess import resize_short_side, to_tensor
from training.dino_features import load_dinov2, model_input
from training.lab02_triage import TriageModel
from training.triage_augment import photo_views

CACHE_VERSION = "fp32"
SHORT_SIDE = 224


def _mobilenet(pretrained: bool):
    model = TriageModel("mobilenet_v3_large", pretrained=pretrained)
    model.freeze_backbone()
    return model.eval(), model.embed, lambda view: to_tensor(resize_short_side(view, SHORT_SIDE))


def _dinov2(pretrained: bool):
    model = load_dinov2(pretrained)
    return model, model, model_input


# Nome do extrator -> construtor de (modelo, função de embedding, entrada de uma vista)
EXTRACTORS = {"mobilenet_v3_large_224": _mobilenet, "dinov2_224": _dinov2}


def extract_views(paths: list[Path], codes: list[str], name: str, n_views: int,
                  device: str = "cpu", pretrained: bool = True) -> np.ndarray:
    """Matriz (fotos, 1 + n_views, dimensão), na ordem de `paths`."""
    if len(paths) != len(codes):
        raise ValueError("Caminhos e pseudônimos com tamanhos diferentes.")
    model, embed, to_input = EXTRACTORS[name](pretrained)
    model.to(device)
    chunks = []
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")  # sem TF32 na atenção do ViT
    try:
        with torch.no_grad(), torch.backends.cudnn.flags(enabled=True, benchmark=False,
                                                         deterministic=False, allow_tf32=False):
            for path, code in zip(paths, codes):
                with Image.open(path) as image:
                    x = torch.stack([to_input(v) for v in photo_views(image, code, n_views)])
                chunks.append(embed(x.to(device)).cpu().numpy())
    finally:
        torch.set_float32_matmul_precision(previous)
    return np.stack(chunks).astype(np.float32)


def cache_path(cache_dir: Path, name: str, n_views: int) -> Path:
    return cache_dir / f"caracteristicas_triagem5_{name}_vistas{n_views}_{CACHE_VERSION}.npz"


def load_or_extract(paths: list[Path], codes: list[str], name: str, n_views: int,
                    cache_dir: Path, device: str = "cpu", pretrained: bool = True) -> np.ndarray:
    """Lê do cache se a lista de pseudônimos for a mesma; senão, extrai e grava."""
    path = cache_path(cache_dir, name, n_views)
    if path.is_file():
        cached = np.load(path)
        if np.array_equal(cached["codes"], np.array(codes)):
            return cached["features"]
    features = extract_views(paths, codes, name, n_views, device, pretrained)
    np.savez(path, codes=np.array(codes), features=features)
    return features
