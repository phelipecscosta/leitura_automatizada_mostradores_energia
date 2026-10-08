"""Características para o D3: baseline manual e espinhas dorsais congeladas.

Protocolo do Lab02, seções 2.3 e 2.9. Cada conjunto é calculado uma vez e
guardado em cache (.npz) na pasta de trabalho, com os pseudônimos na mesma
ordem das linhas, para que o cache nunca seja usado com outra lista de fotos.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import laplace
from skimage.feature import hog

from meter_reader.preprocess import enhance, resize_short_side, to_tensor
from meter_reader.triage import TriageModel
from training.labels import LabeledPhoto

BASELINE_SHORT_SIDE = 224
HSV_BINS = (8, 4, 4)

# Nome do conjunto -> (espinha dorsal ou "baseline", lado menor; None = nativa)
FEATURE_SETS = {
    "baseline_224": ("baseline", 224),
    "mobilenet_v3_large_224": ("mobilenet_v3_large", 224),
    "mobilenet_v3_large_nativa": ("mobilenet_v3_large", None),
    "efficientnet_b0_224": ("efficientnet_b0", 224),
    "efficientnet_b0_nativa": ("efficientnet_b0", None),
}


def _model_image(path: Path, short_side: int | None) -> Image.Image:
    """A imagem que o modelo recebe: realce fixo e redimensionamento."""
    with Image.open(path) as image:
        return resize_short_side(enhance(image), short_side)


def baseline_vector(image: Image.Image) -> np.ndarray:
    """HOG (1.728) + histograma HSV (128) + qualidade (3) = 1.859 dimensões."""
    gray = np.asarray(image.convert("L"), dtype=np.float64) / 255.0
    hog_part = hog(gray, orientations=9, pixels_per_cell=(32, 32),
                   cells_per_block=(2, 2), block_norm="L2-Hys", feature_vector=True)

    hsv = np.asarray(image.convert("HSV")).reshape(-1, 3)
    hist, _ = np.histogramdd(hsv, bins=HSV_BINS, range=[(0, 256)] * 3)
    color_part = (hist / hist.sum()).ravel()

    quality_part = np.array([gray.mean(), gray.std(), laplace(gray).var()])
    return np.concatenate([hog_part, color_part, quality_part]).astype(np.float32)


def extract(photos: list[LabeledPhoto], name: str, device: str = "cpu",
            batch_size: int = 32, pretrained: bool = True) -> np.ndarray:
    """Matriz (n_fotos, dimensão) do conjunto indicado, na ordem de `photos`."""
    kind, short_side = FEATURE_SETS[name]
    if kind == "baseline":
        return np.stack([baseline_vector(_model_image(p.path, short_side)) for p in photos])

    model = TriageModel(kind, pretrained=pretrained)
    model.freeze_backbone()
    model.eval().to(device)
    chunks = []
    with torch.no_grad():
        for start in range(0, len(photos), batch_size):
            batch = photos[start:start + batch_size]
            x = torch.stack([to_tensor(_model_image(p.path, short_side)) for p in batch])
            chunks.append(model.embed(x.to(device)).cpu().numpy())
    return np.concatenate(chunks).astype(np.float32)


def load_or_extract(photos: list[LabeledPhoto], name: str, cache_dir: Path,
                    device: str = "cpu", pretrained: bool = True) -> np.ndarray:
    """Lê do cache se a lista de fotos for a mesma; senão, extrai e grava."""
    path = cache_dir / f"caracteristicas_{name}.npz"
    codes = np.array([p.pseudonym for p in photos])
    if path.is_file():
        cached = np.load(path)
        if np.array_equal(cached["codes"], codes):
            return cached["features"]
    features = extract(photos, name, device, pretrained=pretrained)
    np.savez(path, codes=codes, features=features)
    return features
