"""Pré-processamento da entrada da triagem (spec 2.5.7; E19).

Ordem: abre em RGB -> realce fixo (autocontraste, 1% em cada ponta) na foto
original -> redimensiona mantendo a proporção -> tensor normalizado pelo
ImageNet. O realce vem antes do redimensionamento para reproduzir a imagem
que o rotulador viu ao decidir a legibilidade.
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision.transforms import functional as TF

AUTOCONTRAST_CUTOFF = 1  # percentual cortado em cada ponta do histograma (E19)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def enhance(image: Image.Image) -> Image.Image:
    """Realce fixo usado na rotulagem e obrigatório no pré-processamento."""
    return ImageOps.autocontrast(image.convert("RGB"), cutoff=AUTOCONTRAST_CUTOFF)


def resize_short_side(image: Image.Image, short_side: int | None) -> Image.Image:
    """Redimensiona para o lado menor indicado, mantendo a proporção.

    None mantém a resolução nativa.
    """
    if short_side is None:
        return image
    width, height = image.size
    scale = short_side / min(width, height)
    size = (round(width * scale), round(height * scale))
    return image.resize(size, Image.Resampling.BILINEAR)


def to_tensor(image: Image.Image) -> torch.Tensor:
    """Imagem RGB -> tensor (3, A, L) normalizado pelo ImageNet."""
    return TF.normalize(TF.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)


def preprocess(path: Path, short_side: int | None = None) -> torch.Tensor:
    """Pipeline completo, da foto em disco ao tensor de entrada do modelo."""
    with Image.open(path) as image:
        enhanced = enhance(image)
    return to_tensor(resize_short_side(enhanced, short_side))
