"""Aumentação única da triagem de cinco classes (protocolo, 1.3 e revisão R2).

Aplicada sobre a foto já realçada: rotação de ±10°, translação de até ±5% em
cada eixo (numa só transformação afim), brilho e contraste de ±0,2. Sem
recorte e sem espelhamento: nenhuma vista pode mudar a classe verdadeira
(E57). Os cantos expostos recebem a cor média da foto, e não preto.

Usada para as vistas guardadas do C0 e do C1 (semente por pseudônimo e
vista) e na aumentação durante o treino do C2, do C3 e do C4.
"""

from __future__ import annotations

import torch
from PIL import Image, ImageStat
from torchvision.transforms import functional as TF

from meter_reader.preprocess import enhance
from training.dino_features import view_seed

MAX_ROTATION = 10.0   # graus, nos dois sentidos
MAX_SHIFT = 0.05      # fração da largura ou da altura
MAX_JITTER = 0.20     # variação máxima de brilho e de contraste


def _uniform(generator: torch.Generator, low: float, high: float) -> float:
    return low + (high - low) * torch.rand(1, generator=generator).item()


def augment(enhanced: Image.Image, seed: int) -> Image.Image:
    """Uma vista aumentada, determinada pela semente; mesmo tamanho da entrada."""
    generator = torch.Generator().manual_seed(seed)
    width, height = enhanced.size
    angle = _uniform(generator, -MAX_ROTATION, MAX_ROTATION)
    shift = [round(_uniform(generator, -MAX_SHIFT, MAX_SHIFT) * width),
             round(_uniform(generator, -MAX_SHIFT, MAX_SHIFT) * height)]
    fill = [round(c) for c in ImageStat.Stat(enhanced).mean]
    view = TF.affine(enhanced, angle=angle, translate=shift, scale=1.0, shear=[0.0, 0.0],
                     interpolation=TF.InterpolationMode.BILINEAR, fill=fill)
    view = TF.adjust_brightness(view, _uniform(generator, 1 - MAX_JITTER, 1 + MAX_JITTER))
    return TF.adjust_contrast(view, _uniform(generator, 1 - MAX_JITTER, 1 + MAX_JITTER))


def photo_views(image: Image.Image, pseudonym: str, n_views: int) -> list[Image.Image]:
    """Foto realçada seguida de `n_views` vistas, com semente por pseudônimo e vista."""
    enhanced = enhance(image)
    return [enhanced] + [augment(enhanced, view_seed(pseudonym, v)) for v in range(1, n_views + 1)]


