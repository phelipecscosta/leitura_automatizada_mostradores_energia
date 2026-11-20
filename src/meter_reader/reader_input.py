"""Entrada do leitor: pré-processamento II (spec v18, 2.5, item 11; protocolo do
leitor, seções 1.3 e 1.4).

Ordem: recorte do campo (meter_reader.crops) -> realce fixo (autocontraste de 1%)
-> opcionalmente cinza, repetido nos 3 canais (ablação PP-a) -> redimensiona para
48 px de altura, mantendo a proporção, com redução proporcional se passar de 384
px de largura -> cola no canto superior esquerdo de uma tela fixa de 48 x 384
(preenchimento à direita e, se preciso, embaixo) -> tensor normalizado pelo
ImageNet. A mesma função serve ao produto e ao treino (E33).
"""

from __future__ import annotations

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from meter_reader.preprocess import IMAGENET_MEAN, IMAGENET_STD, enhance

READER_HEIGHT = 48
READER_MAX_WIDTH = 384      # 8 x 48 (seção 1.4)
UPPER_BAND = 0.55           # duas leituras: faixa de 0 a 55% da altura (D-b, seção 1.3)
# Cor do preenchimento: a média do ImageNet, que vira zero depois da normalização
_PAD_COLOR = tuple(round(255 * m) for m in IMAGENET_MEAN)


def upper_band(crop: Image.Image) -> Image.Image:
    """Faixa de cima do recorte, para reler os displays de duas leituras (D-b)."""
    width, height = crop.size
    return crop.crop((0, 0, width, max(1, round(height * UPPER_BAND))))


def rotate_180(crop: Image.Image) -> Image.Image:
    """Recorte girado 180 graus (orientação O-b reduzida)."""
    return crop.transpose(Image.Transpose.ROTATE_180)


def fit_canvas(crop: Image.Image) -> Image.Image:
    """Redimensiona mantendo a proporção e cola numa tela de 48 x 384."""
    width, height = crop.size
    if width <= 0 or height <= 0:
        raise ValueError("Recorte sem área.")
    scale = min(READER_HEIGHT / height, READER_MAX_WIDTH / width)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    canvas = Image.new("RGB", (READER_MAX_WIDTH, READER_HEIGHT), _PAD_COLOR)
    canvas.paste(crop.resize(size, Image.Resampling.BILINEAR), (0, 0))
    return canvas


def prepare(crop: Image.Image, gray: bool = False) -> Image.Image:
    """Realce, cinza opcional e tela fixa; devolve a imagem RGB de 48 x 384."""
    image = enhance(crop)
    if gray:
        image = image.convert("L").convert("RGB")
    return fit_canvas(image)


def to_tensor(image: Image.Image) -> torch.Tensor:
    """Imagem RGB de 48 x 384 -> tensor (3, 48, 384) normalizado pelo ImageNet."""
    return TF.normalize(TF.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)


def reader_tensor(crop: Image.Image, gray: bool = False) -> torch.Tensor:
    """Do recorte em resolução original ao tensor de entrada do leitor."""
    return to_tensor(prepare(crop, gray))
