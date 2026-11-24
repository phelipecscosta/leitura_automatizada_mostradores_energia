"""Entrada do leitor: pré-processamento II (spec v18, 2.5, item 11; protocolo do
leitor, seções 1.3 e 1.4).

Ordem: recorte do campo (meter_reader.crops) -> realce fixo (autocontraste de 1%)
-> opcionalmente cinza, repetido nos 3 canais (ablação PP-a) -> redimensiona para
48 x 192, sem manter a proporção, como no CRNN de Shi et al. (revisão R4) -> tensor
normalizado pelo ImageNet. A mesma função serve ao produto e ao treino (E33).
"""

from __future__ import annotations

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from meter_reader.preprocess import IMAGENET_MEAN, IMAGENET_STD, enhance

READER_HEIGHT = 48
READER_WIDTH = 192          # revisão R4: 48 passos de tempo com o passo horizontal 4
UPPER_BAND = 0.55           # duas leituras: faixa de 0 a 55% da altura (D-b, seção 1.3)


def upper_band(crop: Image.Image) -> Image.Image:
    """Faixa de cima do recorte, para reler os displays de duas leituras (D-b)."""
    width, height = crop.size
    return crop.crop((0, 0, width, max(1, round(height * UPPER_BAND))))


def fit_canvas(crop: Image.Image) -> Image.Image:
    """Redimensiona para 48 x 192, sem manter a proporção (revisão R4).

    Com a tela de 48 x 384 e preenchimento, o texto ocupava de 1/3 a 2/3 da largura e
    a CTC tinha 96 passos para até 11 símbolos: o treino não saía do platô.
    """
    width, height = crop.size
    if width <= 0 or height <= 0:
        raise ValueError("Recorte sem área.")
    return crop.resize((READER_WIDTH, READER_HEIGHT), Image.Resampling.BILINEAR)


def prepare(crop: Image.Image, gray: bool = False) -> Image.Image:
    """Realce, cinza opcional e tamanho fixo; devolve a imagem RGB de 48 x 192."""
    image = enhance(crop)
    if gray:
        image = image.convert("L").convert("RGB")
    return fit_canvas(image)


def to_tensor(image: Image.Image) -> torch.Tensor:
    """Imagem RGB de 48 x 192 -> tensor (3, 48, 192) normalizado pelo ImageNet."""
    return TF.normalize(TF.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)


def reader_tensor(crop: Image.Image, gray: bool = False) -> torch.Tensor:
    """Do recorte em resolução original ao tensor de entrada do leitor."""
    return to_tensor(prepare(crop, gray))
