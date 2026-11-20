"""Aumentação do leitor (protocolo do leitor, seção 1.12).

Aplicada ao recorte em resolução original, antes do realce e da tela fixa
(meter_reader.reader_input). Sem espelhamento e sem giro de 180 graus (O-b).
"""

from __future__ import annotations

import io
import random

from PIL import Image, ImageFilter, ImageOps
from torchvision.transforms import functional as TF

MAX_ROTATION = 5.0        # graus
MAX_PERSPECTIVE = 0.03    # deslocamento dos cantos, em fração do lado
MAX_SHIFT = 0.05          # deslocamento e escala do recorte (variação das caixas)
GRAY_PROBABILITY = 0.20
BLUR_PROBABILITY = 0.30
JPEG_PROBABILITY = 0.30


def _mean_color(image: Image.Image) -> tuple[int, int, int]:
    small = image.resize((1, 1), Image.Resampling.BOX)
    return small.getpixel((0, 0))


def _photometric(image: Image.Image, rng: random.Random) -> Image.Image:
    image = TF.adjust_brightness(image, rng.uniform(0.7, 1.3))
    image = TF.adjust_contrast(image, rng.uniform(0.7, 1.3))
    image = TF.adjust_gamma(image, rng.uniform(0.8, 1.25))
    image = TF.adjust_saturation(image, rng.uniform(0.6, 1.4))
    image = TF.adjust_hue(image, rng.uniform(-0.05, 0.05))
    if rng.random() < GRAY_PROBABILITY:
        image = ImageOps.grayscale(image).convert("RGB")
    return image


def _geometric(image: Image.Image, rng: random.Random) -> Image.Image:
    width, height = image.size
    fill = list(_mean_color(image))
    image = TF.affine(image, angle=rng.uniform(-MAX_ROTATION, MAX_ROTATION),
                      translate=[round(rng.uniform(-MAX_SHIFT, MAX_SHIFT) * width),
                                 round(rng.uniform(-MAX_SHIFT, MAX_SHIFT) * height)],
                      scale=rng.uniform(1 - MAX_SHIFT, 1 + MAX_SHIFT), shear=[0.0, 0.0],
                      interpolation=TF.InterpolationMode.BILINEAR, fill=fill)
    corners = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
    moved = [[x + rng.uniform(-MAX_PERSPECTIVE, MAX_PERSPECTIVE) * width,
              y + rng.uniform(-MAX_PERSPECTIVE, MAX_PERSPECTIVE) * height] for x, y in corners]
    return TF.perspective(image, corners, moved, interpolation=TF.InterpolationMode.BILINEAR,
                          fill=fill)


def _degrade(image: Image.Image, rng: random.Random) -> Image.Image:
    if rng.random() < BLUR_PROBABILITY:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.2)))
    if rng.random() < JPEG_PROBABILITY:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=rng.randint(30, 90))
        buffer.seek(0)
        image = Image.open(buffer).convert("RGB")
    return image


def augment(image: Image.Image, rng: random.Random) -> Image.Image:
    """Recorte RGB -> recorte RGB aumentado, do mesmo tamanho."""
    image = image.convert("RGB")
    return _degrade(_geometric(_photometric(image, rng), rng), rng)
