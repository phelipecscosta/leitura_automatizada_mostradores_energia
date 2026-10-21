"""Características DINOv2 para a triagem de ML (protocolo de ML, seções 1.2 a 1.4).

Pré-processamento igual ao da triagem de DL (spec 2.5.7): realce fixo, 224 px
no lado menor e normalização do ImageNet; cada lado é depois cortado ao
centro para o maior múltiplo de 14, o tamanho do patch do ViT-S/14.

As vistas aumentadas são transformações leves aplicadas sobre a foto já
realçada (o autocontraste desfaria a variação de brilho e contraste se viesse
depois). Cada vista tem semente própria, derivada do pseudônimo, para que a
mesma foto gere sempre as mesmas vistas.
"""

from __future__ import annotations

import zlib

import torch
from PIL import Image, ImageStat
from torchvision.transforms import functional as TF

from meter_reader.preprocess import enhance, resize_short_side, to_tensor

SHORT_SIDE = 224
PATCH = 14
# Aumentação leve (protocolo de ML, seção 1.3)
CROP_AREA = (0.80, 1.00)   # fração da área mantida no recorte
MAX_ROTATION = 10.0        # graus, nos dois sentidos
MAX_JITTER = 0.20          # variação máxima de brilho e de contraste


def crop_to_multiple(image: Image.Image, multiple: int = PATCH) -> Image.Image:
    """Corta ao centro cada lado para o maior múltiplo de `multiple`."""
    width, height = image.size
    new_w, new_h = width - width % multiple, height - height % multiple
    left, top = (width - new_w) // 2, (height - new_h) // 2
    return image.crop((left, top, left + new_w, top + new_h))


def model_input(enhanced: Image.Image) -> torch.Tensor:
    """Foto já realçada -> tensor (3, A, L) de entrada do DINOv2."""
    return to_tensor(crop_to_multiple(resize_short_side(enhanced, SHORT_SIDE)))


def view_seed(pseudonym: str, view: int) -> int:
    """Semente estável por foto e vista (o hash do Python muda a cada sessão)."""
    return zlib.crc32(f"{pseudonym}:{view}".encode("utf-8"))


def _uniform(generator: torch.Generator, low: float, high: float) -> float:
    return low + (high - low) * torch.rand(1, generator=generator).item()


def augment(enhanced: Image.Image, seed: int) -> Image.Image:
    """Uma vista aumentada: recorte na mesma proporção, rotação, brilho e contraste."""
    generator = torch.Generator().manual_seed(seed)
    width, height = enhanced.size

    # Recorte com 80% a 100% da área, na proporção da foto, em posição sorteada
    scale = _uniform(generator, *CROP_AREA) ** 0.5
    crop_w, crop_h = round(width * scale), round(height * scale)
    left = round(_uniform(generator, 0, width - crop_w))
    top = round(_uniform(generator, 0, height - crop_h))
    view = enhanced.crop((left, top, left + crop_w, top + crop_h))

    # Rotação; os cantos recebem a cor média da foto, e não preto
    fill = [round(c) for c in ImageStat.Stat(view).mean]
    angle = _uniform(generator, -MAX_ROTATION, MAX_ROTATION)
    view = TF.rotate(view, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=fill)

    view = TF.adjust_brightness(view, _uniform(generator, 1 - MAX_JITTER, 1 + MAX_JITTER))
    return TF.adjust_contrast(view, _uniform(generator, 1 - MAX_JITTER, 1 + MAX_JITTER))


def photo_inputs(image: Image.Image, pseudonym: str, n_views: int) -> torch.Tensor:
    """Tensor (1 + n_views, 3, A, L): a foto original seguida das vistas."""
    enhanced = enhance(image)
    views = [enhanced] + [augment(enhanced, view_seed(pseudonym, v)) for v in range(1, n_views + 1)]
    return torch.stack([model_input(v) for v in views])
