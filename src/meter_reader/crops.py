"""Recorte dos campos para o leitor (spec v18, 2.5, item 11; T6a.2).

A caixa escolhida pelo detector cresce por uma margem fixa por classe e por eixo
(p99 ponderado da expansão necessária no desenvolvimento, M-b), limitada à borda
da foto. A rotação em múltiplos de 90 graus deixa o texto em pé (O-a).
A foto deve ser aberta como no detector (sem aplicar a orientação EXIF), para que
as coordenadas das caixas valham aqui.
"""

from __future__ import annotations

import math

from PIL import Image

# Margem por lado, em fração da largura e da altura da caixa: (horizontal, vertical).
# p99 ponderado do S1_coco fora da dobra, semente 0, arredondado para cima (T6a.2).
CROP_MARGINS = {
    "display_digital": (0.22, 0.27),
    "display_ciclometrico": (0.23, 0.27),
    "numero_serie": (0.20, 0.32),
}

# Giro no sentido horário que deixa o texto em pé (O-a)
ROTATIONS = (0, 90, 180, 270)
_TRANSPOSE = {90: Image.Transpose.ROTATE_270,   # o PIL gira no sentido anti-horário
              180: Image.Transpose.ROTATE_180,
              270: Image.Transpose.ROTATE_90}


def crop_box(box, field_class: str, width: int, height: int) -> tuple[int, int, int, int]:
    """Caixa do recorte em pixels inteiros: a caixa com a margem da classe,
    limitada à foto de tamanho (width, height)."""
    if field_class not in CROP_MARGINS:
        raise ValueError("Classe de campo sem margem definida.")
    mh, mv = CROP_MARGINS[field_class]
    x0, y0, x1, y1 = (float(v) for v in box)
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        raise ValueError("Caixa sem área.")
    left = max(0, math.floor(x0 - mh * w))
    top = max(0, math.floor(y0 - mv * h))
    right = min(width, math.ceil(x1 + mh * w))
    bottom = min(height, math.ceil(y1 + mv * h))
    if right <= left or bottom <= top:
        raise ValueError("Caixa fora da foto.")
    return left, top, right, bottom


def crop_field(image: Image.Image, box, field_class: str, rotation: int = 0) -> Image.Image:
    """Recorte do campo, girado `rotation` graus no sentido horário."""
    if rotation not in ROTATIONS:
        raise ValueError("A rotação deve ser 0, 90, 180 ou 270.")
    out = image.crop(crop_box(box, field_class, *image.size))
    return out if rotation == 0 else out.transpose(_TRANSPOSE[rotation])
