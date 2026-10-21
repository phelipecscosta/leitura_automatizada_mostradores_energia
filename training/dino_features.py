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
from pathlib import Path

import numpy as np
import timm
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



# ---------------------------------------------------------------------------
# Modelo, extração e cache
# ---------------------------------------------------------------------------

MODEL_NAME = "vit_small_patch14_dinov2.lvd142m"  # pesos oficiais (Apache-2.0)
FEATURE_DIM = 384
# Versão do cache: muda quando o cálculo muda, para que um cache antigo nunca
# seja lido. "fp32": precisão cheia também na GPU, como na E33.
DINO_CACHE_VERSION = "fp32"


def load_dinov2(pretrained: bool = True) -> torch.nn.Module:
    """DINOv2 ViT-S/14 congelado; a saída é o token de classe normalizado (384)."""
    model = timm.create_model(MODEL_NAME, pretrained=pretrained, num_classes=0,
                              dynamic_img_size=True)  # aceita grades não quadradas
    model.requires_grad_(False)
    return model.eval()


def extract_views(paths: list[Path], codes: list[str], n_views: int,
                  model: torch.nn.Module, device: str = "cpu") -> np.ndarray:
    """Matriz (n_fotos, 1 + n_views, 384); a posição 0 é a foto original."""
    if len(paths) != len(codes):
        raise ValueError("Caminhos e pseudônimos com tamanhos diferentes.")
    model = model.to(device).eval()
    chunks = []
    # Sem TF32 nas convoluções (patch) nem nas multiplicações de matrizes
    # (atenção): as características do treino devem ser iguais às da CPU
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        with torch.no_grad(), torch.backends.cudnn.flags(enabled=True, benchmark=False,
                                                         deterministic=False, allow_tf32=False):
            for path, code in zip(paths, codes):
                with Image.open(path) as image:
                    x = photo_inputs(image, code, n_views)
                chunks.append(model(x.to(device)).cpu().numpy())
    finally:
        torch.set_float32_matmul_precision(previous)
    return np.stack(chunks).astype(np.float32)


def cache_path(cache_dir: Path, name: str, n_views: int) -> Path:
    return cache_dir / f"caracteristicas_{name}_vistas{n_views}_{DINO_CACHE_VERSION}.npz"


def load_or_extract_views(paths: list[Path], codes: list[str], name: str, n_views: int,
                          cache_dir: Path, device: str = "cpu",
                          pretrained: bool = True) -> np.ndarray:
    """Lê do cache se a lista de pseudônimos for a mesma; senão, extrai e grava."""
    path = cache_path(cache_dir, name, n_views)
    codes_array = np.array(codes)
    if path.is_file():
        cached = np.load(path)
        if np.array_equal(cached["codes"], codes_array):
            return cached["features"]
    features = extract_views(paths, codes, n_views, load_dinov2(pretrained), device)
    np.savez(path, codes=codes_array, features=features)
    return features
