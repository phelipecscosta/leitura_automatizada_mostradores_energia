"""Assinatura visual (dHash) para encontrar quase-duplicatas.

Usada para evitar vazamento entre os conjuntos de treino e teste: cópias
da mesma foto, recomprimidas ou redimensionadas, têm assinaturas próximas.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

HASH_SIZE = 8  # 8 x 8 = 64 bits


def dhash(path: Path | str) -> str:
    """Retorna a assinatura de 64 bits como texto hexadecimal (16 caracteres)."""
    with Image.open(path) as image:
        image.draft("L", (64, 64))  # JPEG: decodifica já reduzida
        gray = image.convert("L").resize(
            (HASH_SIZE + 1, HASH_SIZE), Image.Resampling.LANCZOS
        )
    pixels = np.asarray(gray, dtype=np.int16)
    # 1 quando o pixel da direita é mais claro que o da esquerda
    bits = pixels[:, 1:] > pixels[:, :-1]
    return np.packbits(bits.flatten()).tobytes().hex()


def hamming(hash_a: str, hash_b: str) -> int:
    """Quantidade de bits diferentes entre duas assinaturas (0 a 64)."""
    return (int(hash_a, 16) ^ int(hash_b, 16)).bit_count()
