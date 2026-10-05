"""Verificação de integridade das imagens (spec 2.5, item 2.3).

Uma imagem é rejeitada como corrompida se não abrir ou não decodificar,
ou se estiver vazia (totalmente preta ou branca). O critério de vazia é
conservador (opção B da T1.5): exige imagem quase uniforme E brilho
extremo, para que nenhuma foto de medidor vá para "Corrompidas" (R1, E09).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from meter_reader.contracts import IntegrityIssue, IntegrityResult

# Lado máximo, em pixels, da versão reduzida usada na medida
_ANALYSIS_SIZE = (256, 256)


@dataclass(frozen=True)
class ImageStats:
    """Medidas de uma imagem em tons de cinza (escala de 0 a 255)."""

    mean: float  # brilho médio
    std: float  # contraste: desvio padrão das intensidades


@dataclass(frozen=True)
class EmptyImageThresholds:
    """Limiares do critério de imagem vazia (opção B da T1.5)."""

    max_std: float  # até este contraste, a imagem é quase uniforme
    max_dark_mean: float  # até este brilho, é preta
    min_bright_mean: float  # a partir deste brilho, é branca


# Calibrados na base real (4 lotes, 12.340 imagens), com inspeção visual das
# 69 imagens de contraste <= 10 e brilho <= 15. A imagem com conteúdo de
# menor contraste mediu 1,55; com contraste <= 1,0 foram descartadas 26 das
# 35 vazias e nenhuma imagem com conteúdo. As vazias restantes seguem para a
# triagem. A base não tem imagem branca: esse limiar só é coberto por testes
# sintéticos. Não aumentar max_std sem nova calibração (restrição R1).
DEFAULT_THRESHOLDS = EmptyImageThresholds(
    max_std=1.0, max_dark_mean=10.0, min_bright_mean=245.0
)


def measure(path: Path | str) -> ImageStats:
    """Abre, decodifica e mede a imagem. Erros de leitura são propagados."""
    with Image.open(path) as image:
        # JPEG: decodifica já reduzida, sem passar pela resolução total.
        # Para outros formatos, draft não faz nada.
        image.draft("L", _ANALYSIS_SIZE)
        gray = image.convert("L")  # convert força a decodificação completa
    pixels = np.asarray(gray, dtype=np.float32)
    return ImageStats(mean=float(pixels.mean()), std=float(pixels.std()))


def classify(stats: ImageStats, thresholds: EmptyImageThresholds) -> IntegrityIssue | None:
    """Aplica o critério de vazia: quase uniforme E brilho extremo."""
    if stats.std > thresholds.max_std:
        return None  # há detalhe na imagem: nunca é vazia
    if stats.mean <= thresholds.max_dark_mean:
        return IntegrityIssue.EMPTY_DARK
    if stats.mean >= thresholds.min_bright_mean:
        return IntegrityIssue.EMPTY_BRIGHT
    return None  # uniforme, mas de brilho intermediário: segue no pipeline


def check_integrity(
    path: Path | str, thresholds: EmptyImageThresholds = DEFAULT_THRESHOLDS
) -> IntegrityResult:
    """Verifica uma imagem e devolve o resultado com a causa técnica, se houver."""
    photo_name = Path(path).name
    try:
        stats = measure(path)
    except (OSError, ValueError, SyntaxError):
        # OSError cobre arquivo vazio, formato não reconhecido e JPEG truncado;
        # ValueError e SyntaxError aparecem em alguns arquivos malformados
        return IntegrityResult(photo_name, IntegrityIssue.UNREADABLE)
    return IntegrityResult(photo_name, classify(stats, thresholds))
