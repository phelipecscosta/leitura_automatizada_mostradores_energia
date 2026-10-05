"""Testes da verificação de integridade (imagens sintéticas geradas no teste)."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from meter_reader.contracts import IntegrityIssue
from meter_reader.integrity import (
    DEFAULT_THRESHOLDS,
    EmptyImageThresholds,
    ImageStats,
    check_integrity,
    classify,
)

RNG = np.random.default_rng(0)  # semente fixa: testes reprodutíveis


def _save(tmp_path: Path, pixels: np.ndarray, name: str = "foto.jpg") -> Path:
    path = tmp_path / name
    Image.fromarray(pixels.astype(np.uint8)).save(path, quality=90)
    return path


def _photo_like() -> np.ndarray:
    """Imagem com bastante variação, como uma foto comum."""
    return RNG.integers(0, 256, size=(480, 640, 3))


def test_normal_photo_is_intact(tmp_path: Path) -> None:
    assert check_integrity(_save(tmp_path, _photo_like())).issue is None


def test_black_image_is_empty_dark(tmp_path: Path) -> None:
    result = check_integrity(_save(tmp_path, np.zeros((480, 640, 3))))
    assert result.issue is IntegrityIssue.EMPTY_DARK


def test_white_image_is_empty_bright(tmp_path: Path) -> None:
    result = check_integrity(_save(tmp_path, np.full((480, 640, 3), 255)))
    assert result.issue is IntegrityIssue.EMPTY_BRIGHT


def test_black_with_sensor_noise_is_empty_dark(tmp_path: Path) -> None:
    # Ruído muito leve (contraste bem abaixo do limiar de 1,0) não faz uma
    # imagem preta deixar de ser vazia
    noisy = np.clip(RNG.normal(2, 0.3, size=(480, 640, 3)), 0, 255)
    assert check_integrity(_save(tmp_path, noisy)).issue is IntegrityIssue.EMPTY_DARK


def test_dark_photo_with_detail_is_kept(tmp_path: Path) -> None:
    # Caso de risco da R1: foto noturna, escura, mas com um visor claro
    night = np.full((480, 640, 3), 4)
    night[200:260, 250:400] = 200  # região clara simulando os dígitos
    assert check_integrity(_save(tmp_path, night)).issue is None


def test_uniform_mid_gray_is_kept(tmp_path: Path) -> None:
    # Opção B: uniforme, mas sem brilho extremo, segue no pipeline
    assert check_integrity(_save(tmp_path, np.full((480, 640, 3), 128))).issue is None


def test_empty_file_is_unreadable(tmp_path: Path) -> None:
    path = tmp_path / "vazio.jpg"
    path.write_bytes(b"")
    assert check_integrity(path).issue is IntegrityIssue.UNREADABLE


def test_non_image_file_is_unreadable(tmp_path: Path) -> None:
    path = tmp_path / "texto.jpg"
    path.write_text("isto não é uma imagem", encoding="utf-8")
    assert check_integrity(path).issue is IntegrityIssue.UNREADABLE


def test_truncated_jpeg_is_unreadable(tmp_path: Path) -> None:
    # Abre (o cabeçalho está íntegro), mas falha ao decodificar
    buffer = io.BytesIO()
    Image.fromarray(_photo_like().astype(np.uint8)).save(buffer, format="JPEG")
    path = tmp_path / "truncada.jpg"
    path.write_bytes(buffer.getvalue()[: len(buffer.getvalue()) // 2])
    assert check_integrity(path).issue is IntegrityIssue.UNREADABLE


def test_result_uses_only_file_name(tmp_path: Path) -> None:
    path = _save(tmp_path, _photo_like(), name="abc.jpg")
    assert check_integrity(path).photo_name == "abc.jpg"


# Limites do critério, testados direto em classify (sem arquivos)
T = EmptyImageThresholds(max_std=5.0, max_dark_mean=10.0, min_bright_mean=245.0)


@pytest.mark.parametrize(
    ("mean", "std", "expected"),
    [
        (10.0, 5.0, IntegrityIssue.EMPTY_DARK),  # no limite: ainda vazia
        (10.0, 5.1, None),  # contraste acima do limite
        (10.1, 5.0, None),  # brilho acima do limite de preta
        (245.0, 5.0, IntegrityIssue.EMPTY_BRIGHT),
        (244.9, 5.0, None),
    ],
)
def test_classify_boundaries(mean, std, expected) -> None:
    assert classify(ImageStats(mean=mean, std=std), T) is expected


def test_default_thresholds_are_consistent() -> None:
    assert DEFAULT_THRESHOLDS.max_dark_mean < DEFAULT_THRESHOLDS.min_bright_mean


def test_default_threshold_keeps_darkest_content_found() -> None:
    # Regressão da calibração da T1.5: a imagem com conteúdo de menor
    # contraste na base real mediu 1,55. O limiar padrão precisa ficar abaixo.
    assert DEFAULT_THRESHOLDS.max_std < 1.55
