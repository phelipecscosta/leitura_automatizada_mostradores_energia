import torch
from PIL import Image

from meter_reader.reader_input import (READER_HEIGHT, READER_MAX_WIDTH, fit_canvas, prepare,
                                       reader_tensor, rotate_180, upper_band)


def _crop(width, height, color=(200, 30, 30)):
    image = Image.new("RGB", (width, height), color)
    image.putpixel((0, 0), (0, 0, 0))  # contraste mínimo para o realce
    return image


def test_tensor_tem_forma_fixa():
    assert reader_tensor(_crop(150, 50)).shape == (3, READER_HEIGHT, READER_MAX_WIDTH)


def test_proporcao_mantida_e_preenchimento_zero():
    tensor = reader_tensor(_crop(100, 50))   # vira 96 x 48
    assert tensor[:, :, 100:].abs().max() < 0.02   # preenchimento ~ média do ImageNet
    assert tensor[:, :, :96].abs().max() > 0.5


def test_recorte_largo_reduz_proporcionalmente():
    canvas = fit_canvas(_crop(800, 50))      # escala 384/800: 384 x 24
    assert canvas.size == (READER_MAX_WIDTH, READER_HEIGHT)
    # a linha 30 já é preenchimento
    assert canvas.getpixel((10, 30)) == canvas.getpixel((383, 47))


def test_cinza_repete_o_canal():
    image = prepare(_crop(100, 40), gray=True)   # vira 120 x 48
    r, g, b = (band.crop((0, 0, 120, 48)).tobytes() for band in image.split())
    assert r == g == b    # o preenchimento fica fora: ele vira zero na normalização


def test_faixa_de_cima_e_rotacao():
    crop = _crop(100, 40)
    assert upper_band(crop).size == (100, 22)
    assert rotate_180(crop).getpixel((99, 39)) == (0, 0, 0)


def test_recorte_sem_area():
    try:
        fit_canvas(Image.new("RGB", (0, 10)))
    except ValueError:
        return
    raise AssertionError("Esperava ValueError.")
