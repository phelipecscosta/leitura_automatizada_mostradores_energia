from PIL import Image

from meter_reader.reader_input import (READER_HEIGHT, READER_WIDTH, fit_canvas, prepare,
                                       reader_tensor, rotate_180, upper_band)


def _crop(width, height, color=(200, 30, 30)):
    image = Image.new("RGB", (width, height), color)
    image.putpixel((0, 0), (0, 0, 0))  # contraste mínimo para o realce
    return image


def test_tensor_tem_forma_fixa():
    assert reader_tensor(_crop(150, 50)).shape == (3, READER_HEIGHT, READER_WIDTH)


def test_tamanho_fixo_sem_manter_a_proporcao():
    for size in ((100, 50), (800, 50), (60, 90)):
        assert fit_canvas(_crop(*size)).size == (READER_WIDTH, READER_HEIGHT)


def test_cinza_repete_o_canal():
    image = prepare(_crop(100, 40), gray=True)
    r, g, b = (band.tobytes() for band in image.split())
    assert r == g == b


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
