import pytest
from PIL import Image

from meter_reader.crops import CROP_MARGINS, crop_box, crop_field


def test_margin_is_applied_per_axis():
    # Caixa 100 x 20 no meio de uma foto grande: margem digital (0,22; 0,27)
    assert crop_box((200, 200, 300, 220), "display_digital", 1000, 1000) == (178, 194, 322, 226)


def test_crop_is_limited_to_the_photo():
    assert crop_box((0, 0, 100, 20), "numero_serie", 110, 22) == (0, 0, 110, 22)


def test_invalid_inputs_are_refused():
    with pytest.raises(ValueError):
        crop_box((0, 0, 10, 10), "display_kvarh", 100, 100)  # kvarh não é lido
    with pytest.raises(ValueError):
        crop_box((5, 5, 5, 10), "display_digital", 100, 100)
    with pytest.raises(ValueError):
        crop_box((200, 200, 300, 220), "display_digital", 100, 100)
    with pytest.raises(ValueError):
        crop_field(Image.new("RGB", (100, 100)), (10, 10, 50, 20), "display_digital", rotation=45)


def test_rotation_is_clockwise():
    # Foto 4 x 2 com o pixel vermelho no canto superior esquerdo; caixa = foto inteira
    image = Image.new("RGB", (4, 2), "black")
    image.putpixel((0, 0), (255, 0, 0))
    out = crop_field(image, (0, 0, 4, 2), "display_digital", rotation=90)
    assert out.size == (2, 4)
    # Girando 90 graus no sentido horário, o canto superior esquerdo vai para o superior direito
    assert out.getpixel((1, 0)) == (255, 0, 0)


def test_only_reader_classes_have_margin():
    assert set(CROP_MARGINS) == {"display_digital", "display_ciclometrico", "numero_serie"}
