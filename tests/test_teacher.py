import base64
import io

import pytest
from PIL import Image

from training.teacher import (MIN_LONG_SIDE, TOOL_NAME, build_request, prepare_crop,
                              tool_schema, validate)


def test_prepare_crop_upscales_small_fields():
    image = Image.new("RGB", (640, 480), "gray")
    png = prepare_crop(image, (100, 100, 140, 115), "display_digital")
    crop = Image.open(io.BytesIO(png))
    assert max(crop.size) == MIN_LONG_SIDE


def test_schema_has_exactly_the_class_fields():
    props = tool_schema("display_digital")["input_schema"]["properties"]
    assert set(props) == {"legivel", "leitura", "funcao", "duas_leituras", "rotacao"}
    assert set(tool_schema("numero_serie")["input_schema"]["properties"]) == {
        "legivel", "numero", "rotacao"}


def test_request_carries_only_image_and_fixed_text():
    req = build_request(b"png", "display_ciclometrico")
    content = req["messages"][0]["content"]
    assert [c["type"] for c in content] == ["image", "text"]
    assert base64.b64decode(content[0]["source"]["data"]) == b"png"
    assert req["tool_choice"] == {"type": "tool", "name": TOOL_NAME}


_DIG = {"legivel": True, "leitura": "0012345", "funcao": "103", "duas_leituras": False,
        "rotacao": 0}


@pytest.mark.parametrize("cls,data", [
    ("display_digital", _DIG),
    ("display_digital", {**_DIG, "funcao": "", "duas_leituras": True}),
    ("display_ciclometrico", {"legivel": True, "leitura": "0124[56]", "rotacao": 90}),
    ("display_ciclometrico", {"legivel": True, "leitura": "01?45", "rotacao": 0}),
    ("numero_serie", {"legivel": True, "numero": "ab12319", "rotacao": 180}),
    ("display_digital", {"legivel": False, "leitura": "", "funcao": "", "duas_leituras": False,
                         "rotacao": 0}),
])
def test_valid_answers(cls, data):
    out = validate(cls, data)
    assert out["legivel"] == data["legivel"]
    if cls == "numero_serie":
        assert out["numero"] == "AB12319"


@pytest.mark.parametrize("cls,data", [
    ("display_digital", {k: v for k, v in _DIG.items() if k != "funcao"}),        # falta funcao
    ("display_digital", {**_DIG, "duas_leituras": "sim"}),                        # não booleano
    ("display_digital", {**_DIG, "funcao": "0A"}),
    ("display_ciclometrico", {"legivel": True, "leitura": "12,5", "rotacao": 0}),  # vírgula
    ("display_ciclometrico", {"legivel": True, "leitura": "12[5]", "rotacao": 0}), # transição incompleta
    ("display_ciclometrico", {"legivel": True, "leitura": "125", "rotacao": 45}),  # rotação inválida
    ("display_ciclometrico", {"legivel": False, "leitura": "125", "rotacao": 0}),  # ilegível com texto
    ("numero_serie", {"legivel": True, "numero": "AB-12", "rotacao": 0}),          # hífen
    ("numero_serie", {"legivel": True, "numero": "123/19", "rotacao": 0}),         # barra
])
def test_invalid_answers(cls, data):
    with pytest.raises(ValueError):
        validate(cls, data)
