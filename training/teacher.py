"""Professor da rotulagem do leitor: Claude Haiku 5.5 pela API (T6a.3; D1, D5, D6).

Monta o recorte enviado (E-a: realce fixo do pipeline na foto, recorte com a margem
da classe, ampliação até o lado maior ter 512 px), o pedido por classe (esquema da
T6a.1, numa ferramenta de resposta obrigatória) e valida a resposta com regras
estritas. Nenhum nome de arquivo nem valor do BaseExtracao entra no pedido.
"""

from __future__ import annotations

import base64
import io
import re

from PIL import Image

from meter_reader.crops import ROTATIONS, crop_field
from meter_reader.preprocess import enhance

MODEL = "claude-haiku-5-5"
MAX_TOKENS = 300
MIN_LONG_SIDE = 512   # E-a
TOOL_NAME = "registrar_leitura"

_READING_RE = re.compile(r"^(?:[0-9?]|\[[0-9]{2}\])+$")  # dígitos, ? e transições como [56]
_FUNCTION_RE = re.compile(r"^[0-9?]+$")
_SERIAL_RE = re.compile(r"^[A-Z0-9?]+$")  # o número físico não tem "/" (o "/" é do CSV)

TEXT_FIELDS = {"display_digital": ("leitura", "funcao"),
               "display_ciclometrico": ("leitura",),
               "numero_serie": ("numero",)}
BOOL_FIELDS = {"display_digital": ("duas_leituras",),
               "display_ciclometrico": (),
               "numero_serie": ()}

SYSTEM = (
    "Você recebe o recorte de uma foto de um medidor de energia elétrica e responde só "
    "pela ferramenta registrar_leitura. Regras: (1) transcreva exatamente o que aparece, "
    "da esquerda para a direita, com os zeros à esquerda; (2) se um caractere não puder "
    "ser lido com segurança, escreva ? na posição dele, e nunca complete nem adivinhe; "
    "(3) se nada puder ser lido, legivel = false e os textos vazios; (4) rotacao é o giro, "
    "em graus no sentido horário (0, 90, 180 ou 270), que deixaria o texto em pé."
)

CLASS_PROMPTS = {
    "display_digital": (
        "O recorte mostra o display de cristal líquido de um medidor digital. Alguns displays "
        "mostram uma só leitura; outros mostram duas, uma acima da outra. leitura: todos os "
        "dígitos da leitura de cima (ou da única), sem pontos, vírgulas, unidades nem ícones. "
        "duas_leituras: true se o display mostrar duas leituras, uma acima da outra. funcao: o "
        "código de função, que fica no canto superior esquerdo do display, completo, com todos "
        "os seus dígitos (por exemplo 03, 103 ou 55); vazio se o display não mostrar código de "
        "função."
    ),
    "display_ciclometrico": (
        "O recorte mostra a janela de tambores numerados de um medidor ciclométrico. leitura: "
        "um dígito por tambor. Se um tambor estiver entre dois números, escreva os dois entre "
        "colchetes, primeiro o que está saindo e depois o que está entrando, por exemplo [56] "
        "ou [90]."
    ),
    "numero_serie": (
        "O recorte mostra o número de série do medidor, impresso junto ao código de barras ou "
        "numa plaqueta. numero: só letras maiúsculas e dígitos, sem espaços, hífens nem barras. "
        "Ignore o código de barras e os demais números da placa."
    ),
}


def prepare_crop(image: Image.Image, box, field_class: str) -> bytes:
    """PNG do recorte enviado ao professor (E-a)."""
    crop = crop_field(enhance(image), box, field_class)
    long_side = max(crop.size)
    if long_side < MIN_LONG_SIDE:
        scale = MIN_LONG_SIDE / long_side
        size = (max(1, round(crop.width * scale)), max(1, round(crop.height * scale)))
        crop = crop.resize(size, Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    crop.save(buf, format="PNG")
    return buf.getvalue()


def tool_schema(field_class: str) -> dict:
    """Ferramenta de resposta obrigatória da classe: campos exatos, nada além deles."""
    props = {"legivel": {"type": "boolean"}}
    for name in TEXT_FIELDS[field_class]:
        props[name] = {"type": "string"}
    for name in BOOL_FIELDS[field_class]:
        props[name] = {"type": "boolean"}
    props["rotacao"] = {"type": "integer", "enum": list(ROTATIONS)}
    return {"name": TOOL_NAME, "description": "Registra a transcrição do recorte.",
            "input_schema": {"type": "object", "properties": props,
                             "required": list(props), "additionalProperties": False}}


def build_request(png: bytes, field_class: str) -> dict:
    """Argumentos de messages.create: só a imagem e o texto fixo da classe."""
    image = {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                         "data": base64.b64encode(png).decode("ascii")}}
    return {"model": MODEL, "max_tokens": MAX_TOKENS, "temperature": 0, "system": SYSTEM,
            "tools": [tool_schema(field_class)],
            "tool_choice": {"type": "tool", "name": TOOL_NAME},
            "messages": [{"role": "user", "content": [
                image, {"type": "text", "text": CLASS_PROMPTS[field_class]}]}]}


def validate(field_class: str, data: dict) -> dict:
    """Confere a resposta com regras estritas; devolve a versão normalizada ou ValueError."""
    expected = {"legivel", "rotacao", *TEXT_FIELDS[field_class], *BOOL_FIELDS[field_class]}
    if set(data) != expected:
        raise ValueError("Campos da resposta diferentes do esquema.")
    if not isinstance(data["legivel"], bool) or data["rotacao"] not in ROTATIONS:
        raise ValueError("legivel ou rotacao fora do esquema.")
    if any(not isinstance(data[n], bool) for n in BOOL_FIELDS[field_class]):
        raise ValueError("Campo verdadeiro ou falso fora do esquema.")
    out = {"legivel": data["legivel"], "rotacao": int(data["rotacao"])}
    out.update({n: data[n] for n in BOOL_FIELDS[field_class]})
    for name in TEXT_FIELDS[field_class]:
        text = str(data[name]).strip()
        if field_class == "numero_serie":
            text = text.upper()
        out[name] = text
    texts = [out[n] for n in TEXT_FIELDS[field_class]]
    if not out["legivel"]:
        if any(texts):
            raise ValueError("Ilegível com texto preenchido.")
        return out
    main = out["numero"] if field_class == "numero_serie" else out["leitura"]
    pattern = _SERIAL_RE if field_class == "numero_serie" else _READING_RE
    if not pattern.fullmatch(main):
        raise ValueError("Texto principal fora do formato.")
    if field_class == "display_digital" and out["funcao"] and not _FUNCTION_RE.fullmatch(out["funcao"]):
        raise ValueError("Função fora do formato.")
    return out
