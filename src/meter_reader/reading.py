"""Estágio de leitura do pipeline (spec 2.5, itens 11, 13 e 14; ET6c, T6c.6).

Liga o detector ao leitor e aos validadores: abre a foto como no cache de treino
(sem a orientação EXIF, em RGB), recorta o display de consumo e o número de série
escolhidos pelo detector, com a margem da classe, lê com o comitê e entrega tudo à
tabela de decisão (meter_reader.validation). Substitui o leitor provisório (E70).
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from meter_reader.contracts import ControlRow, ExtractionRow
from meter_reader.crops import crop_field
from meter_reader.detector import DetectionResult
from meter_reader.validation import decide, function_decision

DISPLAY_CLASS = {"digital": "display_digital", "ciclometrico": "display_ciclometrico"}
SERIAL = "numero_serie"


def _crop(image: Image.Image, box, field_class: str) -> Image.Image | None:
    """Recorte do campo; None se a caixa não tem área ou está fora da foto (campo
    tratado como não localizado)."""
    if box is None:
        return None
    try:
        return crop_field(image, box, field_class)
    except ValueError:
        return None


class ArtifactReader:
    """Leitor real: o comitê do artefato (meter_reader.reader_model) e os validadores."""

    def __init__(self, model, version: str):
        self.model = model      # ReaderModel ou equivalente (read, log_low, function_threshold)
        self.version = version  # vai para a aba Execução

    def read(self, path: Path, row: ExtractionRow, meter_type: str,
             detection: DetectionResult) -> ControlRow:
        path = Path(path)
        fields = detection.fields
        family = fields.family or meter_type
        display = serial = function = None
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        if fields.consumption is not None:
            cls = DISPLAY_CLASS[family]
            crop = _crop(image, fields.consumption, cls)
            if crop is not None:
                out = self.model.read([crop], cls)[0]
                display = out.field()
                if cls == DISPLAY_CLASS["digital"]:
                    function = function_decision(out.function, self.model.function_threshold)
        crop = _crop(image, fields.serial, SERIAL)
        if crop is not None:
            serial = self.model.read([crop], SERIAL)[0].field()
        return decide(path.name, row, family, display, serial, function, self.model.log_low,
                      self.model.serial_log_low)
