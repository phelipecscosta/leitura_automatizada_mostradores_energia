import math

import numpy as np
from PIL import Image

from meter_reader.contracts import ExtractionRow, Reason, Status
from meter_reader.crops import crop_box
from meter_reader.detector import DetectionResult
from meter_reader.fields import FieldSelection
from meter_reader.reader_model import ReaderOutput
from meter_reader.reader_text import FUNCTION_CLASSES
from meter_reader.reading import ArtifactReader

ROW = ExtractionRow(("A123", "B9"), 12345, None, "a.jpg")
BOX = np.array([10.0, 10.0, 60.0, 30.0])


def _probs(code):
    out = dict.fromkeys(FUNCTION_CLASSES, 0.0)
    out[code] = 1.0
    return out


class FakeModel:
    log_low, function_threshold = math.log(0.2), 0.7258

    def __init__(self, display="12345", serial="A123", code="03"):
        self.texts = {"display_digital": display, "display_ciclometrico": display,
                      "numero_serie": serial}
        self.code, self.calls = code, []

    def read(self, crops, field_class):
        self.calls.append((field_class, crops[0].size))
        f = _probs(self.code) if field_class == "display_digital" else None
        return [ReaderOutput(self.texts[field_class], -0.1, 0.9, f,
                             False if f is not None else None)]


def _photo(tmp_path):
    path = tmp_path / "a.jpg"
    Image.new("RGB", (200, 100), "white").save(path)
    return path


def _detection(family, consumption=BOX, serial=BOX):
    fields = FieldSelection(family, consumption, serial)
    return DetectionResult(np.zeros((0, 4)), np.zeros(0), np.zeros(0, int), 0.9, True,
                           family, fields)


def test_digital_aprovado_com_funcao(tmp_path):
    model = FakeModel()
    out = ArtifactReader(model, "v").read(_photo(tmp_path), ROW, "digital", _detection("digital"))
    assert (out.status, out.function, out.reading) == (Status.APPROVED, "03", 12345)
    assert [c for c, _ in model.calls] == ["display_digital", "numero_serie"]
    x0, y0, x1, y1 = crop_box(BOX, "display_digital", 200, 100)   # com a margem da classe
    assert model.calls[0][1] == (x1 - x0, y1 - y0)


def test_digital_com_funcao_incorreta(tmp_path):
    out = ArtifactReader(FakeModel(code="24"), "v").read(_photo(tmp_path), ROW, "digital",
                                                         _detection("digital"))
    assert (out.status, out.reason) == (Status.NEEDS_REVIEW, Reason.WRONG_FUNCTION)


def test_ciclometrico_em_transicao(tmp_path):
    out = ArtifactReader(FakeModel(display="12346"), "v").read(
        _photo(tmp_path), ROW, "ciclometrico", _detection("ciclometrico"))
    assert (out.status, out.reading, out.function) == (Status.APPROVED, 12345, None)


def test_so_numero_de_serie(tmp_path):
    model = FakeModel()
    out = ArtifactReader(model, "v").read(_photo(tmp_path), ROW, "digital",
                                          _detection(None, consumption=None))
    assert (out.status, out.reason) == (Status.NEEDS_REVIEW, Reason.LOW_CONFIDENCE)
    assert [c for c, _ in model.calls] == ["numero_serie"]


def test_sem_campos(tmp_path):
    out = ArtifactReader(FakeModel(), "v").read(_photo(tmp_path), ROW, "digital",
                                                _detection(None, consumption=None, serial=None))
    assert out.status is Status.REJECTED_LOW_QUALITY


def test_caixa_fora_da_foto_vira_campo_ausente(tmp_path):
    far = np.array([500.0, 500.0, 600.0, 600.0])
    model = FakeModel()
    out = ArtifactReader(model, "v").read(_photo(tmp_path), ROW, "digital",
                                          _detection("digital", serial=far))
    assert (out.status, out.reason) == (Status.NEEDS_REVIEW, Reason.METER_NUMBER_NOT_FOUND)
