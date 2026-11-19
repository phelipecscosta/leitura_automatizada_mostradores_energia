"""Seleção dos campos a partir das caixas do detector (spec v18, 2.5, item 10.1; E83 e E86).

Regras da revisão R4 do protocolo do detector (exploracao/detector_README.md, pesquisa):
corte de confiança K (K2) -> junção das caixas de display de classes diferentes na
mesma região, com IoU >= 0,5, ficando a classe da mais confiante (M1) -> família pela
região mais confiante -> no ciclométrico, o display mais próximo do número de série é o
kvarh e é descartado. Os casos ambíguos (G1, provisório até a ET6) seguem para
"Verificar". O digital não tem regra de posição: vale a região digital mais confiante.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from meter_reader.detector import DETECTOR_CLASSES

MERGE_IOU = 0.5  # M1: o mesmo IoU do critério de acerto (D1 c)
_DIGITAL, _CYCLO, _KVARH, _SERIAL = (DETECTOR_CLASSES.index(c) + 1 for c in DETECTOR_CLASSES)
_DISPLAYS = (_DIGITAL, _CYCLO, _KVARH)

# Casos ambíguos da regra do kvarh (G1); todos recebem o motivo CONSUMPTION_NOT_IDENTIFIED (E86)
AMBIGUOUS_SAME_CLASS = "mesma_classe_sem_numero_serie"
AMBIGUOUS_MANY = "mais_de_dois_displays"
AMBIGUOUS_NO_CONSUMPTION = "sem_display_de_consumo"


@dataclass(frozen=True)
class FieldSelection:
    """Campos escolhidos numa foto. `family` é None quando não há display acima do corte."""
    family: str | None
    consumption: np.ndarray | None  # caixa (x0, y0, x1, y1) do display de consumo
    serial: np.ndarray | None       # caixa do número de série mais confiante
    ambiguity: str | None = None


def _iou(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    """IoU de uma caixa com cada uma de `others`."""
    x0 = np.maximum(box[0], others[:, 0])
    y0 = np.maximum(box[1], others[:, 1])
    x1 = np.minimum(box[2], others[:, 2])
    y1 = np.minimum(box[3], others[:, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / np.maximum(area(box) + area(others) - inter, 1e-12)


def merge_displays(boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray,
                   iou: float = MERGE_IOU) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Regiões de display (M1): da mais confiante para a menos, uma caixa só entra se não
    cobrir com IoU >= `iou` uma região já aceita. Devolve as regiões em ordem decrescente
    de confiança, cada uma com a classe da sua caixa."""
    order = np.argsort(-scores, kind="stable")
    kept: list[int] = []
    for i in order:
        if not kept or (_iou(boxes[i], boxes[kept]) < iou).all():
            kept.append(int(i))
    return boxes[kept], scores[kept], labels[kept]


def _center(box: np.ndarray) -> np.ndarray:
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])


def select_fields(boxes, scores, labels, cut: float) -> FieldSelection:
    """Aplica as regras da R4 às caixas de uma foto, com o corte de confiança `cut`."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    above = scores >= cut

    is_serial = above & (labels == _SERIAL)
    serial = boxes[is_serial][np.argmax(scores[is_serial])] if is_serial.any() else None

    is_display = above & np.isin(labels, _DISPLAYS)
    regions, _, classes = merge_displays(boxes[is_display], scores[is_display],
                                         labels[is_display])
    if len(regions) == 0:  # só número de série, ou nada acima do corte: caminho da ET6
        return FieldSelection(None, None, serial)
    if classes[0] == _DIGITAL:  # família pela região mais confiante (A2)
        return FieldSelection("digital", regions[0], serial)

    cyc = classes != _DIGITAL  # display de consumo ou kvarh
    cyc_boxes, cyc_classes = regions[cyc], classes[cyc]
    if len(cyc_boxes) > 2:
        return FieldSelection("ciclometrico", None, serial, AMBIGUOUS_MANY)
    if len(cyc_boxes) == 1:
        if cyc_classes[0] == _CYCLO:
            return FieldSelection("ciclometrico", cyc_boxes[0], serial)
        return FieldSelection("ciclometrico", None, serial, AMBIGUOUS_NO_CONSUMPTION)
    if serial is not None:  # dois displays: o mais próximo do número de série é o kvarh (E83)
        dist = [np.linalg.norm(_center(b) - _center(serial)) for b in cyc_boxes]
        return FieldSelection("ciclometrico", cyc_boxes[int(np.argmax(dist))], serial)
    if cyc_classes[0] != cyc_classes[1]:  # sem número de série: vale a classe
        return FieldSelection("ciclometrico", cyc_boxes[cyc_classes == _CYCLO][0], serial)
    return FieldSelection("ciclometrico", None, serial, AMBIGUOUS_SAME_CLASS)
