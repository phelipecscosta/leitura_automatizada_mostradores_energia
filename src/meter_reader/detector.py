"""Detector de campos do produto (spec v17, 2.5, itens 10 a 12; ET5b e ET5c).

Classes do detector, gerador de âncoras com as formas escolhidas pelos dados,
montagem do SSDlite com a MobileNetV3-Large (o S1) e, a partir da ET5c, a entrada,
a carga do artefato e a decisão: segunda porta, família do Validador 2 e seleção
dos campos (meter_reader.fields). A montagem do Faster R-CNN, que não entra no
produto (E82), sai daqui na T5c.6.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import _mobilenet_extractor as _fpn_extractor
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.ssd import SSD
from torchvision.models.detection.ssdlite import SSDLiteHead, _normal_init
from torchvision.models.detection.ssdlite import _mobilenet_extractor as _ssd_extractor
from torchvision.ops import misc as misc_nn_ops

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision.transforms import functional as TF

from meter_reader.preprocess import enhance

# Classes de caixa, na ordem das saídas do detector (02_dataset.md, seção 5.1).
# O rótulo 0 é o fundo; a classe i do detector é DETECTOR_CLASSES[i - 1].
DETECTOR_CLASSES = ("display_digital", "display_ciclometrico", "display_kvarh", "numero_serie")
NUM_CLASSES = len(DETECTOR_CLASSES) + 1  # mais o fundo

# Corte de confiança da inferência, o mesmo para todos os candidatos (revisão R3 b):
# o padrão do SSDlite. O Faster R-CNN do torchvision usa 0,05 por padrão, o que
# esconde da segunda porta e do leitor as caixas de confiança baixa.
SCORE_THRESH = 0.001

# Espinha pré-treinada no ImageNet (C2 i): a mesma da referência de detecção do torchvision
BACKBONE_WEIGHTS = MobileNet_V3_Large_Weights.IMAGENET1K_V1

Shapes = Sequence[Sequence[tuple[float, float]]]  # por nível: (largura, altura) em px da entrada


class ShapeAnchorGenerator(AnchorGenerator):
    """Âncoras com formas fixas por nível, em pixels da entrada da rede (C4 b).

    Cada nível recebe a sua lista de formas (largura, altura), centradas no
    centro de cada célula do mapa de características. A ordem é a do torchvision
    (posição antes da forma), para casar com as cabeças do SSD e da RPN. Herda de
    AnchorGenerator porque o FasterRCNN recusa outro tipo.
    """

    def __init__(self, shapes: Shapes) -> None:
        nn.Module.__init__(self)  # o __init__ do pai monta as âncoras-padrão, que não são usadas
        if not shapes or any(len(level) == 0 for level in shapes):
            raise ValueError("É preciso ao menos uma forma em cada nível.")
        self.shapes = [[(float(w), float(h)) for w, h in level] for level in shapes]
        self.cell_anchors = [torch.tensor([[-w / 2, -h / 2, w / 2, h / 2] for w, h in level],
                                          dtype=torch.float32) for level in self.shapes]

    def num_anchors_per_location(self) -> list[int]:
        return [len(level) for level in self.shapes]

    def forward(self, image_list: ImageList, feature_maps: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(feature_maps) != len(self.cell_anchors):
            raise ValueError("Número de níveis diferente do número de mapas de características.")
        height, width = image_list.tensors.shape[-2:]
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        levels = []
        for base, fmap in zip(self.cell_anchors, feature_maps):
            fh, fw = fmap.shape[-2:]
            ys = (torch.arange(fh, device=device, dtype=dtype) + 0.5) * (height / fh)
            xs = (torch.arange(fw, device=device, dtype=dtype) + 0.5) * (width / fw)
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            shifts = torch.stack([xx, yy, xx, yy], dim=-1).reshape(-1, 1, 4)
            levels.append((shifts + base.to(device, dtype).view(1, -1, 4)).reshape(-1, 4))
        anchors = torch.cat(levels)
        return [anchors for _ in image_list.image_sizes]


def build_ssd(shapes: Shapes, size: tuple[int, int] = (360, 480),
              pretrained_backbone: bool = False) -> SSD:
    """SSDlite com a MobileNetV3-Large na entrada `size` (largura, altura), sem achatar (C4 a).

    Repete a montagem do ssdlite320_mobilenet_v3_large do torchvision, que fixa
    320 × 320, trocando só o tamanho e as âncoras. São 6 níveis.
    """
    norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
    weights = BACKBONE_WEIGHTS if pretrained_backbone else None
    backbone = mobilenet_v3_large(weights=weights, norm_layer=norm_layer, reduced_tail=False)
    if weights is None:
        _normal_init(backbone)
    backbone = _ssd_extractor(backbone, 6, norm_layer)  # todas as camadas treináveis
    generator = ShapeAnchorGenerator(shapes)
    out_channels = det_utils.retrieve_out_channels(backbone, size)
    if len(out_channels) != len(generator.shapes):
        raise ValueError("O SSDlite tem 6 níveis: são precisas formas para os 6.")
    head = SSDLiteHead(out_channels, generator.num_anchors_per_location(), NUM_CLASSES, norm_layer)
    return SSD(backbone, generator, size, NUM_CLASSES, head=head, score_thresh=SCORE_THRESH,
               nms_thresh=0.55, detections_per_img=300, topk_candidates=300,
               image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5])


# Estágios da MobileNetV3 entregues à FPN (N8): passos 8, 16, 32 e 32; com o nível
# extra da FPN (passo 64), são 5 níveis. O padrão do torchvision usa só os dois
# últimos (passo 32), porque amplia a foto para 800 px de lado.
FPN_RETURNED_LAYERS = (2, 3, 4, 5)

# Propostas da RPN na inferência da versão de baixa resolução do torchvision
# (fasterrcnn_mobilenet_v3_large_320_fpn): usadas no perfil CPU (F1)
LOW_LATENCY_RPN = {"rpn_pre_nms_top_n_test": 150, "rpn_post_nms_top_n_test": 150,
                   "rpn_score_thresh": 0.05}


def build_frcnn(shapes: Shapes, min_size: int, max_size: int,
                pretrained_backbone: bool = False, trainable_layers: int = 3,
                returned_layers: Sequence[int] = FPN_RETURNED_LAYERS,
                **rpn_kwargs) -> FasterRCNN:
    """Faster R-CNN com a MobileNetV3-Large e FPN, como no torchvision, trocando as âncoras.

    Os níveis são os estágios de `returned_layers` mais o nível extra da FPN, com o
    mesmo número de formas em cada um (exigência da RPN).
    A normalização em lote é congelada, como na referência com espinha pré-treinada,
    para que a arquitetura seja a mesma com ou sem os pesos do ImageNet.
    `rpn_kwargs` vai para o FasterRCNN (por exemplo, LOW_LATENCY_RPN).
    """
    if len({len(level) for level in shapes}) != 1:
        raise ValueError("A RPN exige o mesmo número de formas em todos os níveis.")
    weights = BACKBONE_WEIGHTS if pretrained_backbone else None
    backbone = mobilenet_v3_large(weights=weights, norm_layer=misc_nn_ops.FrozenBatchNorm2d)
    backbone = _fpn_extractor(backbone, True, trainable_layers, returned_layers=list(returned_layers))
    generator = ShapeAnchorGenerator(shapes)
    if len(generator.shapes) != len(returned_layers) + 1:
        raise ValueError("É preciso uma lista de formas por nível da FPN (estágios mais o extra).")
    return FasterRCNN(backbone, NUM_CLASSES, rpn_anchor_generator=generator,
                      min_size=min_size, max_size=max_size, box_score_thresh=SCORE_THRESH,
                      **rpn_kwargs)



# ---------------------------------------------------------------------
# Família vista pelo detector, para o Validador 2 (protocolo, 1.8; D5)
# ---------------------------------------------------------------------

_FAMILY = {1: "digital", 2: "ciclometrico", 3: "ciclometrico"}  # rótulo -> família; 4 não tem


def meter_family(labels, scores, threshold: float) -> str | None:
    """Família do display de maior confiança acima do limiar (A2), ou None se só houver
    número de série ou nada (B1)."""
    best, family = -1.0, None
    for label, score in zip(labels, scores):
        label, score = int(label), float(score)
        if score >= threshold and label in _FAMILY and score > best:
            best, family = score, _FAMILY[label]
    return family

# ---------------------------------------------------------------------
# Entrada, artefato e decisão do detector do produto (ET5c, T5c.4)
# ---------------------------------------------------------------------

def load_input(path: Path) -> torch.Tensor:
    """Foto em disco -> tensor (3, A, L) em [0, 1], com o realce fixo da triagem
    (protocolo do detector, 1.5; E19, E33). A normalização é a do próprio SSD.
    Implementação única, usada também pelo treino (E17)."""
    with Image.open(path) as image:
        return TF.to_tensor(enhance(image))


def photo_score(scores) -> float:
    """Escore da foto para a segunda porta: a maior confiança entre as caixas, de
    qualquer classe; 0 sem nenhuma caixa (spec 10.3)."""
    scores = np.asarray(scores)
    return float(scores.max()) if len(scores) else 0.0


ARTIFACT_FORMAT = 1

# SHA-256 do único artefato do detector aceito por esta versão do produto (E72).
# Fica no código, e não no .env; muda junto com o artefato, num commit próprio.
PRODUCT_DETECTOR_SHA256 = "d0d0ff83f1c384f460d8efda307466c0e145b76ef8e202a73c152ed686c1e28e"


class DetectorArtifactError(ValueError):
    """Artefato do detector inválido, alterado ou incompatível com este produto."""


@dataclass(frozen=True)
class DetectionResult:
    """Decisão do detector para uma foto (spec 2.5, itens 10 a 12)."""

    boxes: np.ndarray            # (n, 4) em pixels da foto, (x0, y0, x1, y1)
    scores: np.ndarray           # (n,)
    labels: np.ndarray           # (n,), de 1 a 4, na ordem de DETECTOR_CLASSES
    score: float                 # escore da foto (maior confiança)
    passes_gate: bool            # False: "Indeterminação do tipo de medidor" (10.2)
    family: str | None           # Validador 2 (item 12): família no limiar da porta
    fields: object | None        # meter_reader.fields.FieldSelection; None se não passa


@dataclass(frozen=True)
class DetectorModel:
    """Modelo carregado: o S1 em modo de avaliação, o limiar da porta e o corte dos campos."""

    model: SSD
    gate_threshold: float
    field_cut: float
    info: dict = field(default_factory=dict)  # procedência gravada no artefato

    def decide(self, boxes, scores, labels) -> DetectionResult:
        """Porta, família e campos a partir das caixas de uma foto."""
        from meter_reader.fields import select_fields  # aqui: fields importa este módulo

        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64)
        score = photo_score(scores)
        passes = len(scores) > 0 and score >= self.gate_threshold  # sem caixa: rejeitada
        if not passes:
            return DetectionResult(boxes, scores, labels, score, False, None, None)
        return DetectionResult(boxes, scores, labels, score, True,
                               meter_family(labels, scores, self.gate_threshold),
                               select_fields(boxes, scores, labels, self.field_cut))

    def predict(self, path: Path) -> DetectionResult:
        """Uma foto em disco, em CPU: entrada, S1 e decisão."""
        with torch.no_grad():
            out = self.model([load_input(Path(path))])[0]
        return self.decide(out["boxes"].numpy(), out["scores"].numpy(), out["labels"].numpy())


def load_detector(path: Path, expected_sha256: str | None = None) -> DetectorModel:
    """Carrega o artefato do detector em CPU, conferindo o SHA-256 (se informado) e o
    contrato (formato e ordem das classes)."""
    path = Path(path)
    if expected_sha256 is not None:
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
            raise DetectorArtifactError("O artefato do detector não corresponde ao SHA-256 esperado.")
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data.get("formato") != ARTIFACT_FORMAT:
        raise DetectorArtifactError("Formato de artefato do detector desconhecido.")
    if tuple(data["classes"]) != DETECTOR_CLASSES:
        raise DetectorArtifactError("As classes do artefato não seguem a ordem DETECTOR_CLASSES.")
    model = build_ssd(data["formas"])
    model.load_state_dict(data["pesos"])
    return DetectorModel(model.eval(), float(data["limiar_porta"]), float(data["corte_campos"]),
                         dict(data["info"]))
