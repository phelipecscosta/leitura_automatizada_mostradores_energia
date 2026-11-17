"""Detector de campos do produto (spec v16, 2.5, itens 10.1 e 10.2; ET5b).

Nesta versão (T5b.3): as classes do detector, o gerador de âncoras com as
formas escolhidas pelos dados e a montagem das duas arquiteturas candidatas do
torchvision (SSDlite e Faster R-CNN, ambas com a MobileNetV3-Large). A carga do
artefato e a decisão da segunda porta entram na T5b.8.
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

# Classes de caixa, na ordem das saídas do detector (02_dataset.md, seção 5.1).
# O rótulo 0 é o fundo; a classe i do detector é DETECTOR_CLASSES[i - 1].
DETECTOR_CLASSES = ("display_digital", "display_ciclometrico", "display_kvarh", "numero_serie")
NUM_CLASSES = len(DETECTOR_CLASSES) + 1  # mais o fundo

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
    return SSD(backbone, generator, size, NUM_CLASSES, head=head, score_thresh=0.001,
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
                      min_size=min_size, max_size=max_size, **rpn_kwargs)
