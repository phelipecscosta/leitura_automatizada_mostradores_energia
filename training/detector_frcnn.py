"""Faster R-CNN dos candidatos F1 e F2 do detector (ET5b; protocolo, seção 1.3).

Saiu do pacote do produto na ET5c (opção 3a): o Faster R-CNN não entra no produto
(E82), mas a montagem continua aqui para que o treino e o relatório da ET5b sejam
reproduzíveis. Usa as âncoras e as constantes do detector do produto.
"""
from __future__ import annotations

from collections.abc import Sequence

from torchvision.models import mobilenet_v3_large
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import _mobilenet_extractor as _fpn_extractor
from torchvision.ops import misc as misc_nn_ops

from meter_reader.detector import (BACKBONE_WEIGHTS, NUM_CLASSES, SCORE_THRESH,
                                   ShapeAnchorGenerator, Shapes)

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
