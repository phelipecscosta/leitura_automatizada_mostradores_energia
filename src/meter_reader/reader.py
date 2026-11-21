"""Modelo do leitor: CRNN com duas cabeças de CTC e a cabeça "duas leituras"
(protocolo do leitor, seção 1.5; F-c, D-b).

Espinha convolucional -> as faixas de altura que sobram são concatenadas por coluna
(a posição vertical é preservada) -> projeção -> BiLSTM de 2 camadas -> três cabeças:
texto principal (CTC), função (CTC) e duas leituras (média no tempo, um logit).
Revisão R5: só o texto treina a espinha e a BiLSTM principal; a função tem um ramo
próprio (projeção e BiLSTM pequenas) sobre as características da espinha sem
gradiente para ela, e duas leituras usa a saída da BiLSTM principal sem gradiente.
Candidatos (revisão R2): L1 (CNN própria, do zero); L2-9 e L2-12 (as 9 e as 12
primeiras camadas da MobileNetV3-Small, pesos ImageNet).
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from meter_reader.reader_input import READER_HEIGHT, READER_WIDTH
from meter_reader.reader_text import FUNCTION_ALPHABET, TEXT_ALPHABET

HIDDEN = 256                 # unidades por direção da BiLSTM
PROJECTION = 256
FUNCTION_HIDDEN = 128        # ramo próprio da função (R5)
L1_CHANNELS = (32, 64, 128, 128, 256, 256)
L1_POOLS = ((2, 2), (2, 2), (2, 1), (2, 1), None, None)   # passo total 16 x 4
L2_DEPTHS = {"L2-9": 9, "L2-12": 12}   # camadas da MobileNetV3-Small (revisão R2)
CANDIDATES = ("L1", *L2_DEPTHS)
L2_STRIDE = (8, 4)           # passo total (altura, largura)


def _l1_backbone() -> tuple[nn.Module, int]:
    layers, cin = [], 3
    for cout, pool in zip(L1_CHANNELS, L1_POOLS):
        layers += [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
                   nn.ReLU(inplace=True)]
        if pool:
            layers.append(nn.MaxPool2d(pool))
        cin = cout
    return nn.Sequential(*layers), cin


def _limit_strides(features: nn.Sequential, target: tuple[int, int]) -> None:
    """Percorre as convoluções na ordem e só deixa o passo 2 enquanto o passo total
    não atinge o alvo, em cada eixo; as demais passam a ter passo 1 naquele eixo."""
    total = [1, 1]
    for module in features.modules():
        if isinstance(module, nn.Conv2d) and module.stride != (1, 1):
            stride = []
            for axis in (0, 1):
                s = module.stride[axis] if total[axis] < target[axis] else 1
                total[axis] *= s
                stride.append(s)
            module.stride = tuple(stride)


def _l2_backbone(depth: int, pretrained: bool) -> tuple[nn.Module, int]:
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    features = mobilenet_v3_small(weights=weights).features[:depth]
    _limit_strides(features, L2_STRIDE)
    with torch.no_grad():
        channels = features(torch.zeros(1, 3, READER_HEIGHT, READER_WIDTH)).shape[1]
    return features, channels


class Reader(nn.Module):
    """Saída: logits do texto (B, T, 37), da função (B, T, 11) e de duas leituras (B,)."""

    def __init__(self, candidate: str = "L1", pretrained: bool = True):
        super().__init__()
        if candidate == "L1":
            self.backbone, channels = _l1_backbone()
        elif candidate in L2_DEPTHS:
            self.backbone, channels = _l2_backbone(L2_DEPTHS[candidate], pretrained)
        else:
            raise ValueError("Candidato do leitor desconhecido.")
        self.candidate = candidate
        with torch.no_grad():
            _, c, h, _ = self.backbone(torch.zeros(1, 3, READER_HEIGHT, READER_WIDTH)).shape
        self.project = nn.Sequential(nn.Linear(c * h, PROJECTION), nn.ReLU(inplace=True))
        self.rnn = nn.LSTM(PROJECTION, HIDDEN, num_layers=2, bidirectional=True,
                           batch_first=True)
        self.text_head = nn.Linear(2 * HIDDEN, len(TEXT_ALPHABET) + 1)
        self.function_project = nn.Sequential(nn.Linear(c * h, FUNCTION_HIDDEN),
                                              nn.ReLU(inplace=True))
        self.function_rnn = nn.LSTM(FUNCTION_HIDDEN, FUNCTION_HIDDEN, bidirectional=True,
                                    batch_first=True)
        self.function_head = nn.Linear(2 * FUNCTION_HIDDEN, len(FUNCTION_ALPHABET) + 1)
        self.two_head = nn.Linear(2 * HIDDEN, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        f = self.backbone(x)                                # (B, C, H, T)
        b, c, h, t = f.shape
        f = f.permute(0, 3, 1, 2).reshape(b, t, c * h)      # faixas concatenadas por coluna
        seq, _ = self.rnn(self.project(f))                  # (B, T, 2 * HIDDEN)
        # R5: as cabeças auxiliares não devolvem gradiente ao tronco do texto
        g, _ = self.function_rnn(self.function_project(f.detach()))
        return {"texto": self.text_head(seq), "funcao": self.function_head(g),
                "duas": self.two_head(seq.detach().mean(dim=1)).squeeze(1)}
