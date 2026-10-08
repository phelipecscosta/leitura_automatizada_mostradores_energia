"""Modelo de triagem: espinha dorsal pré-treinada com duas cabeças.

Cabeça de cena: 3 logits (digital, ciclométrico, outros), com softmax na
inferência. Cabeça de legibilidade: 1 logit de "ilegível", com sigmoide na
inferência. As perdas recebem logits; nenhuma ativação é aplicada aqui
(spec 2.5.5; E07; E11; protocolo do Lab02, seção 2.4).
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Large_Weights,
    efficientnet_b0,
    mobilenet_v3_large,
)

# Fonte única da ordem das classes de cena (o treino importa daqui)
SCENE_CLASSES = ("digital", "ciclometrico", "outros")

# Espinhas dorsais comparadas no protocolo: construtor, pesos ImageNet e
# dimensão do vetor de características depois do pooling
BACKBONES = {
    "mobilenet_v3_large": (mobilenet_v3_large, MobileNet_V3_Large_Weights.IMAGENET1K_V2, 960),
    "efficientnet_b0": (efficientnet_b0, EfficientNet_B0_Weights.IMAGENET1K_V1, 1280),
}


class TriageModel(nn.Module):
    """Espinha dorsal (sem o classificador do ImageNet) e duas cabeças lineares."""

    def __init__(self, backbone: str = "mobilenet_v3_large", pretrained: bool = True) -> None:
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(f"Espinha dorsal desconhecida: {backbone}")
        factory, weights, dim = BACKBONES[backbone]
        base = factory(weights=weights if pretrained else None)

        self.backbone_name = backbone
        self.features = base.features          # só as camadas convolucionais
        self.pool = nn.AdaptiveAvgPool2d(1)    # aceita qualquer tamanho de entrada
        self.scene_head = nn.Linear(dim, len(SCENE_CLASSES))
        self.legibility_head = nn.Linear(dim, 1)
        self._backbone_frozen = False

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Vetor de características por imagem: (B, dim)."""
        return self.pool(self.features(x)).flatten(1)

    def heads(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Logits das duas cabeças a partir das características: (B, 3) e (B,)."""
        return self.scene_head(z), self.legibility_head(z).squeeze(1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.heads(self.embed(x))

    def freeze_backbone(self) -> None:
        """Modo extração de características: só as cabeças são treinadas."""
        for p in self.features.parameters():
            p.requires_grad = False
        self._backbone_frozen = True
        self.features.eval()

    def train(self, mode: bool = True) -> "TriageModel":
        # Espinha congelada fica sempre em avaliação: o BatchNorm não pode
        # recalcular estatísticas com os poucos exemplos do treino
        super().train(mode)
        if self._backbone_frozen:
            self.features.eval()
        return self
