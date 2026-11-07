"""Triagem de cinco classes do produto (spec v13, 2.5, itens 5 a 8; E65; revisão R6).

O modelo do produto é o C4: 3 redes MobileNetV3-Large (uma por semente), cada
uma com 12 saídas de grupo. As saídas são combinadas pela média dos logits e
divididas pela temperatura (E31, E32); as probabilidades dos grupos são
somadas por classe do produto (TRIAGE_GROUP_CLASS) e o escore de rejeição é a
soma das três classes rejeitáveis.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

from meter_reader.contracts import PRODUCT_CLASSES, TRIAGE_GROUP_CLASS, TRIAGE_GROUPS

REJECTABLE_CLASSES = ("indeterminado", "multiplos", "outros")
_REJECTABLE = [PRODUCT_CLASSES.index(c) for c in REJECTABLE_CLASSES]

# Matriz (12 grupos × 5 classes) com 1 onde o grupo pertence à classe:
# P(classes) = P(grupos) @ matriz
_GROUP_TO_CLASS = torch.zeros(len(TRIAGE_GROUPS), len(PRODUCT_CLASSES))
_GROUP_TO_CLASS[torch.arange(len(TRIAGE_GROUPS)), torch.tensor(TRIAGE_GROUP_CLASS)] = 1.0


class MobileNetTriageNet(nn.Module):
    """MobileNetV3-Large inteira e uma camada linear de 12 saídas (rede do C2 e do C4).

    `pretrained` só é usado no treino (pesos do ImageNet como ponto de partida);
    no produto, os pesos vêm do artefato da triagem.
    """

    def __init__(self, n_outputs: int = len(TRIAGE_GROUPS), pretrained: bool = False) -> None:
        super().__init__()
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = mobilenet_v3_large(weights=weights).features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(960, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.backbone(x)).flatten(1))


class TriageEnsemble(nn.Module):
    """Média dos logits das redes, dividida pela temperatura: logits de grupo calibrados."""

    def __init__(self, nets: list[nn.Module], temperature: float) -> None:
        super().__init__()
        if not nets or temperature <= 0:
            raise ValueError("É preciso ao menos uma rede e uma temperatura positiva.")
        self.nets = nn.ModuleList(nets)
        self.register_buffer("temperature", torch.tensor(float(temperature)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.stack([net(x) for net in self.nets]).mean(dim=0)
        return logits / self.temperature


def class_probs(group_logits: torch.Tensor) -> torch.Tensor:
    """(n, 12) logits de grupo calibrados -> (n, 5) probabilidades das classes do produto."""
    return torch.softmax(group_logits, dim=1) @ _GROUP_TO_CLASS.to(group_logits)


def rejection_score(probs: torch.Tensor) -> torch.Tensor:
    """P(rejeitável) = P(indeterminado) + P(múltiplos) + P(outros) (spec 2.5, item 8)."""
    return probs[:, _REJECTABLE].sum(dim=1)


def rejection_destination(probs: torch.Tensor) -> list[str]:
    """Classe de destino de cada foto rejeitada: a maior das três rejeitáveis."""
    best = probs[:, _REJECTABLE].argmax(dim=1)
    return [REJECTABLE_CLASSES[i] for i in best.tolist()]



# ---------------------------------------------------------------------
# Artefato da triagem: as 3 redes, a temperatura e o limiar num arquivo (E32)
# ---------------------------------------------------------------------

ARTIFACT_FORMAT = 1


class TriageArtifactError(ValueError):
    """Artefato da triagem inválido, alterado ou incompatível com este produto."""


@dataclass(frozen=True)
class TriageModel:
    """Modelo carregado: o conjunto calibrado e o limiar do escore de rejeição."""

    ensemble: TriageEnsemble
    threshold: float
    info: dict = field(default_factory=dict)  # procedência gravada no artefato


def load_triage(path: Path, expected_sha256: str | None = None) -> TriageModel:
    """Carrega o artefato do C4 em CPU, conferindo o SHA-256 (se informado) e o contrato."""
    path = Path(path)
    if expected_sha256 is not None:
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
            raise TriageArtifactError("O artefato da triagem não corresponde ao SHA-256 esperado.")
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data.get("formato") != ARTIFACT_FORMAT:
        raise TriageArtifactError("Formato de artefato da triagem desconhecido.")
    if tuple(data["grupos"]) != TRIAGE_GROUPS:
        raise TriageArtifactError("As saídas do artefato não seguem a ordem TRIAGE_GROUPS.")
    nets = []
    for state in data["redes"]:
        net = MobileNetTriageNet()
        net.load_state_dict(state)
        nets.append(net)
    ensemble = TriageEnsemble(nets, data["temperatura"]).eval()
    return TriageModel(ensemble, float(data["limiar"]), dict(data["info"]))
