"""Cabeças da função e de duas leituras do leitor (revisões R11 e R12 do protocolo do
leitor; ET6c, T6c.5).

As cabeças leem os atributos do tronco congelado de cada semente. Ficam no produto
porque a inferência precisa delas e o produto nunca importa o pacote training; o
treino (training/reader_heads_train.py) as importa daqui.
"""

from __future__ import annotations

import torch
from torch import nn

from meter_reader.reader_text import CONSUMPTION_CLASSES, FUNCTION_CLASSES  # noqa: F401

HEAD_HIDDEN = 128


def trunk_features(model, x: torch.Tensor) -> torch.Tensor:
    """Atributos do tronco como no `Reader.forward`: (B, T, C * H)."""
    f = model.backbone(x)
    b, c, h, t = f.shape
    return f.permute(0, 3, 1, 2).reshape(b, t, c * h)


class Heads(nn.Module):
    """Cabeça da função (classificação) e de duas leituras sobre os atributos do tronco."""

    def __init__(self, feat_dim: int, hidden: int = HEAD_HIDDEN,
                 n_classes: int = len(FUNCTION_CLASSES)):
        super().__init__()
        self.f_project = nn.Sequential(nn.Linear(feat_dim, hidden), nn.ReLU(inplace=True))
        self.f_rnn = nn.LSTM(hidden, hidden, bidirectional=True, batch_first=True)
        self.f_out = nn.Linear(4 * hidden, n_classes)   # média e máximo no tempo
        self.two = nn.Sequential(nn.Linear(feat_dim, hidden), nn.ReLU(inplace=True),
                                 nn.Linear(hidden, 1))

    def forward(self, f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        g, _ = self.f_rnn(self.f_project(f))
        func = self.f_out(torch.cat([g.mean(1), g.amax(1)], dim=-1))
        return func, self.two(f.mean(1)).squeeze(1)
