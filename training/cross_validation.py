"""Validação por lote deixado de fora para as cabeças lineares (D3).

Protocolo do Lab02: seções 2.2 (dobras), 2.4 (perdas), 2.5 (hiperparâmetros)
e revisão de 08/10 (padronização com o treino de cada dobra).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from meter_reader.triage import SCENE_CLASSES
from training import triage_loss as tl
from training.labels import TEST_SUBSET, LabeledPhoto

EPOCHS = 300
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4


class LinearHeads(nn.Module):
    """As duas cabeças lineares, com os mesmos nomes do TriageModel."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scene_head = nn.Linear(dim, len(SCENE_CLASSES))
        self.legibility_head = nn.Linear(dim, 1)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.scene_head(z), self.legibility_head(z).squeeze(1)


def fit_heads(z: np.ndarray, scene: np.ndarray, illegible: np.ndarray, seed: int,
              device: str = "cpu") -> tuple[LinearHeads, list[float]]:
    """Treina as cabeças em lote completo; devolve as cabeças e a curva de perda."""
    torch.manual_seed(seed)  # mesma inicialização para a mesma semente
    heads = LinearHeads(z.shape[1]).to(device)
    z_t = torch.as_tensor(z, device=device)
    scene_t = torch.as_tensor(scene, device=device)
    leg_t = torch.as_tensor(illegible, device=device)
    weights, pos_weight = tl.scene_weights(scene_t), tl.legibility_pos_weight(leg_t)

    optimizer = torch.optim.AdamW(heads.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    history = []
    for _ in range(EPOCHS):
        optimizer.zero_grad()
        total, _, _ = tl.triage_loss(*heads(z_t), scene_t, leg_t, weights, pos_weight)
        total.backward()
        optimizer.step()
        history.append(total.item())
    return heads, history


@dataclass(frozen=True)
class FoldResult:
    held_out_lot: str
    n_train: int
    final_train_loss: float


def cross_validate(features: np.ndarray, photos: list[LabeledPhoto], seed: int,
                   train_fraction: float = 1.0, device: str = "cpu"):
    """Predições fora da dobra: (p_cena (n, 3), p_ilegível (n,), resultados por dobra)."""
    if any(p.subset == TEST_SUBSET for p in photos):
        raise ValueError("O lote de teste não entra na validação cruzada (protocolo 2.1).")

    lots = np.array([p.lot for p in photos])
    scene = np.array([p.scene for p in photos])
    illegible = np.array([p.illegible for p in photos])
    p_scene = np.full((len(photos), len(SCENE_CLASSES)), np.nan, dtype=np.float32)
    p_illegible = np.full(len(photos), np.nan, dtype=np.float32)
    rng = np.random.default_rng(seed)  # mesmo sorteio para todos os modelos da semente
    folds = []

    for lot in sorted(set(lots)):
        train = np.flatnonzero(lots != lot)
        held = np.flatnonzero(lots == lot)
        if train_fraction < 1.0:
            size = round(train_fraction * len(train))
            train = np.sort(rng.choice(train, size=size, replace=False))

        mean = features[train].mean(axis=0)
        std = features[train].std(axis=0)
        std[std < 1e-6] = 1.0  # dimensão constante no treino: não amplifica ruído
        z_train = (features[train] - mean) / std
        z_held = (features[held] - mean) / std

        heads, history = fit_heads(z_train, scene[train], illegible[train], seed, device)
        with torch.no_grad():
            scene_logits, leg_logits = heads(torch.as_tensor(z_held, device=device))
        p_scene[held] = torch.softmax(scene_logits, dim=1).cpu().numpy()
        p_illegible[held] = torch.sigmoid(leg_logits).cpu().numpy()
        folds.append(FoldResult(str(lot), len(train), history[-1]))

    return p_scene, p_illegible, folds
