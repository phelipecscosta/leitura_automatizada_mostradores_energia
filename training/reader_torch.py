"""Conjunto de dados, amostragem e agrupamento do leitor (protocolo do leitor, seções
1.3, 1.12 e revisão R1).

Lê os recortes do cache (K-b). A amostragem de treino monta lotes com 64 itens de
texto, as três classes em partes iguais (B-b), mais 16 itens só da cabeça "duas
leituras", sorteados à parte (E-a, R1).
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from meter_reader.reader_input import reader_tensor, upper_band
from meter_reader.reader_text import FUNCTION_ALPHABET, TEXT_ALPHABET, encode
from training.reader_augment import augment
from training.reader_cache import crop_key, load_cache
from training.reader_dataset import UPPER

CLASSES = ("display_digital", "display_ciclometrico", "numero_serie")
TEXT_BATCH, HEAD_BATCH = 64, 16


class ReaderDataset(Dataset):
    """Itens de `reader_dataset.build_items`, lidos do cache.

    O cache é aberto de forma preguiçosa em cada processo auxiliar (Windows).
    """

    def __init__(self, items: pd.DataFrame, cache_path: Path, list_sha256: str,
                 train: bool, gray: bool = False, seed: int = 0):
        self.items = items.reset_index(drop=True)
        self.cache_path, self.list_sha256 = cache_path, list_sha256
        self.train, self.gray, self.seed = train, gray, seed
        self._cache = None

    def __len__(self) -> int:
        return len(self.items)

    def image(self, i: int) -> Image.Image:
        if self._cache is None:
            self._cache = load_cache(self.cache_path, self.list_sha256)
        r = self.items.iloc[i]
        crop = Image.fromarray(self._cache[crop_key(r["pseudonimo"], r["classe"])])
        return upper_band(crop) if r["recorte"] == UPPER else crop

    def __getitem__(self, i: int) -> dict:
        r = self.items.iloc[i]
        crop = self.image(i)
        if self.train:
            crop = augment(crop, random.Random(torch.initial_seed() + i))
        return {
            "imagem": reader_tensor(crop, gray=self.gray),
            "texto": encode(r["texto"], TEXT_ALPHABET) if r["usa_texto"] else [],
            "funcao": encode(r["funcao"], FUNCTION_ALPHABET) if r["usa_funcao"] else [],
            "duas": float(bool(r["duas_leituras"])),
            "usa_texto": bool(r["usa_texto"]), "usa_funcao": bool(r["usa_funcao"]),
            "usa_duas": bool(r["usa_duas"]), "classe": CLASSES.index(r["classe"]),
            "indice": i,
        }


def collate(batch: list[dict]) -> dict:
    """Agrupa o lote no formato da CTC: alvos concatenados e comprimentos."""
    out = {"imagem": torch.stack([b["imagem"] for b in batch])}
    for name in ("texto", "funcao"):
        out[name] = torch.tensor([k for b in batch for k in b[name]], dtype=torch.long)
        out[f"{name}_comprimento"] = torch.tensor([len(b[name]) for b in batch],
                                                  dtype=torch.long)
    for name in ("usa_texto", "usa_funcao", "usa_duas"):
        out[name] = torch.tensor([b[name] for b in batch], dtype=torch.bool)
    out["duas"] = torch.tensor([b["duas"] for b in batch], dtype=torch.float32)
    out["classe"] = torch.tensor([b["classe"] for b in batch], dtype=torch.long)
    out["indice"] = torch.tensor([b["indice"] for b in batch], dtype=torch.long)
    return out


class BalancedBatchSampler(Sampler):
    """Lotes de treino: 64 itens de texto com as classes em partes iguais, sorteados
    com reposição dentro da classe, mais 16 itens só da cabeça (R1).

    Uma época tem o número de itens de texto ÷ 64 passos (seção 1.12), salvo quando
    `steps` é dado (leitor por família, revisão R8). Os 64 itens de texto são
    divididos em partes iguais entre as classes presentes.
    """

    def __init__(self, items: pd.DataFrame, seed: int, text_batch: int = TEXT_BATCH,
                 head_batch: int = HEAD_BATCH, steps: int | None = None):
        items = items.reset_index(drop=True)
        text = items["usa_texto"].to_numpy(bool)
        self.by_class = [np.flatnonzero(text & (items["classe"] == c).to_numpy())
                         for c in CLASSES]
        self.head_only = np.flatnonzero(~text & items["usa_duas"].to_numpy(bool))
        self.text_batch, self.head_batch = text_batch, head_batch
        self.steps = steps or max(1, int(text.sum()) // text_batch)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.steps

    def __iter__(self):
        pools = [pool for pool in self.by_class if len(pool)]   # classes presentes
        shares = [self.text_batch // len(pools) + (1 if k < self.text_batch % len(pools) else 0)
                  for k in range(len(pools))]
        for _ in range(self.steps):
            batch = [int(i) for pool, n in zip(pools, shares)
                     for i in self.rng.choice(pool, size=n, replace=True)]
            if len(self.head_only) and self.head_batch:
                batch += [int(i) for i in self.rng.choice(self.head_only,
                                                          size=self.head_batch, replace=True)]
            yield batch
