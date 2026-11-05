"""Ajuste fino da triagem de cinco classes: C2, C3 e C4 (protocolo, 1.2 a 1.4).

Cada candidato é uma espinha pré-treinada com uma camada linear de 12 saídas
(grupos de treino). Entrada: realce fixo e, no treino, a aumentação única
(revisão R2) com semente por foto, execução e época.
"""

from __future__ import annotations

import zlib

import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

from meter_reader.preprocess import enhance, to_tensor
from training.dino_features import load_dinov2, model_input
from training.triage_augment import augment


def epoch_seed(pseudonym: str, run_seed: int, epoch: int) -> int:
    """Semente estável da vista de uma foto numa época de uma execução."""
    return zlib.crc32(f"{pseudonym}:{run_seed}:{epoch}".encode("utf-8"))


class PhotoDataset(Dataset):
    """Fotos com realce; no treino, uma vista nova a cada época."""

    def __init__(self, paths, codes, targets, to_input, train: bool, run_seed: int = 0):
        self.paths, self.codes, self.targets = list(paths), list(codes), list(targets)
        self.to_input, self.train, self.run_seed = to_input, train, run_seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as image:
            view = enhance(image)
        if self.train:
            view = augment(view, epoch_seed(self.codes[i], self.run_seed, self.epoch))
        return self.to_input(view), self.targets[i]


class MobileNetClassifier(nn.Module):
    """C2: MobileNetV3-Large inteira e uma camada linear (a mesma cabeça do C0)."""

    def __init__(self, n_outputs: int, pretrained: bool = True) -> None:
        super().__init__()
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = mobilenet_v3_large(weights=weights).features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(960, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.backbone(x)).flatten(1))


class DinoClassifier(nn.Module):
    """C3: DINOv2 ViT-S/14 inteiro (token de classe) e uma camada linear."""

    def __init__(self, n_outputs: int, pretrained: bool = True) -> None:
        super().__init__()
        self.backbone = load_dinov2(pretrained).requires_grad_(True)
        self.head = nn.Linear(384, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


# Candidato -> (classe do modelo, entrada de uma vista realçada)
# C2: resolução nativa (480 × 360); C3: 224 px com corte em múltiplo de 14 (E37)
CANDIDATES = {
    "C2": (MobileNetClassifier, to_tensor),
    "C3": (DinoClassifier, model_input),
}


def build(candidate: str, n_outputs: int, pretrained: bool = True):
    """(modelo, função de entrada) do candidato."""
    model_class, to_input = CANDIDATES[candidate]
    return model_class(n_outputs, pretrained), to_input

# ---------------------------------------------------------------------
# Treino de uma dobra e uma semente (protocolo, 1.2 a 1.4)
# ---------------------------------------------------------------------

import copy  # noqa: E402
import gc    # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from dataclasses import dataclass  # noqa: E402

import numpy as np  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from training.triage_eval import class_nll  # noqa: E402


@dataclass(frozen=True)
class TrainConfig:
    lr: float                      # taxa da espinha (e da cabeça, se head_lr for None)
    head_lr: float | None
    max_epochs: int
    weight_decay: float = 0.05
    warmup_epochs: int = 1
    batch: int = 32                # lote efetivo
    micro_batch: int = 32          # = batch: sem acumulação, porque o lote cabe na memória (1.3)
    amp: bool = True
    num_workers: int = 6           # processos que preparam as fotos enquanto a GPU trabalha
    eval_workers: int = 2          # predição: poucas fotos; menos processos reservando memória


CONFIGS = {
    "C2": TrainConfig(lr=3e-4, head_lr=None, max_epochs=15),
    "C3": TrainConfig(lr=2e-5, head_lr=1e-3, max_epochs=10),
}


def lr_factor(step: int, warmup_steps: int, total_steps: int) -> float:
    """Aquecimento linear de 0 a 1 e depois cosseno até 0."""
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


@contextmanager
def no_tf32():
    """Precisão cheia nas convoluções e nas multiplicações (E33, E37)."""
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        with torch.backends.cudnn.flags(enabled=True, benchmark=False,
                                        deterministic=False, allow_tf32=False):
            yield
    finally:
        torch.set_float32_matmul_precision(previous)


def make_loader(dataset, cfg: TrainConfig, device: str, shuffle: bool = False,
                generator: torch.Generator | None = None) -> DataLoader:
    """Carregador criado uma vez por execução. No Windows, cada criação dos
    processos auxiliares custa cerca de 27 s (medido): nunca recriar por época."""
    return DataLoader(dataset, batch_size=cfg.micro_batch if shuffle else cfg.batch,
                      shuffle=shuffle,
                      num_workers=cfg.num_workers if shuffle else cfg.eval_workers,
                      persistent_workers=(cfg.num_workers if shuffle else cfg.eval_workers) > 0,
                      pin_memory=device.startswith("cuda"), generator=generator)


def predict(model, loader: DataLoader, device: str) -> np.ndarray:
    """Logits em FP32, sem TF32 e sem aumentação: o que é avaliado e entregue."""
    model.eval()
    chunks = []
    with torch.no_grad(), no_tf32():
        for x, _ in loader:
            chunks.append(model(x.to(device)).float().cpu().numpy())
    return np.concatenate(chunks)


def train_one(candidate: str, train_set: PhotoDataset, internal_set: PhotoDataset,
              internal_class: np.ndarray, held_set: PhotoDataset, run_seed: int,
              output_weight: np.ndarray | None = None, device: str = "cuda",
              config: TrainConfig | None = None, pretrained: bool = True, log=print):
    """Treina, escolhe a época pela divisão interna e prediz o lote avaliado.

    `output_weight`: peso de cada saída (W-d) ou None (W-b). `internal_class`:
    classe do produto das fotos da divisão interna. Devolve
    (logits do lote avaliado, melhor época, histórico de perdas).
    """
    cfg = config or CONFIGS[candidate]
    torch.manual_seed(run_seed)
    model, _ = build(candidate, 12, pretrained)
    model.to(device)

    groups = ([{"params": model.parameters(), "lr": cfg.lr}] if cfg.head_lr is None else
              [{"params": model.backbone.parameters(), "lr": cfg.lr},
               {"params": model.head.parameters(), "lr": cfg.head_lr}])
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
    accumulation = cfg.batch // cfg.micro_batch
    steps_per_epoch = math.ceil(len(train_set) / cfg.batch)
    total, warmup = cfg.max_epochs * steps_per_epoch, cfg.warmup_epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_factor(s, warmup, total))

    weight = None if output_weight is None else torch.as_tensor(output_weight, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=None if weight is None else weight.to(device))
    correction = 0.0 if output_weight is None else np.log(output_weight)
    use_amp = cfg.amp and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    # persistent_workers: os processos auxiliares sobrevivem entre as épocas (no Windows,
    # criá-los custa alguns segundos). A vista de cada foto depende só da semente e da
    # época (epoch_seed), então o paralelismo não muda nenhuma vista.
    # Os processos auxiliares sobrevivem entre as épocas. A vista de cada foto depende
    # só da semente e da época (epoch_seed), então o paralelismo não muda nenhuma vista.
    loader = make_loader(train_set, cfg, device, shuffle=True,
                         generator=torch.Generator().manual_seed(run_seed))
    internal_loader = make_loader(internal_set, cfg, device)

    best = (math.inf, 0, None)  # (perda interna, época, pesos)
    history = []
    for epoch in range(1, cfg.max_epochs + 1):
        start = time.perf_counter()
        train_set.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for i, (x, y) in enumerate(loader, start=1):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss = criterion(model(x.to(device)), y.to(device)) / accumulation
            scaler.scale(loss).backward()
            if i % accumulation == 0 or i == len(loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

        internal = predict(model, internal_loader, device) - correction
        nll = class_nll(internal, internal_class)
        history.append(nll)
        if nll < best[0]:
            best = (nll, epoch, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        log(f"época {epoch}: perda interna {nll:.4f} ({(time.perf_counter() - start) / 60:.1f} min)")

     # Encerra os processos do treino e da divisão interna antes de criar os da predição
    # final: no Windows, cada processo reserva 1 a 2 GB ao carregar o CUDA, e 18 processos
    # simultâneos esgotaram a memória virtual (WinError 1455, medido na primeira execução)
    del loader, internal_loader
    gc.collect()
    model.load_state_dict(best[2])
    held = predict(model, make_loader(held_set, cfg, device), device) - correction
    return held, best[1], history
