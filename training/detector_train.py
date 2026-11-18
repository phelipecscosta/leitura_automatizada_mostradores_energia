"""Treino dos candidatos do detector (ET5b, T5b.6; protocolo, seções 1.4 e 1.5, revisão R2).

Uma execução treina um candidato num conjunto de fotos, escolhe a época pela AP
ponderada da divisão interna e devolve os pesos dessa época. A predição é feita
em precisão completa, sem TF32, para que o que é avaliado seja o que o produto
calcula (E33); o treino usa precisão mista na GPU (T3 a).
"""
from __future__ import annotations

import copy
import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from training.detector_anchors import Candidate
from training.detector_dataset import DetectorDataset, DetectorPhoto, collate
from training.detector_eval import ap_table, selection_score
from training.finetune import lr_factor, no_tf32

# Receitas de referência do torchvision na regra linear para o lote de 8 (T2):
# taxa de aprendizado e decaimento dos pesos
RECIPES = {"S1": (0.15 * 8 / 192, 4e-5), "F1": (0.02 * 8 / 16, 1e-4), "F2": (0.02 * 8 / 16, 1e-4)}


@dataclass(frozen=True)
class DetectorTrainConfig:
    max_epochs: int = 40
    batch: int = 8
    momentum: float = 0.9
    warmup_epochs: int = 1
    amp: bool = True
    num_workers: int = 6    # treino: processos persistentes (E62)
    eval_workers: int = 2


def make_loader(dataset: DetectorDataset, cfg: DetectorTrainConfig, device: str, train: bool,
                seed: int = 0) -> DataLoader:
    """Carregador criado uma vez por execução (E62). O treino embaralha com semente e
    descarta o lote final incompleto (protocolo, 1.5)."""
    workers = cfg.num_workers if train else cfg.eval_workers
    return DataLoader(dataset, batch_size=cfg.batch, shuffle=train, drop_last=train,
                      num_workers=workers, persistent_workers=workers > 0,
                      pin_memory=device.startswith("cuda"), collate_fn=collate,
                      generator=torch.Generator().manual_seed(seed) if train else None)


def predict(model, loader: DataLoader, device: str) -> list[dict]:
    """Caixas, confianças e rótulos de cada foto, em precisão completa e sem TF32."""
    model.eval()
    out = []
    with torch.no_grad(), no_tf32():
        for images, _ in loader:
            for pred in model([x.to(device) for x in images]):
                out.append({k: pred[k].float().cpu().numpy() if k != "labels"
                            else pred[k].cpu().numpy() for k in ("boxes", "scores", "labels")})
    return out


def setup(cand: Candidate, shapes, cfg: DetectorTrainConfig, n_train: int, run_seed: int,
          device: str, pretrained: bool = True, coco_prefix: bool = False) -> dict:
    """Modelo, otimizador e cronograma de uma execução; o cronograma cobre sempre
    cfg.max_epochs épocas, também no treino final."""
    torch.manual_seed(run_seed)
    model = cand.build(shapes, pretrained_backbone=pretrained)
    if coco_prefix:  # ablação COCO (protocolo, 1.4; revisão R4, A-a)
        # Importação aqui: só a ablação precisa dos pesos COCO. A carga não consome o
        # gerador aleatório, então o resto da inicialização é igual ao do S1 (pareado).
        from training.detector_coco import coco_state, load_prefix
        load_prefix(model, coco_state())
    model = model.to(device)
    lr, weight_decay = RECIPES[cand.name]
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=cfg.momentum, weight_decay=weight_decay)
    steps = n_train // cfg.batch  # drop_last: só lotes completos
    total, warmup = cfg.max_epochs * steps, cfg.warmup_epochs * steps
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_factor(s, warmup, total))
    use_amp = cfg.amp and device.startswith("cuda")
    return {"model": model, "optimizer": optimizer, "scheduler": scheduler, "use_amp": use_amp,
            "scaler": torch.amp.GradScaler("cuda", enabled=use_amp)}


def train_epoch(run: dict, loader: DataLoader, dataset: DetectorDataset, epoch: int,
                device: str) -> float:
    """Uma época; devolve a perda média. Uma perda não finita interrompe o treino."""
    model, optimizer, scaler = run["model"], run["optimizer"], run["scaler"]
    dataset.set_epoch(epoch)
    model.train()
    total, count = 0.0, 0
    for images, targets in loader:
        images = [x.to(device) for x in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=run["use_amp"]):
            loss = sum(model(images, targets).values())
        if not math.isfinite(loss.item()):
            raise FloatingPointError("Perda não finita no treino do detector.")
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        run["scheduler"].step()
        total, count = total + loss.item(), count + 1
    return total / max(count, 1)


def train_one(cand: Candidate, shapes, fit: list[DetectorPhoto], internal: list[DetectorPhoto],
              run_seed: int, device: str, cfg: DetectorTrainConfig = DetectorTrainConfig(),
              log: Callable[[str], None] = print, pretrained: bool = True,
              coco_prefix: bool = False):
    """Treina até cfg.max_epochs épocas e devolve (modelo na melhor época, melhor época,
    histórico de (época, perda, métrica interna, segundos))."""
    fit_set, internal_set = DetectorDataset(fit, True, run_seed), DetectorDataset(internal, False)
    fit_loader = make_loader(fit_set, cfg, device, True, run_seed)
    internal_loader = make_loader(internal_set, cfg, device, False)
    run = setup(cand, shapes, cfg, len(fit), run_seed, device, pretrained, coco_prefix)
    best, best_epoch, best_state, history = -1.0, 0, None, []
    for epoch in range(1, cfg.max_epochs + 1):
        start = time.perf_counter()
        loss = train_epoch(run, fit_loader, fit_set, epoch, device)
        score = selection_score(ap_table(internal, predict(run["model"], internal_loader, device))[0])
        seconds = time.perf_counter() - start
        history.append((epoch, loss, score, seconds))
        log(f"época {epoch}: perda {loss:.4f}, AP interna {100 * score:.2f}, {seconds:.0f} s")
        if best_state is None or score > best:  # empate fica com a época anterior
            best, best_epoch = score, epoch
            best_state = copy.deepcopy({k: v.cpu() for k, v in run["model"].state_dict().items()})
    model = run["model"]
    model.load_state_dict(best_state)
    return model, best_epoch, history


def time_one_epoch(cand: Candidate, shapes, fit: list[DetectorPhoto], internal: list[DetectorPhoto],
                   device: str, cfg: DetectorTrainConfig = DetectorTrainConfig()) -> tuple[float, float]:
    """(segundos de uma época de treino, segundos da predição da divisão interna),
    sem guardar nada. Só para estimar o custo antes do treino."""
    fit_set = DetectorDataset(fit, True, 0)
    fit_loader = make_loader(fit_set, cfg, device, True, 0)
    internal_loader = make_loader(DetectorDataset(internal, False), cfg, device, False)
    run = setup(cand, shapes, cfg, len(fit), 0, device)
    start = time.perf_counter()
    train_epoch(run, fit_loader, fit_set, 1, device)
    train_s = time.perf_counter() - start
    start = time.perf_counter()
    predict(run["model"], internal_loader, device)
    return train_s, time.perf_counter() - start


def flatten_predictions(codes: list[str], preds: list[dict]) -> dict[str, np.ndarray]:
    """Predições de várias fotos em vetores planos, com o índice da foto de cada caixa."""
    def cat(parts: list[np.ndarray], empty: np.ndarray) -> np.ndarray:
        return np.concatenate(parts) if parts else empty
    return {
        "codes": np.array(codes),
        "photo": cat([np.full(len(p["scores"]), i, np.int64) for i, p in enumerate(preds)],
                     np.zeros(0, np.int64)),
        "boxes": cat([np.asarray(p["boxes"], np.float32).reshape(-1, 4) for p in preds],
                     np.zeros((0, 4), np.float32)),
        "scores": cat([np.asarray(p["scores"], np.float32) for p in preds], np.zeros(0, np.float32)),
        "labels": cat([np.asarray(p["labels"], np.int64) for p in preds], np.zeros(0, np.int64)),
    }


def unflatten_predictions(data) -> tuple[list[str], list[dict]]:
    """O inverso de flatten_predictions."""
    codes = [str(c) for c in data["codes"]]
    photo = np.asarray(data["photo"])
    preds = [{"boxes": data["boxes"][photo == i], "scores": data["scores"][photo == i],
              "labels": data["labels"][photo == i]} for i in range(len(codes))]
    return codes, preds

