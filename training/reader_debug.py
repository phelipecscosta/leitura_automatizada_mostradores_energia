"""Diagnóstico do treino do leitor (T6b.5).

`python -m training.reader_debug`: separa o efeito da precisão mista e da aumentação
(quatro variantes do L1).
`python -m training.reader_debug longo`: só o texto, 4.000 passos, L1 e L2-9.
`python -m training.reader_debug cabecas`: separa o efeito das cabeças auxiliares
(função e duas leituras) sobre o aprendizado do texto, em três variantes do L1 com
precisão mista e aumentação: só o texto; viés inicial do branco na cabeça da função;
cabeças auxiliares sobre as características desligadas do gradiente (detach).
1.500 passos cada, na primeira dobra. Só números.
"""

from __future__ import annotations

import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from meter_reader.reader import Reader
from meter_reader.reader_text import FUNCTION_ALPHABET, greedy_decode
from training.reader_loss import reader_loss
from training.reader_torch import BalancedBatchSampler, ReaderDataset, collate
from training.reader_train import EVAL_BATCH, evaluate_ma

STEPS, EVERY = 1500, 250
BLANK_BIAS = 6.0   # P(branco) inicial perto de 0,98 em cada passo


def _detached_forward(self, x):
    f = self.backbone(x)
    b, c, h, t = f.shape
    f = f.permute(0, 3, 1, 2).reshape(b, t, c * h)
    seq, _ = self.rnn(self.project(f))
    free = seq.detach()
    return {"texto": self.text_head(seq), "funcao": self.function_head(free),
            "duas": self.two_head(free.mean(dim=1)).squeeze(1)}

def _add_function_branch(model, device: str) -> None:
    """Ramo próprio da função: projeção e BiLSTM pequenas sobre as características da
    espinha sem gradiente para ela (R5, candidata C)."""
    from torch import nn

    from meter_reader.reader_text import FUNCTION_ALPHABET as ALPHABET

    model.function_branch = nn.ModuleDict({
        "proj": nn.Sequential(nn.Linear(model.project[0].in_features, 128), nn.ReLU()),
        "rnn": nn.LSTM(128, 128, batch_first=True, bidirectional=True),
        "head": nn.Linear(256, len(ALPHABET) + 1),
    }).to(device)


def _branch_forward(self, x):
    f = self.backbone(x)
    b, c, h, t = f.shape
    f = f.permute(0, 3, 1, 2).reshape(b, t, c * h)
    seq, _ = self.rnn(self.project(f))
    g, _ = self.function_branch["rnn"](self.function_branch["proj"](f.detach()))
    return {"texto": self.text_head(seq), "funcao": self.function_branch["head"](g),
            "duas": self.two_head(seq.detach().mean(dim=1)).squeeze(1)}

def function_match(model, loader, items, device) -> float:
    """Acerto exato da função nos itens digitais com alvo de função."""
    hits = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["imagem"].to(device))["funcao"].float().cpu()
            for row, idx in enumerate(batch["indice"].tolist()):
                r = items.iloc[idx]
                if r["usa_funcao"]:
                    decoded = greedy_decode(logits[row].argmax(-1).tolist(), FUNCTION_ALPHABET)
                    hits.append(decoded == r["funcao"])
    return float(np.mean(hits)) if hits else float("nan")


def variant(train_items, internal, cache, list_sha, amp: bool, aug: bool, mode: str = "padrao",
            device: str = "cuda", candidate: str = "L1", steps: int = STEPS,
            every: int = EVERY) -> None:
    torch.manual_seed(0)
    sampler = BalancedBatchSampler(train_items, seed=0)
    data = ReaderDataset(train_items, cache, list_sha, train=aug)
    loader = DataLoader(data, batch_sampler=sampler, collate_fn=collate, num_workers=6,
                        persistent_workers=True, pin_memory=True)
    internal_loader = DataLoader(ReaderDataset(internal, cache, list_sha, train=False),
                                 batch_size=EVAL_BATCH, collate_fn=collate, num_workers=2)
    model = Reader(candidate).to(device)
    if mode == "vies_branco":
        with torch.no_grad():
            model.function_head.bias[0] = BLANK_BIAS
    if mode == "separada":
        model.forward = _detached_forward.__get__(model)
    if mode == "ramo_proprio":
        _add_function_branch(model, device)
        model.forward = _branch_forward.__get__(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(device, enabled=amp)
    step, losses, flosses, skipped = 0, [], [], 0
    print(f"\n== {candidate}; {mode}; precisão mista {'sim' if amp else 'não'}; aumentação "
          f"{'sim' if aug else 'não'} ==", flush=True)
    while step < steps:
        for batch in loader:
            if mode == "so_texto":
                batch["usa_funcao"][:] = False
                batch["usa_duas"][:] = False
            model.train()
            with torch.autocast(device, dtype=torch.float16, enabled=amp):
                out = model(batch["imagem"].to(device, non_blocking=True))
            loss, parts = reader_loss(out, batch)
            optimizer.zero_grad(set_to_none=True)
            scale = scaler.get_scale() if amp else 1.0
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            skipped += int(amp and scaler.get_scale() < scale)
            losses.append(parts.get("texto", float("nan")))
            flosses.append(parts.get("funcao", float("nan")))
            step += 1
            if step % every == 0:
                ma = evaluate_ma(model, internal_loader, internal, device)
                fm = function_match(model, internal_loader, internal, device)
                print(f"passo {step:5d}: perda do texto {np.nanmean(losses[-every:]):.4f}; "
                      f"perda da função {np.nanmean(flosses[-every:]) if mode != 'so_texto' else float('nan'):.4f}; "
                      f"passos pulados {skipped}; M-a interna {ma:.3f}; "
                      f"acerto da função {fm:.3f}", flush=True)
            if step >= steps:
                break


def main() -> None:
    from meter_reader.config import get_work_dir
    from training.reader_cache import CACHE_DIR, CACHE_NAME
    from training.reader_dataset import fold_masks, load_items
    from training.reader_labels import CROPS_SHA256

    items, _ = load_items()
    lot = sorted(items["lote"].unique())[0]
    masks = fold_masks(items, lot)
    train_items = items[masks["treino"]].reset_index(drop=True)
    internal = items[masks["interno"] & items["usa_texto"]].reset_index(drop=True)
    cache = get_work_dir() / CACHE_DIR / CACHE_NAME
    if sys.argv[1:] == ["longo"]:
        for candidate in ("L1", "L2-9"):
            variant(train_items, internal, cache, CROPS_SHA256, True, True, "so_texto",
                    candidate=candidate, steps=4000, every=500)
        return
    if sys.argv[1:] == ["ramo_l2"]:
        variant(train_items, internal, cache, CROPS_SHA256, True, True, "ramo_proprio",
                candidate="L2-9", steps=2000, every=500)
        return
    if sys.argv[1:] == ["cabecas_l2"]:
        for mode in ("so_texto", "vies_branco", "separada"):
            variant(train_items, internal, cache, CROPS_SHA256, True, True, mode,
                    candidate="L2-9", steps=2000, every=500)
        return
    if sys.argv[1:] == ["cabecas"]:
        for mode in ("so_texto", "vies_branco", "separada"):
            variant(train_items, internal, cache, CROPS_SHA256, True, True, mode)
        return
    for amp, aug in ((False, False), (True, False), (False, True), (True, True)):
        variant(train_items, internal, cache, CROPS_SHA256, amp, aug)


if __name__ == "__main__":
    main()
