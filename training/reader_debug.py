"""Diagnóstico do treino do leitor (T6b.5): separa o efeito da precisão mista e da
aumentação quando o treino não aprende. Quatro variantes do L1 na primeira dobra,
1.500 passos cada, com a perda do texto, os passos pulados pela precisão mista e a
M-a na divisão interna. Só números.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from meter_reader.reader import Reader
from training.reader_loss import reader_loss
from training.reader_torch import BalancedBatchSampler, ReaderDataset, collate
from training.reader_train import EVAL_BATCH, evaluate_ma

STEPS, EVERY = 1500, 250


def variant(train_items, internal, cache, list_sha, amp: bool, aug: bool,
            device: str = "cuda") -> None:
    torch.manual_seed(0)
    sampler = BalancedBatchSampler(train_items, seed=0)
    data = ReaderDataset(train_items, cache, list_sha, train=aug)
    loader = DataLoader(data, batch_sampler=sampler, collate_fn=collate, num_workers=6,
                        persistent_workers=True, pin_memory=True)
    internal_loader = DataLoader(ReaderDataset(internal, cache, list_sha, train=False),
                                 batch_size=EVAL_BATCH, collate_fn=collate, num_workers=2)
    model = Reader("L1").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(device, enabled=amp)
    step, losses, skipped = 0, [], 0
    print(f"\n== precisão mista {'sim' if amp else 'não'}; aumentação "
          f"{'sim' if aug else 'não'} ==", flush=True)
    while step < STEPS:
        for batch in loader:
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
            step += 1
            if step % EVERY == 0:
                ma = evaluate_ma(model, internal_loader, internal, device)
                print(f"passo {step:5d}: perda do texto {np.nanmean(losses[-EVERY:]):.4f}; "
                      f"não finitas {int(np.sum(~np.isfinite(losses[-EVERY:])))}; "
                      f"passos pulados {skipped}; M-a interna {ma:.3f}", flush=True)
            if step >= STEPS:
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
    for amp, aug in ((False, False), (True, False), (False, True), (True, True)):
        variant(train_items, internal, cache, CROPS_SHA256, amp, aug)


if __name__ == "__main__":
    main()
