"""Verificações de sanidade do leitor antes do treino (T6b.4).

Para cada candidato: perda inicial; sobreajuste de um lote fixo de recortes reais
(sem aumentação) até a perda quase zerar e o acerto chegar a 100%; tempo de um passo
de treino com o carregador real (aumentação, 6 processos auxiliares). Só números.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from meter_reader.reader import CANDIDATES, Reader
from meter_reader.reader_text import FUNCTION_ALPHABET, TEXT_ALPHABET, greedy_decode
from training.reader_loss import reader_loss
from training.reader_torch import CLASSES, collate

SANITY_PER_CLASS = (11, 11, 10)   # digital, ciclométrico, número de série
OVERFIT_STEPS = 300


def sanity_items(items, seed: int = 0):
    """Lote fixo de recortes inteiros com texto, por classe."""
    rng = np.random.default_rng(seed)
    full = items[items["usa_texto"] & (items["recorte"] == "inteiro")]
    picks = [rng.choice(np.flatnonzero((full["classe"] == c).to_numpy()), n, replace=False)
             for c, n in zip(CLASSES, SANITY_PER_CLASS)]
    return full.iloc[np.concatenate(picks)]


def exact_match(logits: torch.Tensor, batch: dict, name: str, alphabet: str) -> float:
    """Fração dos itens da máscara com a sequência gulosa igual ao alvo."""
    lengths = batch[f"{name}_comprimento"].tolist()
    offsets = np.cumsum([0] + lengths)
    targets = batch[name].tolist()
    hits, total = 0, 0
    for i, use in enumerate(batch[f"usa_{name}"].tolist()):
        if not use:
            continue
        target = "".join(alphabet[k - 1] for k in targets[offsets[i]:offsets[i + 1]])
        hits += greedy_decode(logits[i].argmax(-1).tolist(), alphabet) == target
        total += 1
    return hits / total if total else float("nan")


def overfit(candidate: str, batch: dict, device: str, steps: int = OVERFIT_STEPS,
            pretrained: bool = True) -> dict:
    torch.manual_seed(0)
    model = Reader(candidate, pretrained=pretrained).to(device).train()
    x = batch["imagem"].to(device)
    with torch.no_grad():
        _, initial = reader_loss(model(x), batch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(steps):
        loss, parts = reader_loss(model(x), batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        out = model(x)
    return {"inicial": initial, "final": parts,
            "acerto_texto": exact_match(out["texto"].cpu(), batch, "texto", TEXT_ALPHABET),
            "acerto_funcao": exact_match(out["funcao"].cpu(), batch, "funcao",
                                         FUNCTION_ALPHABET)}


def length_match(logits: torch.Tensor, batch: dict) -> float:
    """Fração dos itens de texto em que a sequência gulosa tem o comprimento do alvo."""
    lengths = batch["texto_comprimento"].tolist()
    hits = [len(greedy_decode(logits[i].argmax(-1).tolist(), TEXT_ALPHABET)) == n
            for i, (n, use) in enumerate(zip(lengths, batch["usa_texto"].tolist())) if use]
    return sum(hits) / len(hits)


def curve(candidate: str, batch: dict, device: str, steps: int, every: int,
          lr: float = 1e-3) -> list[tuple]:
    """Sobreajuste com registro: (passo, perda do texto, acerto, comprimento certo)."""
    torch.manual_seed(0)
    model = Reader(candidate).to(device)
    x = batch["imagem"].to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    rows = []
    for step in range(1, steps + 1):
        model.train()
        loss, parts = reader_loss(model(x), batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % every == 0:
            model.eval()
            with torch.no_grad():
                out = model(x)["texto"].cpu()
            rows.append((step, parts["texto"], exact_match(out, batch, "texto", TEXT_ALPHABET),
                         length_match(out, batch)))
    return rows


def step_time(candidate: str, loader, device: str, warmup: int = 5, steps: int = 30) -> float:
    """Segundos por passo de treino (carregador + GPU), com precisão mista."""
    model = Reader(candidate).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler(device)
    it = iter(loader)
    for k in range(warmup + steps):
        if k == warmup:
            torch.cuda.synchronize()
            start = time.perf_counter()
        batch = next(it)
        with torch.autocast(device, dtype=torch.float16):
            out = model(batch["imagem"].to(device, non_blocking=True))
        loss, _ = reader_loss(out, batch)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / steps


def diagnose(fixed, device: str) -> None:
    """Perda do texto ao longo de 1.500 passos, com 4 e com 32 recortes (L1)."""
    for n in (4, len(fixed)):
        picks = [0, 11, 22, 1][:n] if n == 4 else range(n)   # as três classes no lote de 4
        batch = collate([fixed[i] for i in picks])
        print(f"\n== L1, {n} recortes ==")
        for step, loss, acc, length in curve("L1", batch, device, 1500, 250):
            print(f"passo {step:5d}: perda do texto {loss:.4f}; acerto {acc:.3f}; "
                  f"comprimento certo {length:.3f}")


def main() -> None:
    import sys

    from torch.utils.data import DataLoader

    from meter_reader.config import get_work_dir
    from training.reader_cache import CACHE_DIR, CACHE_NAME
    from training.reader_dataset import fold_masks, load_items
    from training.reader_labels import CROPS_SHA256
    from training.reader_torch import BalancedBatchSampler, ReaderDataset

    device = "cuda"
    cache = get_work_dir() / CACHE_DIR / CACHE_NAME
    items, _ = load_items()
    fixed = ReaderDataset(sanity_items(items), cache, CROPS_SHA256, train=False)
    batch = collate([fixed[i] for i in range(len(fixed))])
    if sys.argv[1:] == ["diagnostico"]:
        diagnose(fixed, device)
        return

    train = items[fold_masks(items, sorted(items["lote"].unique())[0])["treino"]]
    data = ReaderDataset(train, cache, CROPS_SHA256, train=True)
    sampler = BalancedBatchSampler(train, seed=0)
    steps_per_epoch = len(sampler)
    loader = DataLoader(data, batch_sampler=sampler, collate_fn=collate, num_workers=6,
                        persistent_workers=True, pin_memory=True)
    # o amostrador é finito por época; para a medição, basta repetir o carregador
    looped = (b for _ in range(10) for b in loader)

    for candidate in CANDIDATES:
        r = overfit(candidate, batch, device)
        n = sum(p.numel() for p in Reader(candidate, pretrained=False).parameters())
        print(f"\n== {candidate} ({n / 1e6:.2f} M parâmetros) ==")
        print("Perda inicial:", {k: round(v, 3) for k, v in r["inicial"].items()})
        print(f"Perda após {OVERFIT_STEPS} passos:",
              {k: round(v, 4) for k, v in r["final"].items()})
        print(f"Acerto no lote fixo: texto {r['acerto_texto']:.3f}; "
              f"função {r['acerto_funcao']:.3f}")
        seconds = step_time(candidate, looped, device)
        print(f"Passo de treino: {seconds * 1000:.0f} ms; época de {steps_per_epoch} passos: "
              f"{seconds * steps_per_epoch:.0f} s; 60 épocas: "
              f"{seconds * steps_per_epoch * 60 / 60:.0f} min")


if __name__ == "__main__":
    main()
