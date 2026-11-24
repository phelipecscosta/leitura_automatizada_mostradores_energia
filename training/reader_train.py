"""Treino fora da dobra do leitor (protocolo do leitor, seções 1.2, 1.12 e R1 a R3).

Um treino = configuração x lote deixado de fora x semente. Ao fim, grava na pasta de
trabalho: os pesos da melhor época (pela M-a na divisão interna), as predições do lote
avaliado (só a 0 grau, E99) e os metadados. O arquivo de metadados é gravado por
último e marca o treino como concluído: o comando pula o que já existe (retomada,
E62). Só números são exibidos.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from meter_reader.reader import CANDIDATES, Reader
from meter_reader.reader_text import TEXT_ALPHABET, greedy_decode
from training.reader_loss import reader_loss
from training.reader_torch import CLASSES, BalancedBatchSampler, ReaderDataset, collate

RUNS_DIR = Path("experimentos") / "leitor" / "treinos"
SEEDS = (0, 1, 2)
MAX_EPOCHS = 240                       # revisões R3 e R6
WARMUP_EPOCHS = 1
LEARNING_RATES = {"L1": 1e-3, "L2-9": 5e-4, "L2-12": 5e-4}
WEIGHT_DECAY = 1e-4
FINE_TUNE_LR_FACTOR = 0.1              # R8: ajuste fino com 1/10 da taxa do candidato
FINE_TUNE_EPOCHS = 60                  # R8: 1/4 das 240 épocas
TRAIN_WORKERS, EVAL_WORKERS, EVAL_BATCH = 6, 2, 128
META_NAME, WEIGHTS_NAME, PREDICTIONS_NAME = "meta.json", "pesos.pt", "predicoes.npz"


def lr_factor(step: int, warmup: int, total: int) -> float:
    """Aquecimento linear e depois cosseno até zero (seção 1.12)."""
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def text_matches(decoded: str, target: str, field_class: str) -> bool:
    """Acerto exato da M-a: leitura por número inteiro (Z-a); número de série por texto."""
    if field_class == "numero_serie":
        return decoded == target
    return decoded.isdigit() and target.isdigit() and int(decoded) == int(target)


def run_dir(work_dir: Path, config: str, lot: str, seed: int) -> Path:
    return work_dir / RUNS_DIR / config / f"{lot}_s{seed}"


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "desconhecido"


def evaluate_ma(model, loader, items: pd.DataFrame, device: str) -> float:
    """M-a: média simples, pelas três classes, do acerto exato do texto."""
    hits = {c: [] for c in CLASSES}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["imagem"].to(device))["texto"].float().cpu()
            for row, idx in enumerate(batch["indice"].tolist()):
                r = items.iloc[idx]
                decoded = greedy_decode(logits[row].argmax(-1).tolist(), TEXT_ALPHABET)
                hits[r["classe"]].append(text_matches(decoded, r["texto"], r["classe"]))
    return float(np.mean([np.mean(h) for h in hits.values() if h]))


def predict(model, items: pd.DataFrame, cache: Path, list_sha: str, gray: bool,
            device: str) -> dict[str, np.ndarray]:
    """Log-probabilidades (float16) do texto e da função, só a 0 grau (E99), nos itens
    com texto, e o logit de duas leituras em todos os itens. As chaves mantêm o sufixo
    "_0" para que as predições já gravadas sigam legíveis pelo mesmo código."""
    from meter_reader.reader_input import reader_tensor

    data = ReaderDataset(items, cache, list_sha, train=False, gray=gray)
    out = {"texto_0": [], "funcao_0": [], "duas": []}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(data), EVAL_BATCH):
            idx = range(start, min(start + EVAL_BATCH, len(data)))
            x = torch.stack([reader_tensor(data.image(i), gray=gray) for i in idx]).to(device)
            o = model(x)
            out["texto_0"].append(o["texto"].float().log_softmax(-1).cpu())
            out["funcao_0"].append(o["funcao"].float().log_softmax(-1).cpu())
            out["duas"].append(o["duas"].float().cpu())
    keep = items["usa_texto"].to_numpy(bool)
    arrays = {k: torch.cat(v).numpy() for k, v in out.items()}
    for k in ("texto_0", "funcao_0"):
        arrays[k] = arrays[k][keep].astype(np.float16)
    arrays["chave_texto"] = (items.loc[keep, "pseudonimo"] + "|" + items.loc[keep, "classe"]
                             + "|" + items.loc[keep, "recorte"]).to_numpy(str)
    arrays["chave"] = (items["pseudonimo"] + "|" + items["classe"] + "|"
                       + items["recorte"]).to_numpy(str)
    return arrays


def train_one(config: str, candidate: str, items: pd.DataFrame, lot: str, seed: int,
              cache: Path, list_sha: str, out_dir: Path, gray: bool = False,
              max_epochs: int = MAX_EPOCHS, device: str = "cuda",
              steps_per_epoch: int | None = None, init_from: Path | None = None,
              lr: float | None = None) -> dict:
    """Um treino completo; devolve os metadados gravados.

    `init_from`: pesos de partida (ajuste fino, R8). Nesse caso, o modelo de partida é
    avaliado antes do treino como "época 0" e pode ser o escolhido.
    """
    from training.reader_dataset import fold_masks

    torch.manual_seed(seed)
    np.random.seed(seed)
    masks = fold_masks(items, lot)
    train_items = items[masks["treino"]].reset_index(drop=True)
    # Os pseudo-rótulos (seção 1.10) nunca escolhem a época
    internal = items[masks["interno"] & items["usa_texto"]
                     & (items["origem"] != "pseudo")].reset_index(drop=True)
    evaluated = items[masks["avaliacao"]].reset_index(drop=True)

    sampler = BalancedBatchSampler(train_items, seed=seed, steps=steps_per_epoch)
    train_loader = DataLoader(ReaderDataset(train_items, cache, list_sha, train=True, gray=gray),
                              batch_sampler=sampler, collate_fn=collate,
                              num_workers=TRAIN_WORKERS, persistent_workers=True,
                              pin_memory=True)
    internal_loader = DataLoader(ReaderDataset(internal, cache, list_sha, train=False,
                                               gray=gray),
                                 batch_size=EVAL_BATCH, collate_fn=collate,
                                 num_workers=EVAL_WORKERS, persistent_workers=True)

    model = Reader(candidate, pretrained=init_from is None).to(device)
    if init_from is not None:
        model.load_state_dict(torch.load(init_from, map_location=device, weights_only=True))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr or LEARNING_RATES[candidate],
                                  weight_decay=WEIGHT_DECAY)
    total = len(sampler) * max_epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_factor(s, len(sampler) * WARMUP_EPOCHS, total))
    scaler = torch.amp.GradScaler(device)

    start, curve, best = time.perf_counter(), [], (-1.0, 0, None)
    if init_from is not None:   # época 0: o modelo de partida, sem ajuste
        ma = evaluate_ma(model, internal_loader, internal, device)
        best = (ma, 0, copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()}))
    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch in train_loader:
            with torch.autocast(device, dtype=torch.float16):
                out = model(batch["imagem"].to(device, non_blocking=True))
            loss, _ = reader_loss(out, batch)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        ma = evaluate_ma(model, internal_loader, internal, device)
        curve.append(round(ma, 5))
        if ma > best[0]:   # empate: fica a época mais cedo
            best = (ma, epoch, copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()}))
    train_seconds = time.perf_counter() - start

    model.load_state_dict(best[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best[2], out_dir / WEIGHTS_NAME)
    np.savez(out_dir / PREDICTIONS_NAME,
             **predict(model, evaluated, cache, list_sha, gray, device))
    meta = {"configuracao": config, "candidato": candidate, "lote_avaliado": lot,
            "semente": seed, "cinza": gray, "epocas_max": max_epochs,
            "epoca_escolhida": best[1], "ma_interna": round(best[0], 5), "curva_ma": curve,
            "itens_treino": int(len(train_items)), "passos_por_epoca": len(sampler),
            "itens_internos": int(len(internal)),
            "inicio": f"{init_from.parent.parent.name}/{init_from.parent.name}" if init_from else "",
            "taxa": lr or LEARNING_RATES[candidate],
            "itens_avaliados": int(len(evaluated)), "segundos_treino": round(train_seconds),
            "segundos_total": round(time.perf_counter() - start),
            "commit_produto": _git_commit(), "torch": torch.__version__,
            "lista_recortes_sha256": list_sha}
    (out_dir / META_NAME).write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
    return meta


def run_grid(config_of: dict, items: pd.DataFrame, max_epochs: int = MAX_EPOCHS,
             seeds=SEEDS, lots=None, items_for=None) -> None:
    """Roda todos os treinos de `config_of`, pulando os concluídos. Ordem:
    configuração, semente, lote.

    `config_of`: configuração -> candidato, ou -> {"candidato": ..., "cinza": ...,
    "passos_por_epoca": {lote: passos}, "inicio": configuração de partida (ajuste fino,
    mesma dobra e mesma semente), "taxa": ..., "max_epocas": ...}.
    `items_for`: função lote avaliado -> itens daquela dobra (pseudo-rótulos e
    ablações que mudam os itens de treino); sem ela, os mesmos itens em todas.
    """
    from meter_reader.config import get_work_dir
    from training.reader_cache import CACHE_DIR, CACHE_NAME
    from training.reader_labels import CROPS_SHA256

    work_dir = get_work_dir()
    cache = work_dir / CACHE_DIR / CACHE_NAME
    lots = lots or sorted(items["lote"].unique())
    jobs = [(c, s, lot) for c in config_of for s in seeds for lot in lots]
    for n, (config, seed, lot) in enumerate(jobs, 1):
        out_dir = run_dir(work_dir, config, lot, seed)
        if (out_dir / META_NAME).exists():
            print(f"[{n}/{len(jobs)}] {config} {lot} s{seed}: já concluído")
            continue
        spec = config_of[config]
        spec = spec if isinstance(spec, dict) else {"candidato": spec}
        fold_items = items_for(lot) if items_for else items
        epochs = spec.get("max_epocas", max_epochs)
        start = spec.get("inicio")
        init = run_dir(work_dir, start, lot, seed) / WEIGHTS_NAME if start else None
        if init is not None and not init.exists():
            raise SystemExit(f"Pesos de partida ausentes: {start} {lot} s{seed}.")
        meta = train_one(config, spec["candidato"], fold_items, lot, seed, cache, CROPS_SHA256,
                         out_dir, gray=spec.get("cinza", False), max_epochs=epochs,
                         steps_per_epoch=spec.get("passos_por_epoca", {}).get(lot),
                         init_from=init, lr=spec.get("taxa"))
        print(f"[{n}/{len(jobs)}] {config} {lot} s{seed}: época {meta['epoca_escolhida']} de "
              f"{epochs}; M-a interna {meta['ma_interna']:.3f}; "
              f"{meta['segundos_total'] / 60:.1f} min", flush=True)


def main() -> None:
    from training.reader_dataset import load_items

    parser = argparse.ArgumentParser(description="Treino fora da dobra do leitor.")
    parser.add_argument("comando", choices=["um", "candidatos"])
    parser.add_argument("--candidato", choices=CANDIDATES, default="L1")
    parser.add_argument("--semente", type=int, default=0)
    args = parser.parse_args()

    items, _ = load_items()
    if args.comando == "um":
        lot = sorted(items["lote"].unique())[0]
        run_grid({args.candidato: args.candidato}, items, seeds=(args.semente,), lots=[lot])
    else:
        run_grid({c: c for c in CANDIDATES}, items)


if __name__ == "__main__":
    main()
