"""F-x4: classificador só da função na faixa esquerda do recorte digital (revisões R15,
item e, e R15-c do protocolo do leitor; ET6c).

Entrada: faixa esquerda de 40% do recorte (inteiro, ou faixa de cima quando a D1
dispara), com o realce fixo, redimensionada para 96 x 192. Rede: as 12 primeiras
camadas da MobileNetV3-Small (ImageNet, passo de 8 x 4, ajuste fino completo) e a
cabeça da função da F-b1. Mesmos rótulos de função das cabeças R12; 3 dobras x 3
sementes; 30 épocas; época pela revocação média interna.

Comandos:
- `treinar`: os 9 treinos (retoma do ponto em que parou);
- `comparar`: regra da R11 contra as cabeças R12, no conjunto de avaliação de cada dobra;
- `regra`: a regra (d) da R15 com o consumo da F-x4, na população gravada pela medida
  da R15 (mesmas leituras e mesma D1), e os erros diretos abertos por origem e código,
  para a R12 e para a F-x4.
Só números.
"""

from __future__ import annotations

import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

X4_NAME = "F-x4"
LEFT_FRACTION = 0.40
X4_HEIGHT, X4_WIDTH = 96, 192      # R15-c
X4_EPOCHS = 30
X4_LR = 5e-4                       # taxa do L2-12
X4_BATCH = 128
X4_WORKERS = 6
COMPARISON_NAME = "funcao_x4_comparacao.json"
RULE_NAME = "funcao_x4_regra.json"


def left_band(crop):
    """Faixa esquerda do recorte, onde fica o código de função."""
    width, height = crop.size
    return crop.crop((0, 0, max(1, round(width * LEFT_FRACTION)), height))


def x4_tensor(crop):
    """Recorte (inteiro ou faixa de cima) -> tensor (3, 96, 192): realce, faixa esquerda,
    tamanho fixo e normalização do ImageNet."""
    from PIL import Image
    from torchvision.transforms import functional as TF

    from meter_reader.preprocess import IMAGENET_MEAN, IMAGENET_STD, enhance

    band = left_band(enhance(crop)).resize((X4_WIDTH, X4_HEIGHT), Image.Resampling.BILINEAR)
    return TF.normalize(TF.to_tensor(band), IMAGENET_MEAN, IMAGENET_STD)


def build_model(pretrained: bool = True):
    import torch
    from torch import nn

    from meter_reader.reader import _l2_backbone
    from training.reader_heads_train import Heads

    class FunctionX4(nn.Module):
        """Tronco L2-12 e a cabeça da função da F-b1; saída: logits das 7 classes."""

        def __init__(self):
            super().__init__()
            self.backbone, _ = _l2_backbone(12, pretrained)
            with torch.no_grad():
                _, c, h, _ = self.backbone(torch.zeros(1, 3, X4_HEIGHT, X4_WIDTH)).shape
            self.heads = Heads(c * h)

        def forward(self, x):
            f = self.backbone(x)
            b, c, h, t = f.shape
            func, _ = self.heads(f.permute(0, 3, 1, 2).reshape(b, t, c * h))
            return func

    return FunctionX4()


def direct_errors(pop: pd.DataFrame, score: str, t: float) -> pd.Series:
    """Erros diretos (aprovados de não consumo fora dos aceitos não julgados) num
    limiar, contados por origem do rótulo e código."""
    approved = pop["alta"].astype(bool) & (pop[score] >= t)
    sel = approved & pop["nc"].astype(bool) & ~pop["nao_julgado"].astype(bool)
    return pop.loc[sel].groupby(["origem", "funcao"]).size()


def _function_items(items: pd.DataFrame) -> pd.DataFrame:
    from training.reader_dataset import DIGITAL
    return items[items["usa_funcao"] & (items["classe"] == DIGITAL)].reset_index(drop=True)


def _dataset(items, cache_path, list_sha, train):
    import torch
    from PIL import Image
    from torch.utils.data import Dataset

    from meter_reader.reader_input import upper_band
    from training.reader_augment import augment
    from training.reader_cache import crop_key, load_cache
    from training.reader_dataset import UPPER
    from training.reader_heads_train import function_class

    class X4Dataset(Dataset):
        def __init__(self):
            self.items, self._cache = items.reset_index(drop=True), None
            self.y = self.items["funcao"].map(function_class).to_numpy()

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            if self._cache is None:
                self._cache = load_cache(cache_path, list_sha)
            r = self.items.iloc[i]
            crop = Image.fromarray(self._cache[crop_key(r["pseudonimo"], r["classe"])])
            if r["recorte"] == UPPER:
                crop = upper_band(crop)
            if train:
                crop = augment(crop, random.Random(torch.initial_seed() + i))
            return x4_tensor(crop), int(self.y[i]), i

    return X4Dataset()


def _predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    import torch
    probs, index = [], []
    model.eval()
    with torch.no_grad():
        for x, _, i in loader:
            with torch.autocast(device, dtype=torch.float16):
                logits = model(x.to(device))
            probs.append(logits.float().softmax(-1).cpu())
            index.append(i)
    return torch.cat(probs).numpy(), torch.cat(index).numpy()


def train_x4(items: pd.DataFrame, lot: str, seed: int, cache, list_sha: str, out_dir: Path,
             device: str = "cuda") -> dict:
    """Um treino da F-x4 (dobra que deixa `lot` de fora)."""
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, RandomSampler

    from training.reader_dataset import fold_masks
    from training.reader_heads_train import class_weights, macro_recall
    from training.reader_train import EVAL_BATCH, WEIGHT_DECAY, lr_factor

    torch.manual_seed(seed)
    np.random.seed(seed)
    f = _function_items(items)
    masks = fold_masks(f, lot)
    train_part = f[masks["treino"]].reset_index(drop=True)
    internal = f[masks["interno"] & (f["origem"] != "pseudo")].reset_index(drop=True)
    train_data = _dataset(train_part, cache, list_sha, train=True)
    internal_data = _dataset(internal, cache, list_sha, train=False)
    weights = class_weights(train_data.y).to(device)
    steps = max(1, len(train_part) // X4_BATCH)
    train_loader = DataLoader(train_data, batch_size=X4_BATCH,
                              sampler=RandomSampler(train_data, replacement=True,
                                                    num_samples=steps * X4_BATCH),
                              num_workers=X4_WORKERS, persistent_workers=True, pin_memory=True)
    internal_loader = DataLoader(internal_data, batch_size=EVAL_BATCH, num_workers=2,
                                 persistent_workers=True)

    model = build_model(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=X4_LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_factor(s, steps, steps * X4_EPOCHS))
    scaler = torch.amp.GradScaler(device)
    best, curve, start = (-1.0, 0, None), [], time.perf_counter()
    for epoch in range(1, X4_EPOCHS + 1):
        model.train()
        for x, y, _ in train_loader:
            with torch.autocast(device, dtype=torch.float16):
                logits = model(x.to(device, non_blocking=True))
            loss = F.cross_entropy(logits.float(), y.to(device), weight=weights)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        probs, index = _predict(model, internal_loader, device)
        mr = macro_recall(internal_data.y[index], probs.argmax(1))
        curve.append(round(mr, 4))
        if mr > best[0]:   # empate: fica a época mais cedo
            best = (mr, epoch, copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()}))

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best[2], out_dir / "pesos_x4.pt")
    meta = {"configuracao": X4_NAME, "lote_avaliado": lot, "semente": seed, "epoca": best[1],
            "revocacao_media_interna": round(best[0], 4), "curva": curve,
            "pesos_classes": [round(float(w), 3) for w in weights.cpu()],
            "segundos": round(time.perf_counter() - start)}
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    return meta


def predict_x4(weights_path: Path, crops: list, device: str = "cuda") -> np.ndarray:
    """Probabilidades das 7 classes para uma lista de recortes (PIL), na ordem."""
    import torch
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(crops), 256):
            x = torch.stack([x4_tensor(c) for c in crops[s:s + 256]]).to(device)
            out.append(model(x).float().softmax(-1).cpu().numpy())
    return np.concatenate(out)


DEV_LOTS = ("2026-05-20", "2026-05-21", "2026-05-22")


def _runs() -> list[tuple[str, int]]:
    """Os 9 treinos: lote deixado de fora x semente."""
    return [(lot, seed) for lot in DEV_LOTS for seed in (0, 1, 2)]


def train() -> None:
    from meter_reader.config import get_work_dir
    from training.reader_cache import CACHE_DIR, CACHE_NAME
    from training.reader_function_r12 import extended_items
    from training.reader_heads_train import HEADS_DIR
    from training.reader_labels import CROPS_SHA256

    work_dir = get_work_dir()
    items = extended_items()
    cache = work_dir / CACHE_DIR / CACHE_NAME
    runs = _runs()
    for n, (lot, seed) in enumerate(runs, 1):
        out_dir = work_dir / HEADS_DIR / X4_NAME / f"{lot}_s{seed}"
        if (out_dir / "meta.json").exists():
            print(f"[{n}/{len(runs)}] {lot}_s{seed}: já concluído")
            continue
        m = train_x4(items, lot, seed, cache, CROPS_SHA256, out_dir)
        print(f"[{n}/{len(runs)}] {lot}_s{seed}: época {m['epoca']} (revocação média interna "
              f"{m['revocacao_media_interna']:.3f}); {m['segundos'] / 60:.1f} min", flush=True)


def compare() -> None:
    from PIL import Image

    from meter_reader.config import get_work_dir
    from meter_reader.reader_input import upper_band
    from training.reader_cache import CACHE_DIR, CACHE_NAME, crop_key, load_cache
    from training.reader_dataset import UPPER, fold_masks
    from training.reader_function_r12 import R12_HEADS, extended_items, predict_function
    from training.reader_function_rule import CONSUMPTION, OUT_DIR
    from training.reader_heads_eval import boot_macro_recall, stratified_weights
    from training.reader_heads_train import CONFIG, FUNCTION_CLASSES, HEADS_DIR, function_class
    from training.reader_labels import CROPS_SHA256
    from training.reader_rule import BOOT_SEED, N_BOOT
    from training.reader_train import RUNS_DIR, WEIGHTS_NAME

    work_dir = get_work_dir()
    f = _function_items(extended_items())
    cache_path = work_dir / CACHE_DIR / CACHE_NAME
    cache = load_cache(cache_path, CROPS_SHA256)
    rows = []
    for lot, seed in _runs():
        run = f"{lot}_s{seed}"
        part = f[fold_masks(f, lot)["avaliacao"]].reset_index(drop=True)
        crops = [Image.fromarray(cache[crop_key(r["pseudonimo"], r["classe"])])
                 for r in part.to_dict("records")]
        crops = [upper_band(c) if cut == UPPER else c for c, cut in zip(crops, part["recorte"])]
        x4 = predict_x4(work_dir / HEADS_DIR / X4_NAME / run / "pesos_x4.pt", crops).argmax(1)
        r12 = predict_function(work_dir / RUNS_DIR / CONFIG / run / WEIGHTS_NAME,
                               work_dir / HEADS_DIR / R12_HEADS / run / "pesos_cabecas.pt",
                               "L2-12", False, part, cache_path, CROPS_SHA256).argmax(1)
        keys = part["pseudonimo"] + "|" + part["classe"] + "|" + part["recorte"]
        for k, y, a, b in zip(keys, part["funcao"].map(function_class), r12, x4):
            rows.append({"chave": k, "semente": seed, "y": y, "R12": a, "F-x4": b})
    df = pd.DataFrame(rows)
    wide = df.pivot(index="chave", columns="semente", values=["y", "R12", "F-x4"])
    y = wide["y"].iloc[:, 0].to_numpy().astype(int)
    hits = {m: wide[m].to_numpy().astype(int).T == y for m in ("R12", "F-x4")}
    w = stratified_weights(y, N_BOOT, np.random.default_rng(BOOT_SEED))
    boot = {m: boot_macro_recall(y, hits[m], w) for m in hits}
    point = {m: float(boot_macro_recall(y, hits[m], np.ones((1, len(y))))[0]) for m in hits}
    low, high = np.percentile(boot["F-x4"] - boot["R12"], [2.5, 97.5])
    adopted = bool(low > 0)
    print(f"Conjunto de avaliação: {len(y)} recortes com função")
    print("Revocação média por classe (média das sementes):",
          {m: round(v, 4) for m, v in point.items()},
          f"dif {point['F-x4'] - point['R12']:.4f} IC [{low:.4f}; {high:.4f}]; adotada: {adopted}")
    names = dict(enumerate(FUNCTION_CLASSES))
    cons = {FUNCTION_CLASSES.index(c) for c in CONSUMPTION}
    for m in ("R12", "F-x4"):
        read_cons = df[m].isin(cons)
        line = {names[c]: f"{float(read_cons[df['y'] == c].mean()):.3f} (n={int((df['y'] == c).sum())})"
                for c in range(len(FUNCTION_CLASSES)) if names[c] not in CONSUMPTION}
        print(f"  {m}: fração lida como consumo, por função verdadeira:", line)
    result = {"revocacao_media": point, "ic_diferenca": [float(low), float(high)],
              "adotada": adopted}
    (work_dir / OUT_DIR / COMPARISON_NAME).write_text(json.dumps(result, indent=1),
                                                      encoding="utf-8")


def rule() -> None:
    from PIL import Image

    from meter_reader.config import get_work_dir
    from meter_reader.reader_input import upper_band
    from training.group_dataset import LABEL_VERSIONS, load_group_photos
    from training.reader_cache import CACHE_DIR, CACHE_NAME, crop_key, load_cache
    from training.reader_dataset import DIGITAL
    from training.reader_function_r12 import extended_items
    from training.reader_function_rule import (CONSUMPTION, GRID, MIN_KEPT, OUT_DIR,
                                               POPULATION_NAME, _population, choose, risk_curve)
    from training.reader_heads_train import FUNCTION_CLASSES, HEADS_DIR
    from training.reader_labels import CROPS_SHA256
    from training.reader_pilot import csv_rows

    work_dir = get_work_dir()
    comparison = json.loads((work_dir / OUT_DIR / COMPARISON_NAME).read_text(encoding="utf-8"))
    judged = pd.read_csv(work_dir / OUT_DIR / "rotulagem_r15_aceitos.csv", dtype=str,
                         keep_default_na=False)
    expected = set(csv_rows(load_group_photos(LABEL_VERSIONS["r4"])))
    pop = _population(extended_items(), judged)
    pop = pop[pop["pseudonimo"].isin(expected)].reset_index(drop=True)
    saved = pd.read_csv(work_dir / OUT_DIR / POPULATION_NAME, dtype={"pseudonimo": str})
    if set(saved["pseudonimo"]) != set(pop["pseudonimo"]):
        raise SystemExit("A população não é a gravada pela medida da R15.")
    pop = pop.merge(saved[["pseudonimo", "alta", "d1", "consumo"]].rename(
        columns={"consumo": "consumo_r12"}), on="pseudonimo")

    cache = load_cache(work_dir / CACHE_DIR / CACHE_NAME, CROPS_SHA256)
    cons_idx = [FUNCTION_CLASSES.index(c) for c in CONSUMPTION]
    pop["consumo_x4"] = np.nan
    for lot in sorted(pop["lote"].unique()):
        sel = pop["lote"] == lot
        crops = []
        for r in pop[sel].to_dict("records"):
            crop = Image.fromarray(cache[crop_key(r["pseudonimo"], DIGITAL)])
            crops.append(upper_band(crop) if r["d1"] else crop)
        probs = np.mean([predict_x4(work_dir / HEADS_DIR / X4_NAME / f"{lot}_s{s}" / "pesos_x4.pt",
                                    crops) for s in (0, 1, 2)], axis=0)
        pop.loc[sel, "consumo_x4"] = probs[:, cons_idx].sum(1)

    pd.set_option("display.width", 160)
    result = {"adotada_pela_r11": comparison["adotada"]}
    for score in ("consumo_r12", "consumo_x4"):
        thresholds = np.unique(np.quantile(pop[score], np.linspace(0, 1, GRID)))
        curve = risk_curve(pop[score], pop["alta"], pop["nc"], pop["julgado"],
                           pop["nao_julgado"], thresholds)
        chosen = choose(curve)
        t = chosen["limiar"] if chosen else float(curve.loc[curve["limite_superior"].idxmin(),
                                                             "limiar"])
        kept = 0.0 if chosen is None else chosen["consumo_mantido"]
        result[score] = {"escolhido": chosen, "gatilho_g_a": kept < MIN_KEPT}
        print(f"\n{score}: curva (a cada 20 quantis)")
        print(curve.iloc[::20].round(4).to_string(index=False))
        print("Escolhido (Y-f2):", "nenhum limiar" if chosen is None else
              {k: round(v, 4) if isinstance(v, float) else v for k, v in chosen.items()})
        print(f"Erros diretos por origem e código em t = {t:.4f}:")
        print(direct_errors(pop, score, t).to_string())
    print("\nF-x4 adotada pela regra da R11:", comparison["adotada"],
          "(a regra (d) com a F-x4 só vale se adotada)")
    (work_dir / OUT_DIR / RULE_NAME).write_text(json.dumps(result, indent=1), encoding="utf-8")


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    commands = {"treinar": train, "comparar": compare, "regra": rule}
    if command not in commands:
        raise SystemExit("Use: treinar | comparar | regra")
    commands[command]()


if __name__ == "__main__":
    main()
