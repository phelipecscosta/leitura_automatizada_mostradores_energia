"""Treino das cabeças da função (F-b1) e de duas leituras (D1) sobre o tronco
congelado da configuração final (protocolo do leitor, revisão R11).

As duas cabeças recebem os atributos do tronco desligados (R5): não mudam a leitura
do texto. Por isso, são treinadas sobre o tronco congelado de cada modelo final
(mesma dobra e mesma semente), sem retreinar o resto:

- função como classificação em 7 classes, pelo prefixo do código (vazio, 03, 003,
  103, 24, 55, outro), com perda ponderada pelo inverso da frequência das classes;
- duas leituras com o peso dos positivos igual a negativos ÷ positivos, a partir da
  média, no tempo, dos atributos do tronco (que guardam a altura, onde aparecem as
  duas linhas).

Época escolhida na divisão interna, separadamente para cada cabeça: revocação média
por classe (função) e AP (duas leituras). Grava os pesos das cabeças e as predições
do lote avaliado na pasta de trabalho. Só números.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, recall_score
from torch.utils.data import DataLoader, RandomSampler
# As cabeças moram no produto (T6c.5); os nomes seguem importáveis daqui
from meter_reader.reader_heads import FUNCTION_CLASSES, Heads, trunk_features  # noqa: F401

CONFIG = "L2-12+auto1"
HEADS_DIR = Path("experimentos") / "leitor" / "cabecas"
HEAD_EPOCHS = 30
HEAD_LR = 1e-3
HEAD_BATCH = 128
HEAD_WORKERS = 6


def function_class(code: str) -> int:
    """Classe do código de função pelo prefixo (YY(Y)XX conta pelo YY(Y))."""
    if code == "":
        return 0
    for k, prefix in ((3, "103"), (2, "003"), (1, "03"), (4, "24"), (5, "55")):
        if code.startswith(prefix):
            return k
    return len(FUNCTION_CLASSES) - 1


def class_weights(classes: np.ndarray, n_classes: int = len(FUNCTION_CLASSES)) -> torch.Tensor:
    """Inverso da frequência ("balanced"): cada classe presente pesa o mesmo na perda,
    e o peso médio por exemplo é 1; classe ausente tem peso 0."""
    counts = np.bincount(classes, minlength=n_classes).astype(float)
    present = counts > 0
    weights = np.zeros(n_classes)
    weights[present] = counts[present].sum() / (present.sum() * counts[present])
    return torch.tensor(weights, dtype=torch.float32)

def macro_recall(y: np.ndarray, pred: np.ndarray) -> float:
    """Revocação média pelas classes presentes em `y`."""
    return float(recall_score(y, pred, labels=np.unique(y), average="macro", zero_division=0))


def _targets(items: pd.DataFrame, class_fn=function_class) -> dict[str, np.ndarray]:
    return {"classe_funcao": items["funcao"].map(class_fn).to_numpy(),
            "usa_funcao": items["usa_funcao"].to_numpy(bool),
            "duas": items["duas_leituras"].astype(bool).to_numpy(),
            "usa_duas": items["usa_duas"].to_numpy(bool)}


def _outputs(model, heads, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Probabilidades da função, logits de duas leituras e índices, sem aumentação."""
    probs, logits, index = [], [], []
    heads.eval()
    with torch.no_grad():
        for batch in loader:
            with torch.autocast(device, dtype=torch.float16):
                f = trunk_features(model, batch["imagem"].to(device))
            func, two = heads(f.float())
            probs.append(func.softmax(-1).cpu())
            logits.append(two.cpu())
            index.append(batch["indice"])
    return torch.cat(probs).numpy(), torch.cat(logits).numpy(), torch.cat(index).numpy()


def train_heads(weights_path: Path, candidate: str, gray: bool, items: pd.DataFrame, lot: str,
                seed: int, cache: Path, list_sha: str, out_dir: Path,
                device: str = "cuda", n_classes: int = len(FUNCTION_CLASSES),
                class_fn=function_class) -> dict:
    """Um treino das duas cabeças sobre o tronco congelado de um modelo final. `n_classes`
    e `class_fn` permitem uma classe a mais na função (R18), sem mudar o padrão."""
    from meter_reader.reader import Reader
    from meter_reader.reader_input import READER_HEIGHT, READER_WIDTH
    from training.reader_dataset import fold_masks
    from training.reader_torch import ReaderDataset, collate
    from training.reader_train import EVAL_BATCH, lr_factor

    torch.manual_seed(seed)
    model = Reader(candidate, pretrained=False).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        feat_dim = trunk_features(model, torch.zeros(1, 3, READER_HEIGHT, READER_WIDTH,
                                                     device=device)).shape[-1]
    heads = Heads(feat_dim, n_classes=n_classes).to(device)

    used = items[items["usa_funcao"] | items["usa_duas"]]
    masks = fold_masks(used, lot)
    parts = {k: used[masks[k]].reset_index(drop=True) for k in ("treino", "interno", "avaliacao")}
    parts["interno"] = parts["interno"][parts["interno"]["origem"] != "pseudo"].reset_index(drop=True)
    tg = {k: _targets(v, class_fn) for k, v in parts.items()}

    t = tg["treino"]
    f_weights = class_weights(t["classe_funcao"][t["usa_funcao"]], n_classes).to(device)
    pos = t["duas"][t["usa_duas"]]
    pos_weight = torch.tensor((~pos).sum() / max(1, pos.sum()), device=device)

    steps = max(1, len(parts["treino"]) // HEAD_BATCH)
    train_data = ReaderDataset(parts["treino"], cache, list_sha, train=True, gray=gray, seed=seed)
    train_loader = DataLoader(train_data, batch_size=HEAD_BATCH, collate_fn=collate,
                              sampler=RandomSampler(train_data, replacement=True,
                                                    num_samples=steps * HEAD_BATCH),
                              num_workers=HEAD_WORKERS, persistent_workers=True,
                              pin_memory=True)

    def eval_loader(name):
        return DataLoader(ReaderDataset(parts[name], cache, list_sha, train=False, gray=gray),
                          batch_size=EVAL_BATCH, collate_fn=collate, num_workers=2)

    internal_loader = eval_loader("interno")
    optimizer = torch.optim.AdamW(heads.parameters(), lr=HEAD_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_factor(s, steps, steps * HEAD_EPOCHS))
    ti = {k: torch.from_numpy(np.array(v)) for k, v in t.items()}   # cópia gravável
    i = tg["interno"]
    best_f, best_t, curve = (-1.0, 0, None), (-1.0, 0, None), []
    start = time.perf_counter()
    for epoch in range(1, HEAD_EPOCHS + 1):
        heads.train()
        for batch in train_loader:
            idx = batch["indice"]
            with torch.no_grad(), torch.autocast(device, dtype=torch.float16):
                f = trunk_features(model, batch["imagem"].to(device, non_blocking=True))
            func, two = heads(f.float())
            use_f, use_t = ti["usa_funcao"][idx].to(device), ti["usa_duas"][idx].to(device)
            loss = func.sum() * 0
            if use_f.any():
                loss = loss + F.cross_entropy(func[use_f], ti["classe_funcao"][idx].to(device)[use_f],
                                              weight=f_weights)
            if use_t.any():
                loss = loss + F.binary_cross_entropy_with_logits(
                    two[use_t], ti["duas"][idx].to(device)[use_t].float(), pos_weight=pos_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
        probs, logits, index = _outputs(model, heads, internal_loader, device)
        uf, ut = i["usa_funcao"][index], i["usa_duas"][index]
        mr = macro_recall(i["classe_funcao"][index][uf], probs[uf].argmax(1)) if uf.any() else 0.0
        y = i["duas"][index][ut]
        ap = float(average_precision_score(y, logits[ut])) if y.any() else 0.0
        curve.append({"epoca": epoch, "revocacao_media_funcao": round(mr, 4), "ap_duas": round(ap, 4)})
        state = {k: v.cpu().clone() for k, v in heads.state_dict().items()}
        if mr > best_f[0]:   # empate: fica a época mais cedo
            best_f = (mr, epoch, copy.deepcopy(state))
        if ap > best_t[0]:
            best_t = (ap, epoch, copy.deepcopy(state))

    # Cada cabeça com os pesos da sua melhor época
    final = {k: (best_t[2][k] if k.startswith("two.") else best_f[2][k]) for k in best_f[2]}
    heads.load_state_dict(final)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(final, out_dir / "pesos_cabecas.pt")
    if len(parts["avaliacao"]):   # as cabeças finais (T6c.4) não têm lote avaliado
        probs, logits, index = _outputs(model, heads, eval_loader("avaliacao"), device)
        ev = parts["avaliacao"].iloc[index]
        np.savez(out_dir / "predicoes_cabecas.npz", funcao=probs.astype(np.float32),
                 duas=logits.astype(np.float32),
                 chave=(ev["pseudonimo"] + "|" + ev["classe"] + "|" + ev["recorte"]).to_numpy(str))
    meta = {"configuracao": CONFIG, "lote_avaliado": lot, "semente": seed,
            "epoca_funcao": best_f[1], "revocacao_media_interna": round(best_f[0], 4),
            "epoca_duas": best_t[1], "ap_interno": round(best_t[0], 4), "curva": curve,
            "pesos_classes": [round(float(w), 3) for w in f_weights.cpu()],
            "peso_positivos": round(float(pos_weight), 2),
            "segundos": round(time.perf_counter() - start)}
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    return meta


def main() -> None:
    from meter_reader.config import get_work_dir
    from training.reader_cache import CACHE_DIR, CACHE_NAME
    from training.reader_dataset import load_items
    from training.reader_labels import CROPS_SHA256
    from training.reader_train import META_NAME, RUNS_DIR, WEIGHTS_NAME

    work_dir = get_work_dir()
    items, _ = load_items()
    cache = work_dir / CACHE_DIR / CACHE_NAME
    runs = sorted((work_dir / RUNS_DIR / CONFIG).glob(f"*/{META_NAME}"))
    for n, meta_path in enumerate(runs, 1):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        out_dir = work_dir / HEADS_DIR / CONFIG / meta_path.parent.name
        if (out_dir / "meta.json").exists():
            print(f"[{n}/{len(runs)}] {meta_path.parent.name}: já concluído")
            continue
        m = train_heads(meta_path.parent / WEIGHTS_NAME, meta["candidato"], meta["cinza"], items,
                        meta["lote_avaliado"], meta["semente"], cache, CROPS_SHA256, out_dir)
        print(f"[{n}/{len(runs)}] {meta_path.parent.name}: função época {m['epoca_funcao']} "
              f"(revocação média {m['revocacao_media_interna']:.3f}); duas leituras época "
              f"{m['epoca_duas']} (AP {m['ap_interno']:.3f}); {m['segundos'] / 60:.1f} min",
              flush=True)


if __name__ == "__main__":
    main()
