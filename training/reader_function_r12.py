"""Revisão R12 do protocolo do leitor: a função dos digitais da fila F-2, rotulada por
humano, entra no treino e na avaliação das cabeças.

- `treinar`: retreina as duas cabeças (receita da R11) com os itens só de função da
  R12 somados aos de antes, na pasta "cabecas/L2-12+auto1-r12".
- `comparar`: refaz as predições da F-b1 e da cabeça nova no conjunto de avaliação
  ampliado e as compara pela regra da R11 (revocação média por classe). Relata o
  acerto com 03 = 003 e a fração de cada função que não é de consumo lida como
  consumo.

Só contagens e proporções.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from training.reader_heads_train import CONFIG, FUNCTION_CLASSES, HEADS_DIR, function_class

R12_JUDGMENT_NAME = Path("experimentos") / "leitor" / "rotulagem_r12_funcao.csv"
R12_HEADS = CONFIG + "-r12"
CONSUMPTION = ("03", "003")


def add_function_labels(items: pd.DataFrame, judged: pd.DataFrame,
                        two_marks: dict[str, bool]) -> pd.DataFrame:
    """Itens só de função da R12 (status "funcao" ou "vazio"). Recorte: a faixa de cima
    quando o professor marcou duas leituras; senão, o inteiro. Se o recorte já existe
    como item (marca de duas leituras), ganha a função; senão, vira item novo."""
    from training.reader_dataset import DIGITAL, FULL, UPPER

    out = items.copy()
    key = out["pseudonimo"] + "|" + out["classe"] + "|" + out["recorte"]
    position = dict(zip(key, out.index))
    new = []
    for r in judged[judged["status"].isin(["funcao", "vazio"])].to_dict("records"):
        cut = UPPER if two_marks.get(r["pseudonimo"], False) else FULL
        k = f"{r['pseudonimo']}|{DIGITAL}|{cut}"
        code = r["funcao"] if r["status"] == "funcao" else ""
        if k in position:
            out.loc[position[k], ["funcao", "usa_funcao"]] = [code, True]
        else:
            new.append({"pseudonimo": r["pseudonimo"], "classe": DIGITAL, "lote": r["lote"],
                        "recorte": cut, "texto": "", "funcao": code, "duas_leituras": False,
                        "usa_texto": False, "usa_funcao": True, "usa_duas": False,
                        "origem": "humano_r12"})
    return pd.concat([out, pd.DataFrame(new, columns=out.columns)], ignore_index=True)


def extended_items() -> pd.DataFrame:
    from meter_reader.config import get_work_dir
    from training.reader_dataset import load_items
    from training.reader_labels import labels_name
    from training.teacher import MODEL

    work_dir = get_work_dir()
    items, _ = load_items()
    judged = pd.read_csv(work_dir / R12_JUDGMENT_NAME, dtype=str, keep_default_na=False)
    raw = pd.read_csv(work_dir / labels_name(MODEL), dtype=str, keep_default_na=False)
    marks = dict(zip(raw["pseudonimo"], raw["duas_leituras"] == "True"))
    return add_function_labels(items, judged, marks)


def train() -> None:
    from meter_reader.config import get_work_dir
    from training.reader_cache import CACHE_DIR, CACHE_NAME
    from training.reader_heads_train import train_heads
    from training.reader_labels import CROPS_SHA256
    from training.reader_train import META_NAME, RUNS_DIR, WEIGHTS_NAME

    work_dir = get_work_dir()
    items = extended_items()
    print("Itens com função:", int(items["usa_funcao"].sum()),
          "| por classe:", items.loc[items["usa_funcao"], "funcao"].map(
              lambda c: FUNCTION_CLASSES[function_class(c)]).value_counts().to_dict())
    cache = work_dir / CACHE_DIR / CACHE_NAME
    runs = sorted((work_dir / RUNS_DIR / CONFIG).glob(f"*/{META_NAME}"))
    for n, meta_path in enumerate(runs, 1):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        out_dir = work_dir / HEADS_DIR / R12_HEADS / meta_path.parent.name
        if (out_dir / "meta.json").exists():
            print(f"[{n}/{len(runs)}] {meta_path.parent.name}: já concluído")
            continue
        m = train_heads(meta_path.parent / WEIGHTS_NAME, meta["candidato"], meta["cinza"], items,
                        meta["lote_avaliado"], meta["semente"], cache, CROPS_SHA256, out_dir)
        print(f"[{n}/{len(runs)}] {meta_path.parent.name}: função época {m['epoca_funcao']} "
              f"(revocação média {m['revocacao_media_interna']:.3f}); duas leituras época "
              f"{m['epoca_duas']} (AP {m['ap_interno']:.3f}); {m['segundos'] / 60:.1f} min",
              flush=True)


def predict_function(trunk_path: Path, heads_path: Path, candidate: str, gray: bool,
                     part: pd.DataFrame, cache: Path, list_sha: str,
                     device: str = "cuda") -> np.ndarray:
    """Probabilidades das 7 classes da função por um par (tronco, cabeças), na ordem de
    `part`."""
    import torch
    from torch.utils.data import DataLoader

    from meter_reader.reader import Reader
    from training.reader_heads_train import Heads, _outputs, trunk_features
    from training.reader_torch import ReaderDataset, collate
    from training.reader_train import EVAL_BATCH

    model = Reader(candidate, pretrained=False).to(device)
    model.load_state_dict(torch.load(trunk_path, map_location=device, weights_only=True))
    model.eval()
    with torch.no_grad():
        dim = trunk_features(model, torch.zeros(1, 3, 48, 192, device=device)).shape[-1]
    heads = Heads(dim).to(device)
    heads.load_state_dict(torch.load(heads_path, map_location=device, weights_only=True))
    loader = DataLoader(ReaderDataset(part, cache, list_sha, train=False, gray=gray),
                        batch_size=EVAL_BATCH, collate_fn=collate, num_workers=2)
    probs, _, index = _outputs(model, heads, loader, device)
    out = np.empty_like(probs)
    out[index] = probs
    return out

def compare() -> None:
    from meter_reader.config import get_work_dir
    from training.reader_cache import CACHE_DIR, CACHE_NAME
    from training.reader_dataset import fold_masks
    from training.reader_heads_eval import boot_macro_recall, stratified_weights
    from training.reader_labels import CROPS_SHA256
    from training.reader_rule import BOOT_SEED, N_BOOT
    from training.reader_train import META_NAME, RUNS_DIR, WEIGHTS_NAME

    work_dir = get_work_dir()
    items = extended_items()
    cache = work_dir / CACHE_DIR / CACHE_NAME
    rows = []
    for meta_path in sorted((work_dir / RUNS_DIR / CONFIG).glob(f"*/{META_NAME}")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        lot = meta["lote_avaliado"]
        part = items[fold_masks(items, lot)["avaliacao"] & items["usa_funcao"]].reset_index(drop=True)
        y = part["funcao"].map(function_class).to_numpy()
        preds = {}
        for name, root in (("F-b1", CONFIG), ("R12", R12_HEADS)):
            heads = work_dir / HEADS_DIR / root / meta_path.parent.name / "pesos_cabecas.pt"
            preds[name] = predict_function(meta_path.parent / WEIGHTS_NAME, heads,
                                           meta["candidato"], meta["cinza"], part, cache,
                                           CROPS_SHA256).argmax(1)
        keys = part["pseudonimo"] + "|" + part["classe"] + "|" + part["recorte"]
        for k, yy, a, b, o in zip(keys, y, preds["F-b1"], preds["R12"], part["origem"]):
            rows.append({"chave": k, "semente": meta["semente"], "y": yy, "F-b1": a, "R12": b,
                         "origem": o})
    df = pd.DataFrame(rows)

    wide = df.pivot(index="chave", columns="semente", values=["y", "F-b1", "R12"])
    y = wide["y"].iloc[:, 0].to_numpy()
    hits = {m: wide[m].to_numpy().T == y for m in ("F-b1", "R12")}
    w = stratified_weights(y, N_BOOT, np.random.default_rng(BOOT_SEED))
    boot = {m: boot_macro_recall(y, hits[m], w) for m in hits}
    point = {m: float(boot_macro_recall(y, hits[m], np.ones((1, len(y))))[0]) for m in hits}
    low, high = np.percentile(boot["R12"] - boot["F-b1"], [2.5, 97.5])
    print(f"Conjunto de avaliação ampliado: {len(y)} recortes com função "
          f"({pd.Series(y).map(dict(enumerate(FUNCTION_CLASSES))).value_counts().to_dict()})")
    print("Revocação média por classe (média das sementes):",
          {m: round(v, 4) for m, v in point.items()},
          f"dif {point['R12'] - point['F-b1']:.4f} IC [{low:.4f}; {high:.4f}]",
          "adotada:", bool(low > 0))

    names = dict(enumerate(FUNCTION_CLASSES))
    cons = [FUNCTION_CLASSES.index(c) for c in CONSUMPTION]
    for m in ("F-b1", "R12"):
        true_cat = df["y"].map(lambda c: "consumo" if c in cons else names[c])
        pred_cat = df[m].map(lambda c: "consumo" if c in cons else names[c])
        print(f"\n{m}: acerto com 03 = 003: {float((true_cat == pred_cat).mean()):.4f}")
        lines = {}
        for code in ("103", "24", "55", "outro", "vazio"):
            sel = true_cat == code
            lines[code] = f"{float((pred_cat[sel] == 'consumo').mean()):.3f} (n={int(sel.sum())})"
        print("  fração lida como consumo, por função verdadeira:", lines)
        print(pd.crosstab(true_cat, pred_cat).to_string())


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "treinar":
        train()
    elif command == "comparar":
        compare()
    else:
        raise SystemExit("Use: treinar | comparar")


if __name__ == "__main__":
    main()
