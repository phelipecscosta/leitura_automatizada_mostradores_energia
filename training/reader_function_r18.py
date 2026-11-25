"""Revisões R17 e R18 do protocolo do leitor (ET6c): classe "ilegível" na cabeça da
função (R12i) e escolha entre R12 e R12i com a meta de 1% e o limiar T-2.

- `treinar`: as 9 cabeças R12i (receita da R11, 8 classes).
- `medir`: na população-ouro da R16, o limiar T-2 de cada candidato, a fração dos
  ilegíveis aprovados, a sensibilidade e a adoção pela R18 (c).
Só números.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R12I_HEADS = "L2-12+auto1-r12i"
ILLEGIBLE = "ILEGIVEL"            # código interno da oitava classe
N_CLASSES_I = 8
MAX_RISK = 0.01                   # R17 (a) e (b)
MIN_KEPT = 0.50                   # R18 (c) (iii)
RESULT_NAME = Path("experimentos") / "leitor" / "funcao_r18.json"


def class_fn_i(code: str) -> int:
    """Classe da função com a oitava classe (ilegível) da R12i."""
    from training.reader_heads_train import function_class
    return N_CLASSES_I - 1 if code == ILLEGIBLE else function_class(code)


def illegible_items(items: pd.DataFrame, gold: pd.DataFrame, two_marks: dict) -> pd.DataFrame:
    """Itens da R12 com a única mudança da R18 (a): as fotos julgadas com função ilegível
    ganham a classe ilegível no lugar da função do professor; as não digitais perdem o
    rótulo de função."""
    from training.reader_dataset import DIGITAL
    from training.reader_function_r12 import add_function_labels

    out = items.copy()
    bad = gold[gold["status"].isin(["ilegivel", "nao_digital"])]
    mine = out["pseudonimo"].isin(bad["pseudonimo"]) & (out["classe"] == DIGITAL)
    out.loc[mine, "usa_funcao"] = False
    out.loc[mine, "funcao"] = ""
    ill = gold[gold["status"] == "ilegivel"].assign(status="funcao", funcao=ILLEGIBLE)
    return add_function_labels(out, ill, two_marks)


def choose_t2(curve: pd.DataFrame, max_risk: float = MAX_RISK) -> dict | None:
    """R17 (c): entre os limiares que cumprem a meta, o de menor limite superior; em
    empate, o menor limiar. None se nenhum cumpre."""
    ok = curve[(curve["aprovados"] > 0) & (curve["limite_superior"] <= max_risk)]
    if ok.empty:
        return None
    row = ok.sort_values(["limite_superior", "limiar"]).iloc[0]
    return {"limiar": float(row["limiar"]), "aprovados": int(row["aprovados"]),
            "erros": int(row["erros"]), "limite_superior": float(row["limite_superior"]),
            "consumo_mantido": float(row["consumo_mantido"])}


def adopt(r12: dict, r12i: dict) -> bool:
    """R18 (c): a R12i substitui a R12 se cumprir a meta, aprovar menos ilegíveis que a
    R12 e mantiver pelo menos 50% dos consumos."""
    c = r12i["escolhido"]
    if c is None:
        return False
    if r12["escolhido"] is not None and \
            r12i["ilegiveis_aprovados"] >= r12["ilegiveis_aprovados"]:
        return False
    return c["consumo_mantido"] >= MIN_KEPT


def train() -> None:
    from meter_reader.config import get_work_dir
    from training.reader_cache import CACHE_DIR, CACHE_NAME
    from training.reader_function_gold import GOLD_NAME, _read
    from training.reader_function_r12 import extended_items
    from training.reader_heads_train import CONFIG, HEADS_DIR, train_heads
    from training.reader_labels import CROPS_SHA256, labels_name
    from training.reader_train import META_NAME, RUNS_DIR, WEIGHTS_NAME
    from training.teacher import MODEL

    work_dir = get_work_dir()
    items = extended_items()
    gold = _read(work_dir / GOLD_NAME)
    raw = _read(work_dir / labels_name(MODEL))
    marks = dict(zip(raw["pseudonimo"], raw["duas_leituras"] == "True"))
    i_items = illegible_items(items, gold, marks)
    f = i_items[i_items["usa_funcao"]]
    print("Itens com função:", int(items["usa_funcao"].sum()), "->", len(f),
          "| ilegíveis:", int((f["funcao"] == ILLEGIBLE).sum()))

    cache = work_dir / CACHE_DIR / CACHE_NAME
    runs = sorted((work_dir / RUNS_DIR / CONFIG).glob(f"*/{META_NAME}"))
    for n, meta_path in enumerate(runs, 1):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        out_dir = work_dir / HEADS_DIR / R12I_HEADS / meta_path.parent.name
        if (out_dir / "meta.json").exists():
            print(f"[{n}/{len(runs)}] {meta_path.parent.name}: já concluído")
            continue
        m = train_heads(meta_path.parent / WEIGHTS_NAME, meta["candidato"], meta["cinza"],
                        i_items, meta["lote_avaliado"], meta["semente"], cache, CROPS_SHA256,
                        out_dir, n_classes=N_CLASSES_I, class_fn=class_fn_i)
        print(f"[{n}/{len(runs)}] {meta_path.parent.name}: função época {m['epoca_funcao']} "
              f"(revocação média {m['revocacao_media_interna']:.3f}); "
              f"{m['segundos'] / 60:.1f} min", flush=True)


def measure(device: str = "cuda") -> None:
    import torch
    from PIL import Image

    from meter_reader.config import get_work_dir
    from meter_reader.reader import Reader
    from meter_reader.reader_input import upper_band
    from training.reader_cache import CACHE_DIR, CACHE_NAME, crop_key, load_cache
    from training.reader_dataset import DIGITAL
    from training.reader_function_gold import GOLD_NAME, VALID, _read
    from training.reader_function_r12 import R12_HEADS
    from training.reader_function_rule import CONSUMPTION, GRID, _outputs, risk_curve
    from training.reader_gold_cost import TABLE_NAME
    from training.reader_heads_train import (CONFIG, FUNCTION_CLASSES, HEADS_DIR, Heads,
                                             function_class, trunk_features)
    from training.reader_labels import CROPS_SHA256
    from training.reader_metrics import cp_upper
    from training.reader_train import RUNS_DIR, WEIGHTS_NAME

    work_dir = get_work_dir()
    table = pd.read_csv(work_dir / TABLE_NAME, dtype={"pseudonimo": str, "lote": str})
    gold = _read(work_dir / GOLD_NAME).set_index("pseudonimo")
    pop = table[table["alta"].astype(bool)].reset_index(drop=True)
    pop["status"] = pop["pseudonimo"].map(gold["status"]).fillna("sem_julgamento")
    pop["funcao"] = pop["pseudonimo"].map(gold["funcao"]).fillna("")
    out_mask = ~pop["status"].isin(VALID)
    ill_mask = pop["status"] == "ilegivel"
    pop["nc"] = [FUNCTION_CLASSES[function_class(c)] not in CONSUMPTION for c in pop["funcao"]]

    cache = load_cache(work_dir / CACHE_DIR / CACHE_NAME, CROPS_SHA256)
    cons_idx = [FUNCTION_CLASSES.index(c) for c in CONSUMPTION]
    candidates = (("R12", R12_HEADS, len(FUNCTION_CLASSES)), ("R12i", R12I_HEADS, N_CLASSES_I))
    for name, root, n_classes in candidates:
        pop[name] = np.nan
        for lot in sorted(pop["lote"].unique()):
            sel = pop["lote"] == lot
            crops = []
            for r in pop[sel].to_dict("records"):
                crop = Image.fromarray(cache[crop_key(r["pseudonimo"], DIGITAL)])
                crops.append(upper_band(crop) if r["d1"] else crop)
            probs = []
            for seed in (0, 1, 2):
                run = f"{lot}_s{seed}"
                model = Reader("L2-12", pretrained=False).to(device)
                model.load_state_dict(torch.load(work_dir / RUNS_DIR / CONFIG / run / WEIGHTS_NAME,
                                                 map_location=device, weights_only=True))
                model.eval()
                with torch.no_grad():
                    dim = trunk_features(model, torch.zeros(1, 3, 48, 192, device=device)).shape[-1]
                heads = Heads(dim, n_classes=n_classes).to(device)
                heads.load_state_dict(torch.load(work_dir / HEADS_DIR / root / run
                                                 / "pesos_cabecas.pt", map_location=device,
                                                 weights_only=True))
                heads.eval()
                probs.append(_outputs(model, heads, crops, device)["funcao"])
            pop.loc[sel, name] = np.mean(probs, axis=0)[:, cons_idx].sum(1)

    pd.set_option("display.width", 160)
    valid = pop[~out_mask].reset_index(drop=True)
    zeros = np.zeros(len(valid), bool)
    results = {}
    for name, _, _ in candidates:
        thresholds = np.unique(np.quantile(valid[name], np.linspace(0, 1, GRID)))
        curve = risk_curve(valid[name], np.ones(len(valid), bool), valid["nc"], zeros, zeros,
                           thresholds)
        chosen = choose_t2(curve)
        t = chosen["limiar"] if chosen else float(curve.loc[curve["limite_superior"].idxmin(),
                                                             "limiar"])
        approved = pop[name] >= t
        n_all, e_all = int(approved.sum()), int((approved & (pop["nc"] | out_mask)).sum())
        results[name] = {"escolhido": chosen, "limiar_relatado": t,
                         "ilegiveis_aprovados": float(approved[ill_mask].mean()),
                         "sensibilidade": {"aprovados": n_all, "erros": e_all,
                                           "limite_superior": cp_upper(e_all, n_all)}}
        print(f"\n{name}: curva (a cada 20 quantis)")
        print(curve.iloc[::20].round(4).to_string(index=False))
        print("Escolhido (R17, T-2):", "nenhum limiar" if chosen is None else
              {k: round(v, 4) if isinstance(v, float) else v for k, v in chosen.items()})
        sel = valid[(valid[name] >= t) & valid["nc"]]
        print(f"Erros em t = {t:.4f}, por código: {sel['funcao'].value_counts().to_dict()}")
        s = results[name]["sensibilidade"]
        print(f"Ilegíveis aprovados: {results[name]['ilegiveis_aprovados']:.1%} de "
              f"{int(ill_mask.sum())}; sensibilidade: {s['erros']} em {s['aprovados']}, "
              f"limite superior {s['limite_superior']:.4f}")
    adopted = adopt(results["R12"], results["R12i"])
    results["r12i_adotada"] = adopted
    print("\nR12i adotada pela R18 (c):", adopted)
    (work_dir / RESULT_NAME).write_text(json.dumps(results, indent=1), encoding="utf-8")


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    commands = {"treinar": train, "medir": measure}
    if command not in commands:
        raise SystemExit("Use: treinar | medir")
    commands[command]()


if __name__ == "__main__":
    main()
