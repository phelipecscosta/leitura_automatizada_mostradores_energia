"""População-ouro da função (revisão R16 do protocolo do leitor; ET6c).

- `treinar`: monta o rótulo humano vigente de cada foto digital (o julgamento mais
  recente entre F-2, R12, Q-a e R16), grava-o na pasta de trabalho e retreina as
  cabeças pela receita da R11 com ele no lugar da função do professor (R12g).
- `medir`: regra (d) da R15 sobre a população-ouro (aprováveis com função humana),
  para a R12 e a R12g, sem projeção; escolhe pela R16 (d) e relata a sensibilidade
  com os ilegíveis e os não digitais contados como erro.
Só números.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R12G_HEADS = "L2-12+auto1-r12g"
GOLD_NAME = Path("experimentos") / "leitor" / "funcao_ouro.csv"
RESULT_NAME = Path("experimentos") / "leitor" / "funcao_r16.json"
R16_JUDGMENT = Path("experimentos") / "leitor" / "rotulagem_r16_ouro.csv"
QA_JUDGMENT = Path("experimentos") / "leitor" / "rotulagem_r15_aceitos.csv"
VALID = ("funcao", "vazio")
TIE = 0.01   # R16 (d): empate dentro de 1 ponto fica a R12


def gold_labels(f2: pd.DataFrame, judgments: list[pd.DataFrame]) -> pd.DataFrame:
    """Rótulo humano vigente por foto: a F-2 (só funções) e depois os julgamentos na
    ordem dada (o último vence). Colunas: pseudonimo, lote, status, funcao, fonte."""
    rows = {r["pseudonimo"]: {**r, "status": "vazio" if r["funcao"] == "" else "funcao",
                              "fonte": "f2"}
            for r in f2[["pseudonimo", "lote", "funcao"]].to_dict("records")}
    for name, j in judgments:
        for r in j[["pseudonimo", "lote", "status", "funcao"]].to_dict("records"):
            rows[r["pseudonimo"]] = {**r, "fonte": name}
    return pd.DataFrame(list(rows.values()),
                        columns=["pseudonimo", "lote", "status", "funcao", "fonte"])


def gold_items(items: pd.DataFrame, gold: pd.DataFrame, two_marks: dict) -> pd.DataFrame:
    """Itens do treino com a função humana no lugar da do professor: nas fotos com
    rótulo humano, a função do professor sai; as válidas entram pelo recorte da R12."""
    from training.reader_dataset import DIGITAL
    from training.reader_function_r12 import add_function_labels

    out = items.copy()
    mine = out["pseudonimo"].isin(gold["pseudonimo"]) & (out["classe"] == DIGITAL)
    out.loc[mine, "usa_funcao"] = False
    out.loc[mine, "funcao"] = ""
    return add_function_labels(out, gold[gold["status"].isin(VALID)], two_marks)


def decide(results: dict) -> str | None:
    """R16 (d): o candidato que cumpre com mais consumos mantidos; empate: R12."""
    ok = {k: v["escolhido"]["consumo_mantido"] for k, v in results.items() if v["escolhido"]}
    if not ok:
        return None
    best = max(ok, key=ok.get)
    return "R12" if "R12" in ok and ok[best] - ok["R12"] <= TIE else best


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def train() -> None:
    from meter_reader.config import get_work_dir
    from training.reader_cache import CACHE_DIR, CACHE_NAME
    from training.reader_function_r12 import R12_JUDGMENT_NAME, extended_items
    from training.reader_heads_train import CONFIG, HEADS_DIR, train_heads
    from training.reader_labels import CROPS_SHA256, labels_name
    from training.reader_train import META_NAME, RUNS_DIR, WEIGHTS_NAME
    from training.teacher import MODEL

    work_dir = get_work_dir()
    items = extended_items()
    f2 = items[(items["origem"] == "humano_f2") & items["usa_funcao"]
               & (items["classe"] == "display_digital")].drop_duplicates("pseudonimo")
    gold = gold_labels(f2, [("r12", _read(work_dir / R12_JUDGMENT_NAME)),
                            ("qa", _read(work_dir / QA_JUDGMENT)),
                            ("r16", _read(work_dir / R16_JUDGMENT))])
    gold.to_csv(work_dir / GOLD_NAME, index=False, encoding="utf-8", lineterminator="\n")
    print("Rótulo humano vigente:", gold.groupby(["fonte", "status"]).size().to_dict())
    raw = _read(work_dir / labels_name(MODEL))
    marks = dict(zip(raw["pseudonimo"], raw["duas_leituras"] == "True"))
    g_items = gold_items(items, gold, marks)
    print("Itens com função:", int(items["usa_funcao"].sum()), "->",
          int(g_items["usa_funcao"].sum()))

    cache = work_dir / CACHE_DIR / CACHE_NAME
    runs = sorted((work_dir / RUNS_DIR / CONFIG).glob(f"*/{META_NAME}"))
    for n, meta_path in enumerate(runs, 1):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        out_dir = work_dir / HEADS_DIR / R12G_HEADS / meta_path.parent.name
        if (out_dir / "meta.json").exists():
            print(f"[{n}/{len(runs)}] {meta_path.parent.name}: já concluído")
            continue
        m = train_heads(meta_path.parent / WEIGHTS_NAME, meta["candidato"], meta["cinza"],
                        g_items, meta["lote_avaliado"], meta["semente"], cache, CROPS_SHA256,
                        out_dir)
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
    from training.reader_function_r12 import R12_HEADS
    from training.reader_function_rule import CONSUMPTION, GRID, _outputs, choose, risk_curve
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
    missing = int((~pop["pseudonimo"].isin(gold.index)).sum())
    pop["status"] = pop["pseudonimo"].map(gold["status"]).fillna("sem_julgamento")
    pop["funcao"] = pop["pseudonimo"].map(gold["funcao"]).fillna("")
    out_mask = ~pop["status"].isin(VALID)
    pop["nc"] = [FUNCTION_CLASSES[function_class(c)] not in CONSUMPTION for c in pop["funcao"]]
    print(f"Aprováveis: {len(pop)}; sem julgamento: {missing}; fora da medida: "
          f"{pop.loc[out_mask, 'status'].value_counts().to_dict()} "
          f"({out_mask.mean():.1%})")

    cache = load_cache(work_dir / CACHE_DIR / CACHE_NAME, CROPS_SHA256)
    cons_idx = [FUNCTION_CLASSES.index(c) for c in CONSUMPTION]
    for name, root in (("R12", R12_HEADS), ("R12g", R12G_HEADS)):
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
                heads = Heads(dim).to(device)
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
    for name in ("R12", "R12g"):
        thresholds = np.unique(np.quantile(valid[name], np.linspace(0, 1, GRID)))
        curve = risk_curve(valid[name], np.ones(len(valid), bool), valid["nc"], zeros, zeros,
                           thresholds)
        chosen = choose(curve)
        t = chosen["limiar"] if chosen else float(curve.loc[curve["limite_superior"].idxmin(),
                                                             "limiar"])
        approved = pop[name] >= t
        excluded = int((approved & out_mask).sum())
        n_all = int(approved.sum())
        e_all = int((approved & (pop["nc"] | out_mask)).sum())
        results[name] = {"escolhido": chosen, "limiar_relatado": t,
                         "sensibilidade": {"aprovados": n_all, "erros": e_all,
                                           "limite_superior": cp_upper(e_all, n_all),
                                           "fora_da_medida_aprovados": excluded}}
        print(f"\n{name}: curva (a cada 20 quantis)")
        print(curve.iloc[::20].round(4).to_string(index=False))
        print("Escolhido (Y-f2):", "nenhum limiar" if chosen is None else
              {k: round(v, 4) if isinstance(v, float) else v for k, v in chosen.items()})
        sel = valid[(valid[name] >= t) & valid["nc"]]
        print(f"Erros em t = {t:.4f}, por código: {sel['funcao'].value_counts().to_dict()}")
        s = results[name]["sensibilidade"]
        print(f"Sensibilidade (ilegíveis e não digitais como erro) em t = {t:.4f}: "
              f"{s['erros']} em {s['aprovados']}; limite superior {s['limite_superior']:.4f}")
    winner = decide(results)
    results["vencedor"] = winner
    print("\nVencedor pela R16 (d):", winner or "nenhum (vale a R15-b, item b)")
    (work_dir / RESULT_NAME).write_text(json.dumps(results, indent=1), encoding="utf-8")


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    commands = {"treinar": train, "medir": measure}
    if command not in commands:
        raise SystemExit("Use: treinar | medir")
    commands[command]()


if __name__ == "__main__":
    main()
