"""Sensibilidade dos limiares do leitor ao TF32 (revisão R19 do protocolo do leitor;
ET6c, T6c.5, X-b).

Com o TF32 desligado e os mesmos modelos de 2 lotes (fora da dobra):
(i) a população dos aprováveis da R16 é refeita (leitura do comitê igual ao CSV, com
    os limiares da D1 por lote da R16) e a medida da função é repetida no limiar fixo
    da R17;
(ii) o limiar inferior é recalculado pela regra da R10 na amostra L-a, com a leitura
    do produto (D1 em 3,063, como na T6c.3).
Aplica o critério da R19 (c) e grava `funcao_r19.json` na pasta de trabalho, sem
sobrescrever. Só números.

Uso, a partir da raiz do repositório do produto:
    python -m training.reader_tf32_check
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

OUT_NAME = Path("experimentos") / "leitor" / "funcao_r19.json"
MAX_RISK = 0.01           # R17 (a)
MAX_BAND_CHANGES = 5      # R19 (c): a escala da T6c.3


def band_changes(log_conf, old_log: float, new_log: float) -> int:
    """Recortes que mudam de faixa (abaixo ou não do limiar) entre os dois limiares."""
    c = np.asarray(log_conf, float)
    return int(((c < old_log) != (c < new_log)).sum())


def verdict(upper: float, changes: int) -> dict:
    """R19 (c): os limiares ficam se os dois critérios valem."""
    return {"funcao_mantida": upper <= MAX_RISK,
            "inferior_mantido": changes <= MAX_BAND_CHANGES}


def _fold_models(work_dir: Path, lot: str, device: str):
    """Troncos e cabeças R12 das 3 sementes da dobra que avalia `lot`."""
    import torch

    from meter_reader.reader import Reader
    from meter_reader.reader_heads import Heads, trunk_features
    from training.reader_function_r12 import R12_HEADS
    from training.reader_heads_train import CONFIG, HEADS_DIR
    from training.reader_train import RUNS_DIR, WEIGHTS_NAME

    pairs = []
    for seed in (0, 1, 2):
        run = f"{lot}_s{seed}"
        model = Reader("L2-12", pretrained=False).to(device)
        model.load_state_dict(torch.load(work_dir / RUNS_DIR / CONFIG / run / WEIGHTS_NAME,
                                         map_location=device, weights_only=True))
        model.eval()
        with torch.no_grad():
            dim = trunk_features(model, torch.zeros(1, 3, 48, 192, device=device)).shape[-1]
        heads = Heads(dim).to(device)
        heads.load_state_dict(torch.load(work_dir / HEADS_DIR / R12_HEADS / run
                                         / "pesos_cabecas.pt", map_location=device,
                                         weights_only=True))
        heads.eval()
        pairs.append((model, heads))
    return pairs


def _read(pairs, crops: list, d1: float, digital: bool, device: str) -> list[dict]:
    """Leitura do comitê (com a D1 no digital), a confiança com T = 1 e P(consumo)."""
    from meter_reader.reader_input import upper_band
    from meter_reader.reader_text import CONSUMPTION_CLASSES, FUNCTION_CLASSES
    from training.reader_ensemble import log_mean_exp
    from training.reader_eval import sequence_logprobs
    from training.reader_function_rule import _committee_read, _outputs

    whole = [_outputs(m, h, crops, device) for m, h in pairs]
    fired = np.zeros(len(crops), bool)
    upper = None
    if digital:
        fired = np.mean([o["duas"] for o in whole], axis=0) >= d1
        idx = np.flatnonzero(fired)
        if len(idx):
            up = [_outputs(m, h, [upper_band(crops[i]) for i in idx], device) for m, h in pairs]
            upper = {i: n for n, i in enumerate(idx)}, up
    cons = [FUNCTION_CLASSES.index(c) for c in CONSUMPTION_CLASSES]
    out = []
    for i in range(len(crops)):
        src, j = (upper[1], upper[0][i]) if fired[i] else (whole, i)
        lps = [o["texto"][j] for o in src]
        text = _committee_read(lps)
        conf = log_mean_exp([sequence_logprobs(lp, [text])[0] for lp in lps]) \
            if text else -math.inf
        p = float(np.mean([o["funcao"][j] for o in src], axis=0)[cons].sum())
        out.append({"texto": text, "log_conf": conf, "consumo": p, "d1": bool(fired[i])})
    return out


def main(device: str = "cuda") -> None:
    from PIL import Image

    from meter_reader.config import get_work_dir
    from training.group_dataset import LABEL_VERSIONS, load_group_photos
    from training.reader_bands import LEGIBILITY_NAME, lower_threshold
    from training.reader_cache import CACHE_DIR, CACHE_NAME, crop_key, load_cache
    from training.reader_dataset import DIGITAL
    from training.reader_function_gold import GOLD_NAME, VALID
    from training.reader_function_r18 import RESULT_NAME as R18_NAME
    from training.reader_function_rule import CONSUMPTION, OUT_DIR, risk_curve
    from training.reader_function_rule import RESULT_NAME as R15_NAME
    from training.reader_gold_cost import TABLE_NAME
    from training.reader_heads_train import FUNCTION_CLASSES, function_class
    from training.reader_labels import CROPS_SHA256
    from training.reader_pilot import csv_rows
    from training.reader_product_parity import set_tf32
    from training.reader_thresholds import OUT_NAME as THRESHOLDS_NAME

    work_dir = get_work_dir()
    out_path = work_dir / OUT_NAME
    if out_path.exists():
        raise SystemExit("A medida da R19 já foi gravada e não é refeita.")
    set_tf32(False)
    thresholds = json.loads((work_dir / THRESHOLDS_NAME).read_text(encoding="utf-8"))
    t_func, d1_final = thresholds["limiar_funcao"], thresholds["d1_logit"]
    old_log = thresholds["limiar_inferior_log"]
    d1_lot = json.loads((work_dir / OUT_DIR / R15_NAME).read_text(encoding="utf-8"))[
        "limiar_d1_por_lote"]
    cache = load_cache(work_dir / CACHE_DIR / CACHE_NAME, CROPS_SHA256)

    # (i) População dos aprováveis e medida da função no limiar fixo
    table = pd.read_csv(work_dir / TABLE_NAME, dtype={"pseudonimo": str, "lote": str})
    expected = {c: r.reading for c, r in csv_rows(load_group_photos(LABEL_VERSIONS["r4"])).items()}
    gold = pd.read_csv(work_dir / GOLD_NAME, dtype=str, keep_default_na=False).set_index("pseudonimo")
    rows = []
    for lot in sorted(table["lote"].unique()):
        codes = list(table.loc[table["lote"] == lot, "pseudonimo"])
        crops = [Image.fromarray(cache[crop_key(c, DIGITAL)]) for c in codes]
        reads = _read(_fold_models(work_dir, lot, device), crops, d1_lot[lot], True, device)
        for c, r in zip(codes, reads):
            rows.append({"pseudonimo": c, "lote": lot, **r,
                         "alta": r["texto"].isdigit() and int(r["texto"]) == int(expected[c])})
        print(f"Lote {lot}: {len(codes)} digitais", flush=True)
    df = pd.DataFrame(rows)
    pop = df[df["alta"]].reset_index(drop=True)
    pop["status"] = pop["pseudonimo"].map(gold["status"]).fillna("sem_julgamento")
    pop["funcao"] = pop["pseudonimo"].map(gold["funcao"]).fillna("")
    valid = pop[pop["status"].isin(VALID)].reset_index(drop=True)
    nc = np.array([FUNCTION_CLASSES[function_class(c)] not in CONSUMPTION for c in valid["funcao"]])
    zeros = np.zeros(len(valid), bool)
    row = risk_curve(valid["consumo"], np.ones(len(valid), bool), nc, zeros, zeros,
                     [t_func]).iloc[0].to_dict()
    r18 = json.loads((work_dir / R18_NAME).read_text(encoding="utf-8"))["R12"]["escolhido"]
    before = set(table.loc[table["alta"].astype(bool), "pseudonimo"])
    now = set(pop["pseudonimo"])
    funcao = {"limiar": t_func, "aprovados": int(row["aprovados"]), "erros": int(row["erros"]),
              "limite_superior": float(row["limite_superior"]),
              "consumo_mantido": float(row["consumo_mantido"]),
              "aprovaveis": len(now), "aprovaveis_r16": len(before),
              "entraram": len(now - before), "sairam": len(before - now),
              "fora_da_medida": pop.loc[~pop["status"].isin(VALID), "status"]
              .value_counts().to_dict(),
              "r18_com_tf32": r18}
    print("\n(i) Função no limiar fixo:", json.dumps(funcao), flush=True)

    # (ii) Limiar inferior na amostra L-a
    judged = pd.read_csv(work_dir / LEGIBILITY_NAME, dtype=str, keep_default_na=False)
    confs, yes = [], []
    for lot, part in judged.groupby("lote"):
        pairs = _fold_models(work_dir, lot, device)
        for c in sorted(part["classe"].unique()):
            sub = part[part["classe"] == c]
            crops = [Image.fromarray(cache[crop_key(p, c)]) for p in sub["pseudonimo"]]
            reads = _read(pairs, crops, d1_final, c == DIGITAL, device)
            confs += [r["log_conf"] for r in reads]
            yes += list(sub["julgamento"] == "sim")
    low = lower_threshold(np.array(confs), np.array(yes))
    new_log = low["log_limiar"] if low["log_limiar"] is not None else -math.inf
    changes = band_changes(confs, old_log, new_log)
    inferior = {**low, "limiar": math.exp(new_log), "limiar_vigente": math.exp(old_log),
                "mudam_de_faixa": changes}
    print("(ii) Limiar inferior:", json.dumps(inferior), flush=True)

    result = {"funcao": funcao, "inferior": inferior,
              "veredito": verdict(funcao["limite_superior"], changes)}
    out_path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print("\nVeredito (R19, c):", result["veredito"])


if __name__ == "__main__":
    main()
