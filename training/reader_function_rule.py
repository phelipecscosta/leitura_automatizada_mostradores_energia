"""Regra de aceitação da função (revisões R15 e R15-b do protocolo do leitor; ET6c).

Fora da dobra, com o comitê das 3 sementes de cada dobra e as cabeças R12:
- leitura pelo fluxo do produto: recorte inteiro e, se a D1 disparar (média dos
  logits das 3 cabeças acima do limiar do F1 escolhido nos outros dois lotes), a
  faixa de cima;
- consumo = P(vazio) + P(03) + P(003), média das 3 cabeças (V-a);
- aprovado em t: leitura livre do comitê igual ao CSV (numérica) e consumo >= t;
- erro: aprovado com função rotulada que não é consumo; nos aceitos não julgados,
  erros estimados pela razão k(t)/m(t) dos 300 julgados (R15-b).
Regra (Y-f2): o menor t, nos 201 quantis do consumo, com o limite superior de 95%
(Clopper-Pearson) dos erros entre os aprovados de no máximo 0,7%. Gatilho G-a:
menos de 50% dos consumos mantidos no t escolhido (ou nenhum t). Também relata o
B10-b nos ciclométricos. Grava a curva e o resultado na pasta de trabalho. Só números.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

MAX_RISK = 0.007            # Y-f2 (E35)
MIN_KEPT = 0.50             # G-a
GRID = 201
FALLBACK_RATE = 2 / 298     # R15-b: fração geral do julgamento Q-a
CONSUMPTION = ("vazio", "03", "003")
OUT_DIR = Path("experimentos") / "leitor"
CURVE_NAME = "funcao_r15_curva.csv"
RESULT_NAME = "funcao_r15.json"
POPULATION_NAME = "funcao_r15_populacao.csv"


def breakdown(p_cons, high, nc, judged, unjudged, t: float) -> dict:
    """Abertura dos erros num limiar: diretos, julgados aprovados (m), não consumo entre
    eles (k), não julgados aprovados (u) e erros projetados (R15-b)."""
    p_cons, high = np.asarray(p_cons, float), np.asarray(high, bool)
    nc, judged, unjudged = (np.asarray(v, bool) for v in (nc, judged, unjudged))
    approved = high & (p_cons >= t)
    direct = int((approved & nc & ~unjudged).sum())
    return {"limiar": round(float(t), 4), "diretos": direct,
            "m": int((approved & judged).sum()), "k": int((approved & judged & nc).sum()),
            "u": int((approved & unjudged).sum()),
            "projetados": errors_at(approved, nc, judged, unjudged) - direct}


def errors_at(approved, nc, judged, unjudged) -> int:
    """Erros num limiar: os rotulados (fora dos aceitos não julgados) mais a estimativa
    por razão nos aceitos não julgados aprovados (R15-b)."""
    approved, nc = np.asarray(approved, bool), np.asarray(nc, bool)
    judged, unjudged = np.asarray(judged, bool), np.asarray(unjudged, bool)
    direct = int((approved & nc & ~unjudged).sum())
    m = int((approved & judged).sum())
    k = int((approved & judged & nc).sum())
    rate = k / m if m else FALLBACK_RATE
    return direct + math.ceil(int((approved & unjudged).sum()) * rate - 1e-12)


def risk_curve(p_cons, high, nc, judged, unjudged, thresholds) -> pd.DataFrame:
    """Uma linha por limiar: aprovados, erros, limite superior e consumos mantidos."""
    from training.reader_metrics import cp_upper

    p_cons, high, nc = np.asarray(p_cons, float), np.asarray(high, bool), np.asarray(nc, bool)
    rows = []
    for t in thresholds:
        approved = high & (p_cons >= t)
        n, e = int(approved.sum()), errors_at(approved, nc, judged, unjudged)
        rows.append({"limiar": float(t), "aprovados": n, "erros": e,
                     "limite_superior": cp_upper(e, n),
                     "consumo_mantido": float((p_cons[~nc] >= t).mean())})
    return pd.DataFrame(rows)


def choose(curve: pd.DataFrame, max_risk: float = MAX_RISK) -> dict | None:
    """O menor limiar que cumpre o Y-f2; None se nenhum cumpre."""
    ok = curve[(curve["aprovados"] > 0) & (curve["limite_superior"] <= max_risk)]
    if ok.empty:
        return None
    row = ok.sort_values("limiar").iloc[0]
    return {"limiar": float(row["limiar"]), "aprovados": int(row["aprovados"]),
            "erros": int(row["erros"]), "limite_superior": float(row["limite_superior"]),
            "consumo_mantido": float(row["consumo_mantido"])}


def _population(items: pd.DataFrame, judged: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por foto digital com rótulo de função; nos 300 julgados, vale o
    julgamento, e os julgados sem função saem."""
    from training.reader_heads_train import FUNCTION_CLASSES, function_class

    f = items[items["usa_funcao"] & (items["classe"] == "display_digital")]
    if f["pseudonimo"].duplicated().any():
        raise SystemExit("Foto com mais de um rótulo de função.")
    pop = f[["pseudonimo", "lote", "funcao", "origem"]].reset_index(drop=True)
    j = judged.set_index("pseudonimo")
    out = j.index[j["status"].isin(["ilegivel", "nao_digital"])]
    pop = pop[~pop["pseudonimo"].isin(out)].reset_index(drop=True)
    pop["julgado"] = pop["pseudonimo"].isin(j.index)
    code = pop["pseudonimo"].map(lambda p: j.at[p, "funcao"] if p in j.index else None)
    pop["funcao"] = np.where(pop["julgado"], code.fillna(""), pop["funcao"])
    pop["nc"] = [FUNCTION_CLASSES[function_class(c)] not in CONSUMPTION for c in pop["funcao"]]
    pop["nao_julgado"] = (pop["origem"] == "haiku_csv") & ~pop["julgado"]
    return pop


def _outputs(model, heads, crops, device, batch: int = 256) -> dict:
    """Texto (log-probabilidades), função (probabilidades) e logit de duas leituras."""
    import torch

    from meter_reader.reader_input import reader_tensor
    from training.reader_heads_train import trunk_features

    text, func, two = [], [], []
    with torch.no_grad():
        for s in range(0, len(crops), batch):
            x = torch.stack([reader_tensor(c) for c in crops[s:s + batch]]).to(device)
            text.append(model(x)["texto"].float().log_softmax(-1).cpu().numpy())
            f, d = heads(trunk_features(model, x).float())
            func.append(f.softmax(-1).cpu().numpy())
            two.append(d.cpu().numpy())
    return {"texto": np.concatenate(text), "funcao": np.concatenate(func),
            "duas": np.concatenate(two)}


def _committee_read(lps: list[np.ndarray]) -> str:
    from meter_reader.reader_text import TEXT_ALPHABET, greedy_decode
    from training.reader_ensemble import log_mean_exp
    from training.reader_eval import sequence_logprobs

    reads = {greedy_decode(lp.argmax(-1).tolist(), TEXT_ALPHABET) for lp in lps} - {""}
    if not reads:
        return ""
    conf = {r: log_mean_exp([sequence_logprobs(lp, [r])[0] for lp in lps]) for r in reads}
    return max(sorted(conf), key=conf.get)


def main(device: str = "cuda") -> None:
    import torch
    from PIL import Image

    from meter_reader.config import get_work_dir
    from meter_reader.reader import Reader
    from meter_reader.reader_input import upper_band
    from training.group_dataset import LABEL_VERSIONS, load_group_photos
    from training.reader_cache import CACHE_DIR, CACHE_NAME, crop_key, load_cache
    from training.reader_dataset import CYCLOMETRIC, DIGITAL
    from training.reader_ensemble import ENSEMBLE
    from training.reader_eval import EVAL_DIR
    from training.reader_function_r12 import R12_HEADS, extended_items
    from training.reader_heads_eval import f1_threshold
    from training.reader_heads_train import CONFIG, FUNCTION_CLASSES, HEADS_DIR, Heads, \
        trunk_features
    from training.reader_labels import CROPS_SHA256
    from training.reader_pilot import csv_rows
    from training.reader_train import RUNS_DIR, WEIGHTS_NAME

    work_dir = get_work_dir()
    items = extended_items()
    judged = pd.read_csv(work_dir / OUT_DIR / "rotulagem_r15_aceitos.csv", dtype=str,
                         keep_default_na=False)
    expected = {c: r.reading for c, r in csv_rows(load_group_photos(LABEL_VERSIONS["r4"])).items()}
    pop = _population(items, judged)
    pop = pop[pop["pseudonimo"].isin(expected)].reset_index(drop=True)
    two_items = items[items["usa_duas"] & (items["classe"] == DIGITAL)]
    cache = load_cache(work_dir / CACHE_DIR / CACHE_NAME, CROPS_SHA256)
    cons_idx = [FUNCTION_CLASSES.index(c) for c in CONSUMPTION]

    # Saídas por lote e semente: recorte inteiro (população e marcas de duas leituras)
    # e faixa de cima (população)
    out = {}
    for lot in sorted(pop["lote"].unique()):
        codes = sorted(set(pop.loc[pop["lote"] == lot, "pseudonimo"])
                       | set(two_items.loc[two_items["lote"] == lot, "pseudonimo"]))
        whole = [Image.fromarray(cache[crop_key(c, DIGITAL)]) for c in codes]
        upper = [upper_band(c) for c in whole]
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
            out[(lot, seed)] = {"pos": {c: i for i, c in enumerate(codes)},
                                "inteiro": _outputs(model, heads, whole, device),
                                "cima": _outputs(model, heads, upper, device)}
        print(f"Lote {lot}: {len(codes)} recortes", flush=True)

    # D1: média dos logits das 3 sementes; limiar pelo F1 nos outros dois lotes
    logit = {}
    for (lot, seed), o in out.items():
        for c, v in zip(o["pos"], o["inteiro"]["duas"]):
            logit.setdefault(c, []).append(float(v))
    marks = two_items.set_index("pseudonimo")["duas_leituras"].astype(bool)
    lot_of = two_items.set_index("pseudonimo")["lote"]
    d1 = {}
    for lot in sorted(pop["lote"].unique()):
        fit = [c for c in marks.index if lot_of[c] != lot]
        d1[lot] = f1_threshold(marks[fit].to_numpy(), np.array([np.mean(logit[c]) for c in fit]))

    # Leitura e função do comitê, foto a foto
    p_cons, high, fired = [], [], []
    for r in pop.to_dict("records"):
        lot, c = r["lote"], r["pseudonimo"]
        pos = [out[(lot, s)]["pos"][c] for s in (0, 1, 2)]
        up = float(np.mean(logit[c])) >= d1[lot]
        key = "cima" if up else "inteiro"
        lps = [out[(lot, s)][key]["texto"][p] for s, p in zip((0, 1, 2), pos)]
        probs = np.mean([out[(lot, s)][key]["funcao"][p] for s, p in zip((0, 1, 2), pos)], axis=0)
        text = _committee_read(lps)
        high.append(text.isdigit() and int(text) == int(expected[c]))
        p_cons.append(float(probs[cons_idx].sum()))
        fired.append(up)
    pop["consumo"], pop["alta"], pop["d1"] = p_cons, high, fired

    thresholds = np.unique(np.quantile(pop["consumo"], np.linspace(0, 1, GRID)))
    curve = risk_curve(pop["consumo"], pop["alta"], pop["nc"], pop["julgado"],
                       pop["nao_julgado"], thresholds)
    curve.to_csv(work_dir / OUT_DIR / CURVE_NAME, index=False)
    pop[["pseudonimo", "lote", "nc", "julgado", "nao_julgado", "alta", "d1", "consumo"]].to_csv(
        work_dir / OUT_DIR / POPULATION_NAME, index=False)
    chosen = choose(curve)
    kept = 0.0 if chosen is None else chosen["consumo_mantido"]
    result = {"limiar_d1_por_lote": d1, "escolhido": chosen, "gatilho_g_a": kept < MIN_KEPT,
              "populacao": int(len(pop)), "nao_consumo": int(pop["nc"].sum()),
              "julgados": int(pop["julgado"].sum()), "nao_julgados": int(pop["nao_julgado"].sum()),
              "faixa_alta": int(pop["alta"].sum()), "d1_disparou": int(pop["d1"].sum())}
    (work_dir / OUT_DIR / RESULT_NAME).write_text(json.dumps(result, indent=1), encoding="utf-8")

    pd.set_option("display.width", 160)
    print("\nPopulação:", {k: v for k, v in result.items() if k not in ("escolhido",)})
    print("Funções que não são consumo, por código:",
          pop.loc[pop["nc"], "funcao"].value_counts().to_dict())
    print("\nCurva (a cada 20 quantis):")
    print(curve.iloc[::20].round(4).to_string(index=False))
    print("\nEscolhido (Y-f2):", "nenhum limiar" if chosen is None else
          {k: round(v, 4) if isinstance(v, float) else v for k, v in chosen.items()})
    print("Gatilho G-a (F-x4):", result["gatilho_g_a"])
    print("\nAbertura dos erros (a cada 20 quantis):")
    print(pd.DataFrame([breakdown(pop["consumo"], pop["alta"], pop["nc"], pop["julgado"],
                                  pop["nao_julgado"], t) for t in curve["limiar"].iloc[::20]])
          .to_string(index=False))
    sel = pop["julgado"] & pop["nc"]
    print("Julgados de não consumo (Q-a): consumo previsto",
          [round(v, 3) for v in pop.loc[sel, "consumo"]], "; faixa alta",
          list(pop.loc[sel, "alta"]))

    # B10-b nos ciclométricos (só relato)
    comite = pd.read_csv(work_dir / EVAL_DIR / f"{ENSEMBLE}.csv", dtype={"livre": str},
                         keep_default_na=False)
    cyc = comite[comite["classe"] == CYCLOMETRIC].copy()
    cyc["esperado"] = cyc["chave"].str.split("|").str[0].map(expected).astype(int)
    digits = cyc["livre"].str.fullmatch(r"\d+")
    diff = np.where(digits, (cyc["livre"].where(digits, "0").astype(int) - cyc["esperado"]).abs(), -1)
    print(f"\nB10-b, ciclométricos fora da dobra ({len(cyc)}): igual ao CSV "
          f"{float((diff == 0).mean()):.3f}; com a tolerância de 1 {float(np.isin(diff, [0, 1]).mean()):.3f}")
    lab = items[(items["classe"] == CYCLOMETRIC) & items["usa_texto"]]
    lab_diff = (lab["texto"].astype(int) - lab["pseudonimo"].map(expected).astype(int)).abs()
    print(f"  Rótulos com valor verdadeiro diferente do CSV: {int((lab_diff > 0).sum())}; "
          f"dos quais diferença de 1: {int((lab_diff == 1).sum())}")


if __name__ == "__main__":
    main()
