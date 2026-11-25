"""Limiares e temperatura do leitor para o modelo final (ET6c, T6c.3; decisões D-1,
L-2 e K-1).

- D1 (D-1): o limiar de maior F1 da cabeça de duas leituras (R12), com as notas fora
  da dobra dos 3 lotes juntas (média dos logits das 3 sementes).
- Limiar inferior (L-2): a regra da R10 (V = 10%, pelo menos 30 abaixo) na amostra
  L-a, com a leitura do produto: comitê das 3 sementes de cada dobra e, nos digitais
  em que a D1 dispara, a faixa de cima.
- Temperatura do comitê (K-1): ajustada pela entropia cruzada entre a confiança da
  leitura livre do comitê e o acerto, fora da dobra, nos alvos de texto (sem os
  pseudo-rótulos). A leitura escolhida é a de T = 1; a temperatura só reescala a
  confiança relatada.
Grava `limiares_leitor.json` na pasta de trabalho (com o limiar da função da R17 e
da R18), sem sobrescrever. Só números.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

OUT_NAME = Path("experimentos") / "leitor" / "limiares_leitor.json"
OLD_LOWER = 0.202   # E103, sem a releitura pela D1


def committee_pick(probs: np.ndarray, texts: list[list[str]]) -> tuple[list[str], np.ndarray]:
    """Leitura do comitê por item: entre as leituras livres das sementes (sem as vazias),
    a de maior probabilidade média. `probs[k, s, i]`: probabilidade, na semente s, da
    leitura livre da semente k no item i; `texts[k][i]`: essa leitura. Devolve as
    leituras e a probabilidade média delas (0 quando todas são vazias)."""
    n_cand, _, n = probs.shape
    mean = probs.mean(axis=1)                                   # (candidatos, itens)
    out_text, out_p = [], np.zeros(n)
    for i in range(n):
        best = None
        for k in range(n_cand):
            t = texts[k][i]
            if t and (best is None or mean[k, i] > mean[best, i]
                      or (mean[k, i] == mean[best, i] and t < texts[best][i])):
                best = k
        out_text.append(texts[best][i] if best is not None else "")
        out_p[i] = mean[best, i] if best is not None else 0.0
    return out_text, out_p


def fit_committee_temperature(lps: list[np.ndarray], texts: list[str],
                              correct: np.ndarray) -> float:
    """T que minimiza a entropia cruzada binária entre a probabilidade média (nas
    sementes) da leitura escolhida e o acerto."""
    from scipy.optimize import minimize_scalar

    from meter_reader.reader_text import TEXT_ALPHABET
    from training.reader_calibration import LOG_T_BOUNDS, free_confidence

    correct = np.asarray(correct, float)

    def nll(log_t: float) -> float:
        p = np.mean([free_confidence(lp, texts, TEXT_ALPHABET, float(np.exp(log_t)))
                     for lp in lps], axis=0)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return float(-np.mean(correct * np.log(p) + (1 - correct) * np.log(1 - p)))

    return float(np.exp(minimize_scalar(nll, bounds=LOG_T_BOUNDS, method="bounded").x))


def _committee_from_runs(work_dir: Path, items: pd.DataFrame):
    """Predições fora da dobra das 3 sementes alinhadas por item, a leitura do comitê
    (T = 1), o acerto e as log-probabilidades por semente. Sem pseudo-rótulos."""
    from meter_reader.reader_text import TEXT_ALPHABET
    from training.reader_calibration import free_confidence, free_readings
    from training.reader_heads_train import CONFIG
    from training.reader_train import PREDICTIONS_NAME, RUNS_DIR, text_matches

    index = items.set_index(items["pseudonimo"] + "|" + items["classe"] + "|" + items["recorte"])
    all_lps, all_texts, all_ok = [[], [], []], [], []
    for lot in sorted(items["lote"].unique()):
        preds = [np.load(work_dir / RUNS_DIR / CONFIG / f"{lot}_s{s}" / PREDICTIONS_NAME)
                 for s in (0, 1, 2)]
        keys = [str(k) for k in preds[0]["chave_texto"]]
        for p in preds[1:]:
            if [str(k) for k in p["chave_texto"]] != keys:
                raise SystemExit("As sementes não têm os mesmos itens.")
        rows = index.loc[keys]
        keep = (rows["origem"] != "pseudo").to_numpy()
        rows = rows[keep]
        lps = [p["texto_0"].astype(np.float32)[keep] for p in preds]
        free = [free_readings(lp, TEXT_ALPHABET) for lp in lps]
        probs = np.array([[free_confidence(lp, free[k], TEXT_ALPHABET) for lp in lps]
                          for k in range(3)])
        texts, _ = committee_pick(probs, free)
        has = np.array([bool(t) for t in texts])
        for s in range(3):
            all_lps[s].append(lps[s][has])
        all_texts += [t for t, h in zip(texts, has) if h]
        all_ok += [text_matches(t, tg, c) for t, tg, c, h
                   in zip(texts, rows["texto"], rows["classe"], has) if h]
    return [np.concatenate(v) for v in all_lps], all_texts, np.array(all_ok, bool)


def main(device: str = "cuda") -> None:
    import torch
    from PIL import Image

    from meter_reader.config import get_work_dir
    from meter_reader.reader import Reader
    from meter_reader.reader_input import reader_tensor, upper_band
    from meter_reader.reader_text import TEXT_ALPHABET
    from training.reader_bands import LEGIBILITY_NAME, lower_threshold
    from training.reader_cache import CACHE_DIR, CACHE_NAME, crop_key, load_cache
    from training.reader_calibration import ece, free_confidence
    from training.reader_dataset import DIGITAL, load_items
    from training.reader_ensemble import log_mean_exp
    from training.reader_eval import sequence_logprobs
    from training.reader_function_r12 import extended_items
    from training.reader_function_r18 import RESULT_NAME as R18_NAME
    from training.reader_function_rule import _committee_read
    from training.reader_gold_cost import TABLE_NAME
    from training.reader_heads_eval import f1_threshold
    from training.reader_heads_train import CONFIG
    from training.reader_labels import CROPS_SHA256
    from training.reader_train import RUNS_DIR, WEIGHTS_NAME

    work_dir = get_work_dir()
    out_path = work_dir / OUT_NAME
    if out_path.exists():
        raise SystemExit("Os limiares do leitor já foram gravados e não são sobrescritos.")

    # D1 (D-1): F1 com os 3 lotes juntos
    table = pd.read_csv(work_dir / TABLE_NAME, dtype={"pseudonimo": str, "lote": str})
    logit = dict(zip(table["pseudonimo"], table["logit_duas"]))
    items = extended_items()
    two = items[items["usa_duas"] & (items["classe"] == DIGITAL)].drop_duplicates("pseudonimo")
    two = two[two["pseudonimo"].isin(logit)]
    d1 = f1_threshold(two["duas_leituras"].astype(bool).to_numpy(),
                      two["pseudonimo"].map(logit).to_numpy())
    print(f"D1 (D-1): limiar do logit {d1:.3f}; marcas {len(two)}; "
          f"dispara em {float((table['logit_duas'] >= d1).mean()):.1%} dos digitais")

    # Limiar inferior (L-2): leitura do produto na amostra L-a
    judged = pd.read_csv(work_dir / LEGIBILITY_NAME, dtype=str, keep_default_na=False)
    cache = load_cache(work_dir / CACHE_DIR / CACHE_NAME, CROPS_SHA256)
    reads = []
    for lot, part in judged.groupby("lote"):
        crops = []
        fired = []
        for p, c in zip(part["pseudonimo"], part["classe"]):
            crop = Image.fromarray(cache[crop_key(p, c)])
            up = c == DIGITAL and logit.get(p, -math.inf) >= d1
            crops.append(upper_band(crop) if up else crop)
            fired.append(up)
        outs = []
        for seed in (0, 1, 2):
            model = Reader("L2-12", pretrained=False).to(device)
            model.load_state_dict(torch.load(work_dir / RUNS_DIR / CONFIG / f"{lot}_s{seed}"
                                             / WEIGHTS_NAME, map_location=device,
                                             weights_only=True))
            model.eval()
            with torch.no_grad():
                x = torch.stack([reader_tensor(c) for c in crops]).to(device)
                outs.append(model(x)["texto"].float().log_softmax(-1).cpu().numpy())
        for k, r in enumerate(part.to_dict("records")):
            lps = [o[k] for o in outs]
            text = _committee_read(lps)
            conf = log_mean_exp([sequence_logprobs(lp, [text])[0] for lp in lps]) \
                if text else -math.inf
            reads.append({"julgamento": r["julgamento"], "log_conf": conf, "d1": fired[k]})
    df = pd.DataFrame(reads)
    low = lower_threshold(df["log_conf"].to_numpy(), (df["julgamento"] == "sim").to_numpy())
    new_log = low["log_limiar"] if low["log_limiar"] is not None else -math.inf
    below_old = df["log_conf"] < math.log(OLD_LOWER)
    below_new = df["log_conf"] < new_log
    print(f"\nLimiar inferior (L-2): {low}")
    print(f"  probabilidade {math.exp(new_log):.4f} (antes {OLD_LOWER}); D1 disparou em "
          f"{int(df['d1'].sum())} de {len(df)} recortes; mudam de faixa: "
          f"{int((below_old != below_new).sum())}")

    # Temperatura do comitê (K-1)
    base, _ = load_items()
    lps, texts, ok = _committee_from_runs(work_dir, base)
    t_fit = fit_committee_temperature(lps, texts, ok)
    conf = {T: np.mean([free_confidence(lp, texts, TEXT_ALPHABET, T) for lp in lps], axis=0)
            for T in (1.0, t_fit)}
    print(f"\nTemperatura do comitê (K-1): {t_fit:.3f}; itens {len(texts)}; acerto "
          f"{ok.mean():.3f}; ECE {ece(conf[1.0], ok):.3f} -> {ece(conf[t_fit], ok):.3f}")

    r18 = json.loads((work_dir / R18_NAME).read_text(encoding="utf-8"))
    result = {"d1_logit": d1, "limiar_inferior_log": new_log,
              "limiar_inferior": math.exp(new_log), "temperatura_texto": t_fit,
              "limiar_funcao": r18["R12"]["escolhido"]["limiar"], "meta_risco": 0.01,
              "fonte": "modelos de 2 lotes, fora da dobra (T6c.3)"}
    out_path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print("\nGravado:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in result.items()})


if __name__ == "__main__":
    main()
