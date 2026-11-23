"""Faixas e limiar inferior do leitor (protocolo do leitor, seção 1.11, L-a; revisão R10).

Para cada recorte da amostra de legibilidade (julgada antes de ver as confianças),
cada modelo fora da dobra da configuração final faz a leitura do produto: recorte
inteiro e, num digital com duas leituras, a faixa de cima (D-b). Com a leitura livre
e a sua confiança:

- limiar inferior: o maior que mantém em no máximo 10% (V) a fração de "sim" entre
  os recortes com confiança abaixo dele, com pelo menos 30 abaixo; um por semente,
  e o adotado é a mediana dos três (R10). Sem leitura extraída, a confiança é -inf
  (sempre "muito baixa");
- faixas (alta, intermediária, muito baixa) cruzadas com o julgamento humano: um
  "outro valor" na faixa alta é uma confirmação falsa num caso real.

Só contagens e limiares.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

CONFIG = "L2-12+auto1"
LEGIBILITY_NAME = Path("experimentos") / "leitor" / "legibilidade_L-a.csv"
MAX_YES_BELOW = 0.10   # V
MIN_BELOW = 30


def lower_threshold(log_conf: np.ndarray, yes: np.ndarray, max_yes: float = MAX_YES_BELOW,
                    min_below: int = MIN_BELOW) -> dict:
    """O maior limiar t (entre as confianças observadas, ou +inf) tal que, entre os
    recortes com confiança estritamente abaixo de t, haja pelo menos `min_below` e a
    fração de "sim" seja de no máximo `max_yes`."""
    log_conf, yes = np.asarray(log_conf, float), np.asarray(yes, bool)
    best = {"log_limiar": None, "abaixo": 0, "sim_abaixo": 0}
    for t in np.append(np.unique(log_conf[np.isfinite(log_conf)]), np.inf):
        below = log_conf < t
        n = int(below.sum())
        if n >= min_below and yes[below].mean() <= max_yes:
            best = {"log_limiar": float(t), "abaixo": n, "sim_abaixo": int(yes[below].sum())}
    return best


def band(text: str, log_conf: float, csv_logp: float, expected: int, log_high: float,
         log_low: float | None) -> str:
    """Faixa da seção 1.11: alta (leitura igual ao CSV e probabilidade do CSV acima do
    limiar alto); muito baixa (sem leitura ou confiança abaixo do limiar inferior);
    intermediária, o resto."""
    if text.isdigit() and int(text) == expected and csv_logp >= log_high:
        return "alta"
    if not text or (log_low is not None and log_conf < log_low):
        return "muito_baixa"
    return "intermediaria"


def read_sample(model, crops: list, classes: list[str], expected: list[int], gray: bool,
                device: str) -> list[dict]:
    """Leitura do produto em cada recorte: texto livre, confiança, log P do valor do CSV
    e se a cabeça de duas leituras disparou."""
    from meter_reader.reader_input import reader_tensor, upper_band
    from meter_reader.reader_text import TEXT_ALPHABET, greedy_decode
    from training.reader_dataset import DIGITAL
    from training.reader_eval import sequence_logprobs, value_logprob

    out = []
    model.eval()
    with torch.no_grad():
        for crop, cls, exp in zip(crops, classes, expected):
            o = model(reader_tensor(crop, gray=gray).unsqueeze(0).to(device))
            two = cls == DIGITAL and float(o["duas"][0]) > 0
            if two:   # D-b: relê a faixa de cima
                o = model(reader_tensor(upper_band(crop), gray=gray).unsqueeze(0).to(device))
            lp = o["texto"][0].float().log_softmax(-1).cpu().numpy()
            text = greedy_decode(lp.argmax(-1).tolist(), TEXT_ALPHABET)
            conf = float(sequence_logprobs(lp, [text])[0]) if text else -math.inf
            out.append({"texto_livre": text, "log_conf": conf, "duas": two,
                        "log_csv": value_logprob(lp, str(exp), cls)})
    return out


def main(device: str = "cuda") -> None:
    from PIL import Image

    from meter_reader.config import get_work_dir
    from meter_reader.reader import Reader
    from training.group_dataset import LABEL_VERSIONS, load_group_photos
    from training.reader_cache import CACHE_DIR, CACHE_NAME, crop_key, load_cache
    from training.reader_dataset import load_items
    from training.reader_eval import score_config
    from training.reader_labels import CROPS_SHA256
    from training.reader_metrics import config_table
    from training.reader_pilot import csv_rows
    from training.reader_train import META_NAME, RUNS_DIR, WEIGHTS_NAME

    config = sys.argv[1] if len(sys.argv) > 1 else CONFIG
    work_dir = get_work_dir()
    judged = pd.read_csv(work_dir / LEGIBILITY_NAME, dtype=str, keep_default_na=False)
    rows = csv_rows(load_group_photos(LABEL_VERSIONS["r4"]))
    judged["esperado"] = judged["pseudonimo"].map(lambda p: int(rows[p].reading))
    cache = load_cache(work_dir / CACHE_DIR / CACHE_NAME, CROPS_SHA256)
    items, _ = load_items()
    highs = config_table(score_config(work_dir, config, items)).set_index(["semente", "classe"])

    reads = []
    for meta_path in sorted((work_dir / RUNS_DIR / config).glob(f"*/{META_NAME}")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        part = judged[judged["lote"] == meta["lote_avaliado"]]
        model = Reader(meta["candidato"], pretrained=False).to(device)
        model.load_state_dict(torch.load(meta_path.parent / WEIGHTS_NAME, map_location=device,
                                         weights_only=True))
        crops = [Image.fromarray(cache[crop_key(p, c)])
                 for p, c in zip(part["pseudonimo"], part["classe"])]
        for r, res in zip(part.to_dict("records"),
                          read_sample(model, crops, part["classe"].tolist(),
                                      part["esperado"].tolist(), meta["cinza"], device)):
            reads.append({**r, **res, "semente": meta["semente"]})
    df = pd.DataFrame(reads)

    print(f"Amostra L-a: {judged['julgamento'].value_counts().to_dict()}")
    print("Julgamento por classe:")
    print(pd.crosstab(judged["classe"], judged["julgamento"]).to_string())
    lows = {}
    for seed, part in df.groupby("semente"):
        lows[seed] = lower_threshold(part["log_conf"].to_numpy(),
                                     (part["julgamento"] == "sim").to_numpy())
        print(f"Semente {seed}: {lows[seed]}")
    # R10: semente sem limiar válido conta como -inf (só "sem leitura" fica abaixo)
    values = [v["log_limiar"] if v["log_limiar"] is not None else -math.inf
              for v in lows.values()]
    adopted = float(np.median(values))
    print(f"\nLimiar inferior adotado (mediana das sementes): log {adopted}"
          + (f" (probabilidade {math.exp(adopted):.4f})" if adopted is not None
             and np.isfinite(adopted) else ""))

    df["faixa"] = [band(r["texto_livre"], r["log_conf"], r["log_csv"], r["esperado"],
                        highs.loc[(r["semente"], r["classe"]), "log_limiar"], adopted)
                   for r in df.to_dict("records")]
    print("\nFaixa x julgamento (3 sementes somadas):")
    print(pd.crosstab([df["classe"], df["julgamento"]], df["faixa"], margins=True,
                      margins_name="total").to_string())
    print(f"\nDuas leituras disparadas nos digitais: "
          f"{int(df.loc[df['classe'] == 'display_digital', 'duas'].sum())} "
          f"de {int((df['classe'] == 'display_digital').sum())}")


if __name__ == "__main__":
    main()
