"""Comitê das 3 sementes do leitor (protocolo do leitor, revisão R10, item b).

Para cada recorte, a probabilidade de uma sequência é a média das probabilidades dos
3 modelos da dobra. A leitura livre é, entre as leituras livres dos 3, a de maior
probabilidade média; a confiança, a probabilidade do rótulo e as dos impostores
seguem a mesma média. As notas vão para `avaliacao/<configuração>+comite.csv`, como
uma configuração de semente única, e o comitê é comparado com o modelo único pela
regra da seção 1.9 (bootstrap pareado por recorte). Adotado se a diferença de M-c
tiver o IC acima de zero e a latência p95 dos 3 modelos em sequência (CPU, 4
threads, lote 1) for de no máximo 50 ms. Só números.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from training.reader_heads_train import CONFIG

ENSEMBLE = CONFIG + "+comite"
LATENCY_CROPS = 200


def log_mean_exp(values) -> float:
    """log da média das probabilidades, a partir dos logs."""
    v = np.asarray(values, dtype=float)
    if np.all(np.isneginf(v)):
        return -np.inf
    m = v[np.isfinite(v)].max()
    return float(m + np.log(np.mean(np.exp(v - m))))


def ensemble_item(lps: list[np.ndarray], label: str, field_class: str, key: str) -> dict:
    """Notas de um item pelo comitê, nas mesmas colunas de `score_item`."""
    from meter_reader.reader_text import TEXT_ALPHABET, greedy_decode
    from training.reader_eval import impostors, same_value, sequence_logprobs, value_logprob

    reads = {greedy_decode(np.asarray(lp).argmax(-1).tolist(), TEXT_ALPHABET) for lp in lps}
    reads.discard("")
    if reads:
        conf = {r: log_mean_exp([sequence_logprobs(lp, [r])[0] for lp in lps]) for r in reads}
        text = max(sorted(conf), key=conf.get)
        best = conf[text]
    else:
        text, best = "", -np.inf

    def ens(value: str) -> float:
        return log_mean_exp([value_logprob(lp, value, field_class) for lp in lps])

    uniform, confusable = impostors(label, field_class, key)
    return {"livre": text, "conf_livre": best, "acerto": same_value(text, label, field_class),
            "logp_rotulo": ens(label),
            "imp_uniforme_lido": same_value(text, uniform, field_class),
            "logp_imp_uniforme": ens(uniform),
            "tem_imp_confundivel": confusable is not None,
            "imp_confundivel_lido": confusable is not None
            and same_value(text, confusable, field_class),
            "logp_imp_confundivel": ens(confusable) if confusable else np.nan}


def score_ensemble(work_dir, items: pd.DataFrame) -> pd.DataFrame:
    """Notas do comitê nos 3 lotes, gravadas como a configuração ENSEMBLE."""
    from training.reader_eval import EVAL_DIR
    from training.reader_train import PREDICTIONS_NAME, RUNS_DIR

    out_path = work_dir / EVAL_DIR / f"{ENSEMBLE}.csv"
    by_key = items.set_index(items["pseudonimo"] + "|" + items["classe"] + "|" + items["recorte"])
    frames = []
    for lot in sorted({p.name.rsplit("_s", 1)[0] for p in (work_dir / RUNS_DIR / CONFIG).iterdir()}):
        preds = [np.load(work_dir / RUNS_DIR / CONFIG / f"{lot}_s{s}" / PREDICTIONS_NAME)
                 for s in (0, 1, 2)]
        keys = preds[0]["chave_texto"]
        if any(list(p["chave_texto"]) != list(keys) for p in preds[1:]):
            raise SystemExit(f"Recortes diferentes entre as sementes no lote {lot}.")
        rows = []
        for k, key in enumerate(keys):
            r = by_key.loc[key]
            lps = [p["texto_0"][k].astype(np.float32) for p in preds]
            rows.append({"semente": 0, "lote": lot, "chave": key, "classe": r["classe"],
                         "origem": r["origem"], **ensemble_item(lps, r["texto"], r["classe"], key)})
        frames.append(pd.DataFrame(rows))
        print(f"Lote {lot}: {len(rows)} recortes", flush=True)
    scores = pd.concat(frames, ignore_index=True)
    scores.to_csv(out_path, index=False)
    return scores


def ensemble_latency_ms(candidate: str, crops: list, n_models: int = 3) -> dict:
    """p50 e p95 em CPU (4 threads, lote 1) dos n modelos em sequência, com o
    pré-processamento."""
    import torch

    from meter_reader.reader import Reader
    from meter_reader.reader_input import reader_tensor
    from training.reader_rule import LATENCY_THREADS, LATENCY_WARMUP

    torch.set_num_threads(LATENCY_THREADS)
    models = [Reader(candidate, pretrained=False).eval() for _ in range(n_models)]
    times = []
    with torch.no_grad():
        for i, crop in enumerate(crops):
            start = time.perf_counter()
            x = reader_tensor(crop).unsqueeze(0)
            for m in models:
                m(x)
            if i >= LATENCY_WARMUP:
                times.append((time.perf_counter() - start) * 1000)
    return {"p50": float(np.percentile(times, 50)), "p95": float(np.percentile(times, 95))}


def main() -> None:
    from PIL import Image

    from meter_reader.config import get_work_dir
    from training.reader_cache import CACHE_DIR, CACHE_NAME, crop_key, load_cache
    from training.reader_dataset import load_items
    from training.reader_labels import CROPS_SHA256
    from training.reader_phases import compare
    from training.reader_rule import LATENCY_CEILING_MS

    work_dir = get_work_dir()
    items, _ = load_items()
    score_ensemble(work_dir, items)
    result = compare(work_dir, items, ENSEMBLE, CONFIG)
    print("Comitê contra modelo único:", json.dumps(result, ensure_ascii=False))

    cache = load_cache(work_dir / CACHE_DIR / CACHE_NAME, CROPS_SHA256)
    sample = items[items["usa_texto"]].sample(LATENCY_CROPS, random_state=0)
    crops = [Image.fromarray(cache[crop_key(p, c)])
             for p, c in zip(sample["pseudonimo"], sample["classe"])]
    latency = ensemble_latency_ms("L2-12", crops)
    print(f"Latência do comitê (CPU, 4 threads, lote 1): p50 {latency['p50']:.1f} ms; "
          f"p95 {latency['p95']:.1f} ms (teto {LATENCY_CEILING_MS} ms)")
    adopted = result["ic_inferior"] > 0 and latency["p95"] <= LATENCY_CEILING_MS
    print("Comitê adotado:", adopted)


if __name__ == "__main__":
    main()
