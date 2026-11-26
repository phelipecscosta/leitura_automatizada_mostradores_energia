"""Paridade e latência do leitor do produto (ET6c, T6c.5; decisões Pa-1 e Lt-1).

Paridade (Pa-1), na amostra L-a (600 displays; o número de série vem das mesmas
fotos), com o artefato final:
- produto em CPU (meter_reader.reader_model) x caminho de medida do treino em GPU
  (as funções usadas na R16 a R18 e na T6c.3);
- produto em CPU x produto em GPU.
Critério: as decisões são as mesmas em 100% dos recortes (texto lido, faixa muito
baixa, D1 e decisão da função). Relata também a maior diferença numérica e a menor
folga até cada limiar, por parte da decisão. A GPU roda com o TF32 ligado (o padrão
do cuDNN, com que os limiares foram medidos) e desligado (diagnóstico).

Latência (Lt-1), como na E67: CPU com 4 threads, lote de 1, com o preparo da entrada;
220 fotos da amostra numa permutação com semente 0 (20 de aquecimento e 200
medidas); por foto, o display e o número de série (quando existem).
Grava `paridade_latencia_leitor.json` na pasta de trabalho. Só números.

Uso, a partir da raiz do repositório do produto:
    python -m training.reader_product_parity
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

OUT_NAME = Path("experimentos") / "leitor" / "paridade_latencia_leitor.json"
CLASSES = ("display_digital", "display_ciclometrico", "numero_serie")
WARMUP, MEASURED, THREADS = 20, 200, 4
PARTS = ("texto", "faixa_muito_baixa", "d1", "funcao")


def set_tf32(enabled: bool) -> None:
    """TF32 da GPU (padrão do cuDNN nas convoluções da série RTX 30): liga ou desliga."""
    import torch

    torch.backends.cudnn.allow_tf32 = enabled
    torch.backends.cuda.matmul.allow_tf32 = enabled


def decision(text: str, log_conf: float, function: dict | None, two_lines: bool | None,
             log_low: float, function_threshold: float) -> tuple:
    """O que o produto decide com uma saída do leitor: texto, faixa muito baixa, D1 e
    a decisão da função (aceita, coluna, motivo)."""
    from meter_reader.validation import function_decision

    low = not text or log_conf < log_low
    f = None
    if function is not None:
        d = function_decision(function, function_threshold)
        f = (d.accepted, d.column, d.reason)
    return text, low, two_lines, f


def compare(a: list[dict], b: list[dict], log_low: float, t_func: float) -> dict:
    """Decisões divergentes e maiores diferenças entre duas listas de saídas."""
    da = [decision(**x, log_low=log_low, function_threshold=t_func) for x in a]
    db = [decision(**y, log_low=log_low, function_threshold=t_func) for y in b]
    diverge = [i for i, (x, y) in enumerate(zip(da, db)) if x != y]
    parts = {name: sum(x[k] != y[k] for x, y in zip(da, db))
             for k, name in enumerate(PARTS)}
    conf = [abs(x["log_conf"] - y["log_conf"]) for x, y in zip(a, b)
            if math.isfinite(x["log_conf"]) and math.isfinite(y["log_conf"])]
    func = [max(abs(x["function"][k] - y["function"][k]) for k in x["function"])
            for x, y in zip(a, b) if x["function"] is not None]
    return {"itens": len(a), "decisoes_divergentes": len(diverge), "por_parte": parts,
            "max_dif_log_conf": max(conf, default=0.0), "max_dif_prob_funcao": max(func, default=0.0)}


def _as_dict(o) -> dict:
    return {"text": o.text, "log_conf": o.log_conf, "function": o.function,
            "two_lines": o.two_lines}


def _training_path(model, crops: list, field_class: str, device: str) -> list[dict]:
    """O caminho de medida do treino (R16 a R18 e T6c.3) com os pesos do artefato."""
    from meter_reader.reader_text import FUNCTION_CLASSES
    from training.reader_ensemble import log_mean_exp
    from training.reader_eval import sequence_logprobs
    from training.reader_function_rule import _committee_read, _outputs

    pairs = list(zip(model.trunks, model.heads))
    whole = [_outputs(t, h, crops, device) for t, h in pairs]
    fired = [False] * len(crops)
    upper = None
    if field_class == "display_digital":
        from meter_reader.reader_input import upper_band
        logit = np.mean([o["duas"] for o in whole], axis=0)
        fired = list(logit >= model.thresholds["d1_logit"])
        upper = [_outputs(t, h, [upper_band(c) for c in crops], device) for t, h in pairs]
    out = []
    for i in range(len(crops)):
        src = upper if fired[i] else whole
        lps = [o["texto"][i] for o in src]
        text = _committee_read(lps)
        conf = log_mean_exp([sequence_logprobs(lp, [text])[0] for lp in lps]) \
            if text else -math.inf
        func = None
        if field_class == "display_digital":
            func = dict(zip(FUNCTION_CLASSES, np.mean([o["funcao"][i] for o in src], axis=0)
                            .tolist()))
        out.append({"text": text, "log_conf": conf, "function": func,
                    "two_lines": bool(fired[i]) if field_class == "display_digital" else None})
    return out


def margins(outs: list[dict], log_low: float, t_func: float) -> dict:
    """Menor folga até o limiar inferior e até o limiar da função."""
    from meter_reader.reader_text import CONSUMPTION_CLASSES

    conf = [abs(o["log_conf"] - log_low) for o in outs if math.isfinite(o["log_conf"])]
    func = [abs(sum(o["function"][c] for c in CONSUMPTION_CLASSES) - t_func)
            for o in outs if o["function"] is not None]
    return {"folga_limiar_inferior": min(conf, default=math.inf),
            "folga_limiar_funcao": min(func, default=math.inf)}


def main() -> None:
    import torch
    from PIL import Image

    from meter_reader.config import get_work_dir
    from meter_reader.reader_model import load_reader
    from training.reader_bands import LEGIBILITY_NAME
    from training.reader_cache import CACHE_DIR, CACHE_NAME, crop_key, load_cache
    from training.reader_labels import CROPS_SHA256

    work_dir = get_work_dir()
    artifact = work_dir / "modelos" / "leitor" / "leitor_l2-12.pt"
    judged = pd.read_csv(work_dir / LEGIBILITY_NAME, dtype=str, keep_default_na=False)
    cache = load_cache(work_dir / CACHE_DIR / CACHE_NAME, CROPS_SHA256)
    cpu = load_reader(artifact, device="cpu")
    gpu = load_reader(artifact, device="cuda")
    log_low, t_func = cpu.log_low, cpu.function_threshold

    # Paridade (Pa-1). A L-a só tem displays: o número de série vem das mesmas fotos.
    keys = set(cache.files)
    codes = {c: list(judged.loc[judged["classe"] == c, "pseudonimo"]) for c in CLASSES[:2]}
    codes["numero_serie"] = [p for p in sorted(judged["pseudonimo"].unique())
                             if crop_key(p, "numero_serie") in keys]
    result = {"paridade": {}}
    all_cpu = []
    for c in CLASSES:
        crops = [Image.fromarray(cache[crop_key(p, c)]) for p in codes[c]]
        if not crops:
            continue
        prod_cpu = [_as_dict(o) for o in cpu.read(crops, c)]
        all_cpu += prod_cpu
        result["paridade"][c] = {"d1_disparou": int(sum(bool(o["two_lines"]) for o in prod_cpu))}
        for tf32 in (True, False):
            set_tf32(tf32)
            prod_gpu = [_as_dict(o) for o in gpu.read(crops, c)]
            train_gpu = _training_path(gpu, crops, c, "cuda")
            tag = "tf32_ligado" if tf32 else "tf32_desligado"
            result["paridade"][c][tag] = {
                "produto_cpu_x_treino_gpu": compare(prod_cpu, train_gpu, log_low, t_func),
                "produto_cpu_x_produto_gpu": compare(prod_cpu, prod_gpu, log_low, t_func)}
        print(c, json.dumps(result["paridade"][c]), flush=True)
    set_tf32(True)
    result["folgas"] = margins(all_cpu, log_low, t_func)
    print("Folgas:", result["folgas"])

    # Latência (Lt-1)
    torch.set_num_threads(THREADS)
    photos = []
    for p in sorted(judged["pseudonimo"].unique()):
        display = next((c for c in CLASSES[:2] if crop_key(p, c) in keys), None)
        if display is not None:
            photos.append((p, display, crop_key(p, "numero_serie") in keys))
    order = np.random.default_rng(0).permutation(len(photos))[:WARMUP + MEASURED]
    times = []
    for n, k in enumerate(order):
        p, display, has_serial = photos[k]
        crop = Image.fromarray(cache[crop_key(p, display)])
        serial = Image.fromarray(cache[crop_key(p, "numero_serie")]) if has_serial else None
        start = time.perf_counter()
        cpu.read([crop], display)
        if serial is not None:
            cpu.read([serial], "numero_serie")
        if n >= WARMUP:
            times.append((time.perf_counter() - start) * 1000)
    result["latencia_ms"] = {"threads": THREADS, "fotos": len(times),
                             "p50": float(np.percentile(times, 50)),
                             "p95": float(np.percentile(times, 95)),
                             "media": float(np.mean(times))}
    print("Latência por foto (ms):", {k: round(v, 1) if isinstance(v, float) else v
                                      for k, v in result["latencia_ms"].items()})
    (work_dir / OUT_NAME).write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
