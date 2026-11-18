"""Leitura única do teste 03/07 pelo detector (protocolo, 1.10 e revisão R4; ET5c, T5c.5).

Modos, a partir da raiz do repositório do produto:
    python -m training.detector_holdout ensaio   (mesmas métricas nas predições fora da
                                                  dobra do desenvolvimento, semente 0;
                                                  nada é gravado)
    python -m training.detector_holdout ler      (leitura única do 03/07 pelo produto, em
                                                  CPU; recusa uma segunda leitura)

Na leitura, as predições e as latências são gravadas antes das métricas; se as métricas
falharem, uma nova chamada as reaproveita sem rodar o modelo de novo. O resultado é
gravado uma vez e nunca recalculado (E44, E68). Saídas: só contagens e medidas.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

from training.detector_eval import (GATE_MAX_LOSS, ap_table, coverage_summary, kish_n,
                                    loss_budget, photo_score, selection_score)
from training.detector_fields_eval import CYCLO, ambiguity_frequency, outcome_rates
from training.detector_report import gate_at, kvarh_confusion, validator2

PRED_NAME = "teste_0307_detector.npz"
RESULT_NAME = "teste_0307_detector.json"
WARMUP = 20            # fotos do desenvolvimento antes de medir a latência (E67)
LATENCY_THREADS = 4    # como na E67 e na seção 1.11


def evaluate(photos, preds, gate: float, cut: float, triage_type: dict[str, str]) -> dict:
    """Métricas da seção 1.10 e da revisão R4 sobre predições alinhadas a `photos`."""
    ap, matches = ap_table(photos, preds)
    has = np.array([p.has_field for p in photos])
    w = np.array([p.weight for p in photos])
    rejected = np.array([photo_score(p) < gate for p in preds])
    porta = gate_at(photos, preds, gate)
    porta["fotos_sem_campo_rejeitadas"] = int((rejected & ~has).sum())
    cyc = [i for i, p in enumerate(photos) if CYCLO in p.labels]
    return {
        "fotos": len(photos), "com_campo": int(has.sum()), "sem_campo": int((~has).sum()),
        "orcamento_perdas": loss_budget(kish_n(w[has])),
        "porta": porta, "meta_cumprida": bool(porta["perda_lim_sup"] <= GATE_MAX_LOSS),
        "ap": ap, "selecao": selection_score(ap),
        "cobertura_campo": coverage_summary(matches).reset_index().to_dict("records"),
        "kvarh_confundido": kvarh_confusion(photos, preds, gate),
        "validador2": validator2(photos, preds, gate, triage_type),
        "regra_kvarh": outcome_rates([photos[i] for i in cyc], [preds[i] for i in cyc], cut),
        "fotos_regra_kvarh": len(cyc),
        "ambiguos": ambiguity_frequency(photos, preds, cut, gate),
    }


def show(out: dict) -> None:
    """Resumo na tela: só contagens e medidas."""
    g = out["porta"]
    print(f"Fotos {out['fotos']} ({out['com_campo']} com campo, {out['sem_campo']} sem); "
          f"orçamento de perdas {out['orcamento_perdas']}")
    print(f"Porta: perda {100 * g['perda']:.3f}% (lim. sup. {100 * g['perda_lim_sup']:.3f}%, "
          f"{g['fotos_com_campo_perdidas']} foto(s)); meta de 0,5% cumprida: {out['meta_cumprida']}; "
          f"cobertura {100 * g['cobertura']:.1f}% ({g['fotos_sem_campo_rejeitadas']} foto(s) sem campo rejeitadas)")
    print("AP ponderada: " + ", ".join(f"{c} {100 * v:.1f}" for c, v in out["ap"].items())
          + f"; escolha {100 * out['selecao']:.2f}")
    print("Cobertura do campo (p5; mediana): " + "; ".join(
        f"{r['classe']} {r['cobertura_p5']}, {r['cobertura_p50']}" for r in out["cobertura_campo"]))
    v2, kv = out["validador2"], out["kvarh_confundido"]["fracao_confundida"]
    print(f"Tipo divergente {100 * v2['diverge']:.2f}%; duas famílias {100 * v2['duas_familias']:.2f}%; "
          f"sem família {100 * v2['sem_familia']:.2f}%; kvarh confundido "
          + ("—" if kv is None else f"{100 * kv:.1f}%"))
    print(f"Regra do kvarh ({out['fotos_regra_kvarh']} fotos): " + ", ".join(
        f"{k} {100 * v:.2f}%" for k, v in out["regra_kvarh"].items()))
    a = out["ambiguos"]
    print(f"Ambíguos entre as {a['fotos_que_passam']} fotos que passam: " + ", ".join(
        f"{k} {100 * v['fracao']:.2f}% ({v['fotos']})" for k, v in a.items() if k != "fotos_que_passam"))
    if "latencia_ms" in out:
        print(f"Latência em CPU ({LATENCY_THREADS} threads, foto a foto): p50 "
              f"{out['latencia_ms']['p50']:.1f} ms, p95 {out['latencia_ms']['p95']:.1f} ms")


def _context():
    from meter_reader.config import get_work_dir
    from training.box_sample import population
    from training.detector_dataset import load_detector_photos

    work_dir = get_work_dir()
    pop = population()
    return work_dir, load_detector_photos(), dict(zip(pop["pseudonimo"], pop["tipo_triagem"]))


def rehearse() -> None:
    """Ensaio: as mesmas métricas nas predições fora da dobra da semente 0 do S1_coco."""
    from training.detector_ablation import SUMMARY_NAME as ABLATION_SUMMARY
    from training.detector_ablation import load_seeds
    from training.detector_fields_eval import FOLDER, SUMMARY_NAME as FIELDS_SUMMARY
    from training.run_detector import OUTPUT_FOLDER

    work_dir, photos, triage_type = _context()
    dev = [p for p in photos if p.subset == "desenvolvimento"]
    gate = json.loads((work_dir / OUTPUT_FOLDER / ABLATION_SUMMARY)
                      .read_text(encoding="utf-8"))["decisao"]["limiar_porta"]
    cut = json.loads((work_dir / OUTPUT_FOLDER / FIELDS_SUMMARY)
                     .read_text(encoding="utf-8"))["corte_adotado"]
    preds = load_seeds(work_dir, FOLDER, dev, sorted({p.lot for p in dev}))[0][0]
    print("Ensaio no desenvolvimento (semente 0, fora da dobra); esperado: escolha 85,74, "
          "perda 0,000%, regra do kvarh com acerto 77,91%")
    show(evaluate(dev, preds, gate, cut, triage_type))


def read_test() -> None:
    """Leitura única do 03/07 pelo detector do produto, em CPU."""
    import torch

    from meter_reader.detector import PRODUCT_DETECTOR_SHA256, load_detector
    from training.build_detector import ARTIFACT_NAME
    from training.detector_train import unflatten_predictions
    from training.evaluation import _product_commit
    from training.run_detector import FINAL_FOLDER, OUTPUT_FOLDER, save_part

    work_dir, photos, triage_type = _context()
    folder = work_dir / OUTPUT_FOLDER
    result_path, pred_path = folder / RESULT_NAME, folder / PRED_NAME
    if result_path.exists():
        raise SystemExit("O 03/07 já foi lido pelo detector: o resultado não é recalculado.")
    product = load_detector(work_dir / FINAL_FOLDER / ARTIFACT_NAME, PRODUCT_DETECTOR_SHA256)
    test = [p for p in photos if p.subset == "teste"]

    if pred_path.exists():  # retomada: as predições já foram feitas uma vez
        with np.load(pred_path) as data:
            meta = json.loads(str(data["meta"]))
            codes, preds = unflatten_predictions(data)
        if codes != [p.pseudonym for p in test]:
            raise SystemExit("As predições gravadas não correspondem às fotos do teste.")
        print("Predições do 03/07 já gravadas: só as métricas são calculadas.", flush=True)
    else:
        torch.set_num_threads(LATENCY_THREADS)
        for p in [p for p in photos if p.subset == "desenvolvimento"][:WARMUP]:
            product.predict(p.path)  # aquecimento, fora da medição
        preds, times = [], []
        for p in test:
            start = time.perf_counter()
            r = product.predict(p.path)
            times.append(1000 * (time.perf_counter() - start))
            preds.append({"boxes": r.boxes, "scores": r.scores, "labels": r.labels})
        meta = {"latencias_ms": times, "sha_artefato": PRODUCT_DETECTOR_SHA256,
                "commit_produto": _product_commit(), "threads": LATENCY_THREADS}
        save_part(pred_path, [p.pseudonym for p in test], preds, meta)
        print(f"Predições de {len(test)} fotos gravadas.", flush=True)

    out = evaluate(test, preds, product.gate_threshold, product.field_cut, triage_type)
    lat = np.array(meta["latencias_ms"])
    out["latencia_ms"] = {"p50": float(np.median(lat)), "p95": float(np.quantile(lat, 0.95))}
    out.update({"limiar_porta": product.gate_threshold, "corte_campos": product.field_cut,
                "sha_artefato": meta["sha_artefato"], "commit_produto": meta["commit_produto"]})
    tmp = result_path.with_name(result_path.name + ".tmp")
    tmp.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, result_path)
    show(out)
    print(f"Resultado gravado em {RESULT_NAME}, na pasta de trabalho; não será recalculado.")


if __name__ == "__main__":  # obrigatório no Windows
    mode = sys.argv[1:]
    if mode == ["ensaio"]:
        rehearse()
    elif mode == ["ler"]:
        read_test()
    else:
        raise SystemExit("Uso: python -m training.detector_holdout {ensaio | ler}")
