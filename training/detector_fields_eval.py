"""Medida da seleção dos campos nas predições fora da dobra (revisão R4, K2; ET5c, T5c.3).

Escolhe o corte K2 da regra do kvarh no S1_coco: o de maior acerto ponderado nas fotos
do desenvolvimento com display de consumo ciclométrico anotado, por semente, com
empate para o corte maior; vale o da semente 0 (a do modelo final). Relata os erros
(kvarh escolhido como consumo), os ambíguos e, no corte escolhido, a frequência de cada
caso ambíguo entre as fotos que passam pela porta. Só contagens e medidas; resumo em
JSON na pasta de trabalho. Nenhuma foto do teste é lida.
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np

from meter_reader.fields import (AMBIGUOUS_MANY, AMBIGUOUS_NO_CONSUMPTION, AMBIGUOUS_SAME_CLASS,
                                 FieldSelection, select_fields)
from training.detector_eval import IOU_MATCH, iou_matrix, photo_score

CUT_GRID = np.round(np.arange(0.01, 0.951, 0.01), 2)  # grade fixa da R4
CYCLO, KVARH = 2, 3                                   # rótulos do detector
AMBIGUOUS = (AMBIGUOUS_SAME_CLASS, AMBIGUOUS_MANY, AMBIGUOUS_NO_CONSUMPTION)
OUTCOMES = ("acerto", "erro_kvarh", *AMBIGUOUS, "outro")
FOLDER = "S1_coco"
SUMMARY_NAME = "relatorio_campos.json"


def _covers(box, gt_boxes, gt_labels, label: int) -> bool:
    """A caixa tem IoU >= 0,5 com alguma caixa anotada da classe `label`?"""
    gt = np.asarray([b for b, lab in zip(gt_boxes, gt_labels) if lab == label],
                    dtype=float).reshape(-1, 4)
    return len(gt) > 0 and bool((iou_matrix(np.asarray(box, float)[None], gt) >= IOU_MATCH).any())


def photo_outcome(selection: FieldSelection, gt_boxes, gt_labels) -> str:
    """Resultado da seleção numa foto contra o gabarito, na ordem de OUTCOMES."""
    if selection.ambiguity is not None:
        return selection.ambiguity
    if selection.consumption is not None:
        if _covers(selection.consumption, gt_boxes, gt_labels, CYCLO):
            return "acerto"
        if _covers(selection.consumption, gt_boxes, gt_labels, KVARH):
            return "erro_kvarh"
    return "outro"


def outcome_rates(photos, preds, cut: float) -> dict[str, float]:
    """Frações ponderadas de cada resultado, nas fotos dadas, com o corte `cut`."""
    acc, total = dict.fromkeys(OUTCOMES, 0.0), 0.0
    for photo, pred in zip(photos, preds):
        sel = select_fields(pred["boxes"], pred["scores"], pred["labels"], cut)
        acc[photo_outcome(sel, photo.boxes, photo.labels)] += photo.weight
        total += photo.weight
    return {k: v / total for k, v in acc.items()}


def choose_cut(curve: dict[float, dict[str, float]]) -> float:
    """Corte de maior acerto; no empate, o maior (R4, K2)."""
    best = max(r["acerto"] for r in curve.values())
    return max(c for c, r in curve.items() if r["acerto"] == best)


def ambiguity_frequency(photos, preds, cut: float, gate: float) -> dict:
    """Entre as fotos que passam pela porta (escore >= gate): fração ponderada e número
    de fotos de cada caso ambíguo, no corte `cut`."""
    weights, counts, total, n = Counter(), Counter(), 0.0, 0
    for photo, pred in zip(photos, preds):
        if photo_score(pred) < gate:
            continue
        total, n = total + photo.weight, n + 1
        amb = select_fields(pred["boxes"], pred["scores"], pred["labels"], cut).ambiguity
        if amb is not None:
            weights[amb] += photo.weight
            counts[amb] += 1
    return {"fotos_que_passam": n,
            **{a: {"fracao": weights[a] / total, "fotos": counts[a]} for a in AMBIGUOUS}}


def main() -> None:
    from meter_reader.config import get_work_dir
    from training.detector_ablation import SUMMARY_NAME as ABLATION_SUMMARY
    from training.detector_ablation import load_seeds
    from training.detector_dataset import load_detector_photos
    from training.run_detector import OUTPUT_FOLDER

    work_dir = get_work_dir()
    gate = json.loads((work_dir / OUTPUT_FOLDER / ABLATION_SUMMARY)
                      .read_text(encoding="utf-8"))["decisao"]["limiar_porta"]
    photos = [p for p in load_detector_photos() if p.subset == "desenvolvimento"]
    lots = sorted({p.lot for p in photos})
    seeds, _ = load_seeds(work_dir, FOLDER, photos, lots)
    pop = [i for i, p in enumerate(photos) if CYCLO in p.labels]
    print(f"{len(pop)} fotos com display de consumo ciclométrico anotado; porta t = {gate:.4f}")

    per_seed = []
    for s, preds in enumerate(seeds):
        sub_photos, sub_preds = [photos[i] for i in pop], [preds[i] for i in pop]
        curve = {float(c): outcome_rates(sub_photos, sub_preds, float(c)) for c in CUT_GRID}
        cut = choose_cut(curve)
        per_seed.append({"semente": s, "corte": cut, "curva": curve})
        print(f"s{s}: corte escolhido {cut:.2f}; " + ", ".join(
            f"{k} {100 * v:.2f}%" for k, v in curve[cut].items()))
    cut = per_seed[0]["corte"]  # vale o da semente 0 (R4)
    print(f"\nCorte adotado (semente 0): {cut:.2f}")
    for s, r in enumerate(per_seed):
        print(f"  s{s} no corte adotado: " + ", ".join(
            f"{k} {100 * v:.2f}%" for k, v in r["curva"][cut].items()))
    print("\nCurva da semente 0 (a cada 0,05): corte, acerto, erro_kvarh, ambíguos")
    for c in CUT_GRID[4::5]:
        r = per_seed[0]["curva"][float(c)]
        print(f"  {c:.2f}: {100 * r['acerto']:.2f}%, {100 * r['erro_kvarh']:.2f}%, "
              f"{100 * sum(r[a] for a in AMBIGUOUS):.2f}%")
    freq = [ambiguity_frequency(photos, preds, cut, gate) for preds in seeds]
    print("\nCasos ambíguos entre as fotos que passam pela porta, no corte adotado:")
    for s, fr in enumerate(freq):
        print(f"  s{s} ({fr['fotos_que_passam']} fotos): " + ", ".join(
            f"{a} {100 * fr[a]['fracao']:.2f}% ({fr[a]['fotos']} fotos)" for a in AMBIGUOUS))
    out = {"porta": gate, "corte_adotado": cut, "por_semente": per_seed, "ambiguos": freq,
           "fotos_populacao": len(pop)}
    (work_dir / OUTPUT_FOLDER / SUMMARY_NAME).write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Resumo gravado em {SUMMARY_NAME}, na pasta de trabalho.")


if __name__ == "__main__":
    main()
