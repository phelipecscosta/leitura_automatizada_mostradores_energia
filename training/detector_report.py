"""Relatório do desenvolvimento e aplicação das regras do protocolo do detector (ET5b, T5b.7).

Lê as predições fora da dobra gravadas pelo training.run_detector e aplica, sem
nenhum parâmetro novo, as regras do protocolo (exploracao/detector_README.md,
pesquisa): métrica de escolha e AP por classe (1.6), segunda porta (1.7),
Validador 2 (1.8) e regra de escolha (1.9), com as sementes da revisão R2.
Saídas: só contagens, medidas e pesos; grava um resumo em JSON na pasta de trabalho.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from meter_reader.detector import DETECTOR_CLASSES, meter_family
from training.detector_dataset import BOXES_SHA256, DetectorPhoto
from training.detector_eval import (GATE_MAX_LOSS, IOU_MATCH, SELECTION_CLASSES, ap_table,
                                    bootstrap_seed_mean_diff, choose_gate_threshold,
                                    coverage_summary, gate_curve, iou_matrix, photo_score,
                                    selection_score, weighted_cp_upper)
from training.detector_train import unflatten_predictions

SEEDS = (0, 1, 2)
DELTA = 0.05       # Δ3: 5 pontos de AP (protocolo, 1.9)
DELTA_GPU = 0.10   # G2: o dobro do Δ
MAX_P95_MS = 200   # L1
# Latência p95 em CPU medida antes do treino (protocolo, 1.11)
LATENCY_P95_MS = {"S1": 31.5, "F1": 102.4, "F2": 358.6}
CPU_CANDIDATES, GPU_CANDIDATE = ("S1", "F1"), "F2"   # na ordem de simplicidade
SUMMARY_NAME = "relatorio_desenvolvimento.json"
# Corte de confiança das partes gravadas antes da revisão R3, que não o registravam:
# o padrão de cada arquitetura no torchvision
LEGACY_SCORE_THRESH = {"S1": 0.001, "F1": 0.05, "F2": 0.05}


@dataclass
class SeedResult:
    ap: dict[str, float]
    selection: float
    threshold: float | None
    matches: dict = field(repr=False)
    preds: list[dict] = field(repr=False)


def align(codes: list[str], preds: list[dict], photos: list[DetectorPhoto]) -> list[dict]:
    """Predições na ordem de `photos`; exige cada foto exatamente uma vez."""
    by_code = dict(zip(codes, preds))
    if len(by_code) != len(codes) or set(by_code) != {p.pseudonym for p in photos}:
        raise ValueError("As partes não cobrem as fotos do desenvolvimento exatamente uma vez.")
    return [by_code[p.pseudonym] for p in photos]


def seed_result(photos: list[DetectorPhoto], preds: list[dict]) -> SeedResult:
    """AP por classe, métrica de escolha e limiar da porta de uma semente."""
    ap, matches = ap_table(photos, preds)
    curve = gate_curve([photo_score(p) for p in preds], [p.has_field for p in photos],
                       [p.weight for p in photos])
    return SeedResult(ap, selection_score(ap), choose_gate_threshold(curve), matches, preds)


def gate_at(photos: list[DetectorPhoto], preds: list[dict], t: float) -> dict:
    """Perda, limite superior e cobertura da segunda porta no limiar t."""
    has = np.array([p.has_field for p in photos])
    w = np.array([p.weight for p in photos])
    rejected = np.array([photo_score(p) < t for p in preds])
    return {"perda": float((w * (rejected & has)).sum() / w[has].sum()),
            "perda_lim_sup": weighted_cp_upper(rejected[has], w[has]),
            "cobertura": float((w * (rejected & ~has)).sum() / w[~has].sum()),
            "fotos_com_campo_perdidas": int((rejected & has).sum())}


def kvarh_confusion(photos: list[DetectorPhoto], preds: list[dict], t: float) -> dict:
    """Caixas kvarh anotadas cobertas (IoU >= 0,5) por uma caixa prevista de display de
    consumo ciclométrico acima de t: o erro que o leitor e os validadores teriam de segurar."""
    kvarh, cic = DETECTOR_CLASSES.index("display_kvarh") + 1, \
        DETECTOR_CLASSES.index("display_ciclometrico") + 1
    total = confused = 0.0
    for photo, pred in zip(photos, preds):
        gt = np.array([b for b, lab in zip(photo.boxes, photo.labels) if lab == kvarh]).reshape(-1, 4)
        keep = (np.asarray(pred["labels"]) == cic) & (np.asarray(pred["scores"]) >= t)
        total += photo.weight * len(gt)
        if len(gt) and keep.any():
            hit = (iou_matrix(gt, np.asarray(pred["boxes"])[keep]) >= IOU_MATCH).any(axis=1)
            confused += photo.weight * hit.sum()
    return {"kvarh_anotadas_peso": total, "fracao_confundida": confused / total if total else None}


def validator2(photos: list[DetectorPhoto], preds: list[dict], t: float,
               triage_type: dict[str, str]) -> dict:
    """Família vista pelo detector contra o tipo da triagem, nas fotos que passam pela porta
    (protocolo, 1.8): frações ponderadas de cada situação."""
    kinds = {"sem_familia": 0.0, "duas_familias": 0.0, "duas_familias_diverge": 0.0,
             "diverge": 0.0}
    total = 0.0
    for photo, pred in zip(photos, preds):
        if photo_score(pred) < t:
            continue
        labels, scores = np.asarray(pred["labels"]), np.asarray(pred["scores"])
        above = scores >= t
        families = {meter_family([lab], [1.0], 0.0) for lab in labels[above]} - {None}
        family = meter_family(labels, scores, t)
        diverges = family is not None and family != triage_type[photo.pseudonym]
        total += photo.weight
        kinds["sem_familia"] += photo.weight * (family is None)
        kinds["duas_familias"] += photo.weight * (len(families) == 2)
        kinds["duas_familias_diverge"] += photo.weight * (len(families) == 2 and diverges)
        kinds["diverge"] += photo.weight * diverges
    return {k: v / total for k, v in kinds.items()}


def summarize(name: str, photos: list[DetectorPhoto], seeds: list[list[dict]],
              triage_type: dict[str, str]) -> tuple[dict, list[SeedResult]]:
    """Resumo de um candidato com as 3 sementes (revisão R2)."""
    results = [seed_result(photos, preds) for preds in seeds]
    thresholds = [r.threshold for r in results]
    viable = all(t is not None for t in thresholds)
    out = {"candidato": name, "viavel": viable,
           "ap_media": {c: float(np.mean([r.ap[c] for r in results])) for c in DETECTOR_CLASSES},
           "selecao_media": float(np.mean([r.selection for r in results])),
           "selecao_por_semente": [r.selection for r in results],
           "limiares_por_semente": thresholds, "latencia_p95_ms": LATENCY_P95_MS[name]}
    if viable:
        t = min(thresholds)  # o lado seguro (R2)
        out["limiar"] = t
        out["porta_por_semente"] = [gate_at(photos, r.preds, t) for r in results]
        out["kvarh_por_semente"] = [kvarh_confusion(photos, r.preds, t) for r in results]
        out["validador2_por_semente"] = [validator2(photos, r.preds, t, triage_type)
                                         for r in results]
    return out, results


def choose(summaries: dict[str, dict], results: dict[str, list[SeedResult]],
           photos: list[DetectorPhoto]) -> dict:
    """Regra de escolha (protocolo, 1.9): perfil CPU pelo mais simples dentro de Δ3 do
    melhor; perfil GPU só com ganho de G2 sobre o vencedor do CPU."""
    eligible = [c for c in CPU_CANDIDATES
                if summaries[c]["viavel"] and summaries[c]["latencia_p95_ms"] <= MAX_P95_MS]
    if not eligible:
        return {"cpu": None, "gpu": None, "motivo": "nenhum candidato do perfil CPU elegível"}
    best = max(eligible, key=lambda c: summaries[c]["selecao_media"])
    decision = {"melhor_cpu": best, "comparacoes": {}}
    winner = best
    for cand in eligible:  # na ordem de simplicidade: o primeiro não inferior vence
        if cand == best:
            winner = cand
            break
        diffs = bootstrap_seed_mean_diff(photos, [r.matches for r in results[cand]],
                                         [r.matches for r in results[best]])
        low = float(np.quantile(diffs, 0.025))
        decision["comparacoes"][f"{cand}-{best}"] = {"diferenca_media": float(diffs.mean()),
                                                     "ic95_inferior": low}
        if low > -DELTA:
            winner = cand
            break
    decision["cpu"] = winner
    gpu = None
    if summaries[GPU_CANDIDATE]["viavel"]:
        diffs = bootstrap_seed_mean_diff(photos, [r.matches for r in results[GPU_CANDIDATE]],
                                         [r.matches for r in results[winner]])
        low = float(np.quantile(diffs, 0.025))
        decision["comparacoes"][f"{GPU_CANDIDATE}-{winner}"] = {
            "diferenca_media": float(diffs.mean()), "ic95_inferior": low}
        gpu = GPU_CANDIDATE if low >= DELTA_GPU else None
    decision["gpu"] = gpu
    return decision


def main() -> None:
    from meter_reader.config import get_work_dir
    from training.box_sample import population
    from training.detector_dataset import load_detector_photos
    from training.run_detector import OUTPUT_FOLDER, part_path

    work_dir = get_work_dir()
    photos = [p for p in load_detector_photos() if p.subset == "desenvolvimento"]
    lots = sorted({p.lot for p in photos})
    pop = population()
    triage_type = dict(zip(pop["pseudonimo"], pop["tipo_triagem"]))

    summaries, results, formas, cuts = {}, {}, set(), set()
    for name in (*CPU_CANDIDATES, GPU_CANDIDATE):
        seeds = []
        for seed in SEEDS:
            codes, preds = [], []
            for lot in lots:
                with np.load(part_path(work_dir, name, lot, seed)) as data:
                    meta = json.loads(str(data["meta"]))
                    c, p = unflatten_predictions(data)
                if meta["sha_caixas"] != BOXES_SHA256:
                    raise SystemExit(f"{name}: parte treinada com outro gabarito.")
                formas.add(meta["sha_formas"])
                cuts.add((name, meta.get("corte_confianca", LEGACY_SCORE_THRESH[name])))
                codes += c
                preds += p
            seeds.append(align(codes, preds, photos))
        summaries[name], results[name] = summarize(name, photos, seeds, triage_type)
    if len(formas) != 1:
        raise SystemExit("As partes usam arquivos de formas diferentes.")
    by_name = {}
    for name, cut in cuts:
        by_name.setdefault(name, set()).add(cut)
    if any(len(v) != 1 for v in by_name.values()):
        raise SystemExit("Partes de um mesmo candidato com cortes de confiança diferentes.")
    for name, (cut,) in by_name.items():
        summaries[name]["corte_confianca"] = cut
    decision = choose(summaries, results, photos)

    pd.set_option("display.width", 200)
    table = pd.DataFrame({n: {**{c: round(100 * s["ap_media"][c], 2) for c in DETECTOR_CLASSES},
                              "selecao": round(100 * s["selecao_media"], 2),
                              "viavel": s["viavel"], "limiar": s.get("limiar"),
                              "corte": s["corte_confianca"]}
                          for n, s in summaries.items()}).T
    print("AP ponderada média das 3 sementes (pontos de 0 a 100) e limiar da porta:")
    print(table, end="\n\n")
    for n, s in summaries.items():
        print(f"{n}: seleção por semente {[round(100 * v, 2) for v in s['selecao_por_semente']]}; "
              f"limiares por semente {s['limiares_por_semente']}")
        if s["viavel"]:
            for i, g in enumerate(s["porta_por_semente"]):
                v2 = s["validador2_por_semente"][i]
                kv = s["kvarh_por_semente"][i]["fracao_confundida"]
                print(f"  s{i}: perda {100 * g['perda']:.3f}% (lim. sup. {100 * g['perda_lim_sup']:.3f}%, "
                      f"{g['fotos_com_campo_perdidas']} foto(s)); cobertura {100 * g['cobertura']:.1f}%; "
                      f"kvarh confundido {100 * kv:.1f}%; tipo divergente {100 * v2['diverge']:.2f}%, "
                      f"duas famílias {100 * v2['duas_familias']:.2f}% "
                      f"(divergindo {100 * v2['duas_familias_diverge']:.2f}%), "
                      f"sem família {100 * v2['sem_familia']:.2f}%")
    print("\nCobertura do campo (semente 0) do vencedor do perfil CPU:")
    winner = decision.get("cpu")
    if winner:
        print(coverage_summary(results[winner][0].matches))
    print("\nDecisão pela regra (protocolo, 1.9):", json.dumps(decision, ensure_ascii=False))
    out = {"resumos": summaries, "decisao": decision, "meta_perda": GATE_MAX_LOSS,
           "classes_da_escolha": list(SELECTION_CLASSES)}
    (work_dir / OUTPUT_FOLDER / SUMMARY_NAME).write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Resumo gravado em {SUMMARY_NAME}, na pasta de trabalho.")


if __name__ == "__main__":
    main()
