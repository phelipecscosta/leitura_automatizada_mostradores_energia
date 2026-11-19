"""Margem do recorte do leitor (T6a.2).

Mede quanto a caixa que o leitor recebe precisa crescer para conter a caixa
anotada: caixas escolhidas pelo select_fields (corte adotado na E86), nas
predições fora da dobra da semente 0 do S1_coco, no desenvolvimento. Margem por
classe e por eixo pelo p99 ponderado (M-b). Os números do teste 03/07 não são
usados (E68, E88). Saídas: só contagens e medidas.
"""

from __future__ import annotations

import json

import numpy as np

from meter_reader.fields import select_fields
from training.detector_eval import iou_inter, iou_matrix

CLASSES = {"display_digital": 1, "display_ciclometrico": 2, "numero_serie": 4}
KVARH = 3
MATCH_IOU = 0.5   # o mesmo critério de acerto do detector (D1 c)
QUANTILE = 0.99   # M-b
SEED = 0          # a semente do modelo final e do corte adotado
SUMMARY_NAME = "relatorio_margem_recorte.json"


def expansion_needed(pred, gt) -> np.ndarray:
    """Fração da largura (esquerda, direita) e da altura (topo, base) da caixa
    prevista que falta para ela conter a anotada; 0 nos lados já cobertos."""
    px0, py0, px1, py1 = (float(v) for v in pred)
    gx0, gy0, gx1, gy1 = (float(v) for v in gt)
    w, h = px1 - px0, py1 - py0
    if w <= 0 or h <= 0:
        raise ValueError("Caixa prevista sem área.")
    need = np.array([(px0 - gx0) / w, (gx1 - px1) / w, (py0 - gy0) / h, (gy1 - py1) / h])
    return np.clip(need, 0.0, None)


def weighted_quantile(values, weights, q: float) -> float:
    """Quantil ponderado: o menor valor cuja fração acumulada dos pesos alcança q."""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if len(v) == 0:
        return float("nan")
    order = np.argsort(v, kind="stable")
    v, w = v[order], w[order]
    cum = np.cumsum(w) / w.sum()
    idx = min(int(np.searchsorted(cum, q, side="left")), len(v) - 1)
    return float(v[idx])


def _best_match(box, gt_boxes: np.ndarray):
    """A anotada de maior IoU com a caixa, se o IoU for >= MATCH_IOU; senão None."""
    if len(gt_boxes) == 0:
        return None
    ious = iou_matrix(np.asarray(box, dtype=float).reshape(1, 4), gt_boxes)[0]
    j = int(np.argmax(ious))
    return gt_boxes[j] if ious[j] >= MATCH_IOU else None


def collect(photos, preds: list[dict], cut: float):
    """Por classe: expansão necessária (n, 4) e pesos das caixas casadas; e, para o
    consumo ciclométrico, a caixa prevista com os kvarh anotados da foto."""
    needs = {c: [] for c in CLASSES}
    weights = {c: [] for c in CLASSES}
    cyc = []
    for photo, pred in zip(photos, preds):
        sel = select_fields(pred["boxes"], pred["scores"], pred["labels"], cut)
        gt = np.asarray(photo.boxes, dtype=float).reshape(-1, 4)
        lab = np.asarray(photo.labels)
        chosen = []
        if sel.consumption is not None:
            cls = "display_digital" if sel.family == "digital" else "display_ciclometrico"
            chosen.append((cls, sel.consumption))
        if sel.serial is not None:
            chosen.append(("numero_serie", sel.serial))
        for cls, box in chosen:
            match = _best_match(box, gt[lab == CLASSES[cls]])
            if match is None:
                continue  # a caixa escolhida não é o campo anotado: não mede a margem
            needs[cls].append(expansion_needed(box, match))
            weights[cls].append(photo.weight)
            if cls == "display_ciclometrico":
                cyc.append((np.asarray(box, dtype=float), gt[lab == KVARH], photo.weight))
    return ({c: np.array(v, dtype=float).reshape(-1, 4) for c, v in needs.items()},
            {c: np.array(v, dtype=float) for c, v in weights.items()}, cyc)


def margins(needs: dict, weights: dict, q: float = QUANTILE) -> dict:
    """Margem por lado, por classe: (horizontal, vertical), o quantil ponderado do
    maior lado de cada eixo (a margem é aplicada igual nos dois lados)."""
    return {c: (weighted_quantile(n[:, :2].max(axis=1), weights[c], q),
                weighted_quantile(n[:, 2:].max(axis=1), weights[c], q))
            for c, n in needs.items()}


def expand(box, mh: float, mv: float) -> np.ndarray:
    """Caixa com a margem mh da largura e mv da altura em cada lado."""
    x0, y0, x1, y1 = (float(v) for v in box)
    w, h = x1 - x0, y1 - y0
    return np.array([x0 - mh * w, y0 - mv * h, x1 + mh * w, y1 + mv * h])


def kvarh_touched(cyc, mh: float, mv: float) -> float:
    """Fração ponderada dos recortes de consumo ciclométrico que, com a margem,
    tocam algum display de kvarh anotado (verificação da V-b)."""
    if not cyc:
        return float("nan")
    hit = np.array([any(iou_inter(expand(b, mh, mv), k) > 0 for k in kv) for b, kv, _ in cyc])
    w = np.array([c[2] for c in cyc])
    return float((w * hit).sum() / w.sum())

def dev_predictions():
    """Fotos anotadas do desenvolvimento, predições fora da dobra da semente SEED do
    S1_coco (na ordem das fotos) e o corte adotado na E86."""
    from meter_reader.config import get_work_dir
    from training.detector_ablation import ABLATION
    from training.detector_dataset import BOXES_SHA256, load_detector_photos
    from training.detector_fields_eval import SUMMARY_NAME as FIELDS_SUMMARY
    from training.detector_report import align
    from training.detector_train import unflatten_predictions
    from training.run_detector import OUTPUT_FOLDER, part_path

    work_dir = get_work_dir()
    photos = [p for p in load_detector_photos() if p.subset == "desenvolvimento"]
    cut = json.loads((work_dir / OUTPUT_FOLDER / FIELDS_SUMMARY)
                     .read_text(encoding="utf-8"))["corte_adotado"]
    codes, preds = [], []
    for lot in sorted({p.lot for p in photos}):
        with np.load(part_path(work_dir, ABLATION, lot, SEED)) as data:
            if json.loads(str(data["meta"]))["sha_caixas"] != BOXES_SHA256:
                raise SystemExit("Parte treinada com outro gabarito.")
            c, p = unflatten_predictions(data)
        codes += c
        preds += p
    return photos, align(codes, preds, photos), cut

def main() -> None:
    from meter_reader.config import get_work_dir
    from training.detector_ablation import ABLATION

    work_dir = get_work_dir()
    photos, preds, cut = dev_predictions()

    needs, weights, cyc = collect(photos, preds, cut)
    m = margins(needs, weights)
    print(f"{ABLATION}, semente {SEED}, corte {cut:.2f}, quantil {QUANTILE}")
    summary = {"corte": cut, "quantil": QUANTILE, "classes": {}}
    for cls, n in needs.items():
        h, v, w = n[:, :2].max(axis=1), n[:, 2:].max(axis=1), weights[cls]
        row = {"casadas": len(n),
               "sem_expansao": float(np.average((h == 0) & (v == 0), weights=w)),
               "horizontal_p50_p95_p99": [weighted_quantile(h, w, 0.5),
                                          weighted_quantile(h, w, 0.95), m[cls][0]],
               "vertical_p50_p95_p99": [weighted_quantile(v, w, 0.5),
                                        weighted_quantile(v, w, 0.95), m[cls][1]]}
        summary["classes"][cls] = row
        print(f"{cls}: {row['casadas']} casadas; sem expansão {100 * row['sem_expansao']:.1f}%; "
              "horizontal p50/p95/p99 " + "/".join(f"{x:.3f}" for x in row["horizontal_p50_p95_p99"])
              + "; vertical p50/p95/p99 " + "/".join(f"{x:.3f}" for x in row["vertical_p50_p95_p99"]))
    mh, mv = m["display_ciclometrico"]
    summary["kvarh_tocado"] = kvarh_touched(cyc, mh, mv)
    print(f"Consumo ciclométrico com a margem tocando um kvarh anotado: "
          f"{100 * summary['kvarh_tocado']:.2f}% de {len(cyc)} recortes")
    out_dir = work_dir / "experimentos" / "leitor"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / SUMMARY_NAME).write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                        encoding="utf-8")


if __name__ == "__main__":
    main()
