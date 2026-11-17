"""Avaliação comum do detector (ET5b; protocolo em exploracao/detector_README.md).

T5b.2 (D3 e D4): fotos com campo, tamanho efetivo de Kish, limite superior de
Clopper-Pearson ponderado (Korn e Graubard, 1998) e orçamento de perdas.
T5b.5 (seções 1.6 a 1.9): casamento das caixas, AP ponderada, cobertura do
campo, bootstrap pareado da métrica de escolha e curva da segunda porta.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import beta

from training.box_labels import Box
from training.box_stats import TINY_SIDE
from training.triage_eval import ALPHA

GATE_MAX_LOSS = 0.005        # D3 b 1: perda da segunda porta <= 0,5% pelo limite superior de 95%
MIN_FIELD_SIDE = TINY_SIDE   # D3 a ii: campo com os dois lados >= 4 px (medido na T5b.1)


def photos_with_field(boxes: Iterable[Box], min_side: float = MIN_FIELD_SIDE) -> set[str]:
    """Pseudônimos com ao menos uma caixa (de qualquer classe) com os dois lados >= min_side."""
    return {b.pseudonym for b in boxes if min(b.x1 - b.x0, b.y1 - b.y0) >= min_side}


def kish_n(weights: np.ndarray) -> float:
    """Tamanho efetivo de Kish: (soma dos pesos)² ÷ soma dos quadrados dos pesos."""
    w = np.asarray(weights, dtype=float)
    return float(w.sum() ** 2 / (w ** 2).sum()) if len(w) else 0.0


def _upper(k: float, n: float, alpha: float) -> float:
    """Clopper-Pearson unilateral com k e n reais (quantil da Beta)."""
    return 1.0 if k >= n else float(beta.ppf(1 - alpha, k + 1, n - k))


def weighted_cp_upper(lost: np.ndarray, weights: np.ndarray, alpha: float = ALPHA) -> float:
    """Limite superior da taxa ponderada: taxa × tamanho efetivo faz o papel das perdas (D4 a)."""
    lost, w = np.asarray(lost, dtype=bool), np.asarray(weights, dtype=float)
    n_eff = kish_n(w)
    rate = float((w * lost).sum() / w.sum())
    return _upper(rate * n_eff, n_eff, alpha)


def loss_budget(n_eff: float, max_loss: float = GATE_MAX_LOSS, alpha: float = ALPHA) -> int:
    """Maior número de perdas (em fotos efetivas) cujo limite superior cumpre a meta;
    -1 se nem zero perdas cumpre."""
    k = 0
    while _upper(k, n_eff, alpha) <= max_loss:
        k += 1
    return k - 1

# ---------------------------------------------------------------------
# Localização (protocolo, 1.6): casamento, AP ponderada e cobertura do campo
# ---------------------------------------------------------------------

IOU_MATCH = 0.5        # D1 c: acerto com IoU >= 0,5 (PASCAL VOC)
MAX_DETECTIONS = 100   # caixas previstas por foto entram na AP, pelas maiores confianças (COCO)
RECALL_POINTS = np.linspace(0.0, 1.0, 101)  # 101 pontos de revocação (COCO)
SELECTION_CLASSES = ("display_digital", "display_ciclometrico", "numero_serie")  # D2 a
N_BOOT = 2000          # bootstrap pareado por foto (protocolo, 1.9)
BOOT_SEED = 0


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU entre caixas (x0, y0, x1, y1): (n, 4) × (m, 4) -> (n, m)."""
    a, b = np.asarray(a, dtype=float).reshape(-1, 4), np.asarray(b, dtype=float).reshape(-1, 4)
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter)


def match_photo(pred_boxes: np.ndarray, pred_scores: np.ndarray, gt_boxes: np.ndarray,
                iou: float = IOU_MATCH) -> tuple[np.ndarray, np.ndarray]:
    """Casamento guloso de uma classe numa foto, da maior confiança para a menor.

    Devolve (acerto de cada prevista, índice da prevista casada com cada anotada
    ou -1). Cada anotada casa com no máximo uma prevista.
    """
    n_pred, n_gt = len(pred_scores), len(gt_boxes)
    tp, matched = np.zeros(n_pred, dtype=bool), np.full(n_gt, -1)
    if n_pred == 0 or n_gt == 0:
        return tp, matched
    ious = iou_matrix(pred_boxes, gt_boxes)
    for p in np.argsort(-np.asarray(pred_scores), kind="stable"):
        free = np.flatnonzero(matched < 0)
        if len(free) == 0:
            break
        g = free[np.argmax(ious[p, free])]
        if ious[p, g] >= iou:
            tp[p], matched[g] = True, p
    return tp, matched


def _top(pred: dict, k: int = MAX_DETECTIONS) -> dict:
    """As k caixas previstas de maior confiança de uma foto."""
    order = np.argsort(-np.asarray(pred["scores"]), kind="stable")[:k]
    return {key: np.asarray(pred[key])[order] for key in ("boxes", "scores", "labels")}


def iou_inter(a: np.ndarray, b: np.ndarray) -> float:
    """Área da interseção de duas caixas."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(w, 0.0) * max(h, 0.0)


@dataclass(frozen=True)
class ClassMatches:
    """Previstas e anotadas de uma classe em todas as fotos, já casadas."""
    scores: np.ndarray      # confiança de cada prevista
    tp: np.ndarray          # acerto de cada prevista
    pred_photo: np.ndarray  # índice da foto de cada prevista
    gt_photo: np.ndarray    # índice da foto de cada anotada
    coverage: np.ndarray    # fração de cada anotada casada dentro da prevista (nan sem casamento)


def match_class(photos, preds: list[dict], label: int) -> ClassMatches:
    """Casa uma classe (rótulo 1 a 4) em todas as fotos; `preds` na ordem de `photos`."""
    scores, tp, pred_photo, gt_photo, coverage = [], [], [], [], []
    for i, (photo, pred) in enumerate(zip(photos, preds)):
        pred = _top(pred)
        keep = pred["labels"] == label
        boxes, conf = pred["boxes"][keep], pred["scores"][keep]
        gt = np.array([b for b, lab in zip(photo.boxes, photo.labels) if lab == label],
                      dtype=float).reshape(-1, 4)
        hit, matched = match_photo(boxes, conf, gt)
        cov = np.full(len(gt), np.nan)
        for g, p in enumerate(matched):
            if p >= 0:
                area = (gt[g, 2] - gt[g, 0]) * (gt[g, 3] - gt[g, 1])
                cov[g] = iou_inter(gt[g], boxes[p]) / area
        scores.append(conf)
        tp.append(hit)
        pred_photo.append(np.full(len(conf), i))
        gt_photo.append(np.full(len(gt), i))
        coverage.append(cov)
    return ClassMatches(_cat(scores, float), _cat(tp, bool), _cat(pred_photo, int),
                        _cat(gt_photo, int), _cat(coverage, float))


def _cat(parts: list[np.ndarray], dtype) -> np.ndarray:
    return np.concatenate(parts).astype(dtype) if parts else np.zeros(0, dtype)


def weighted_ap(m: ClassMatches, photo_weights: np.ndarray) -> float:
    """AP com cada caixa pesando o peso da sua foto, em 101 pontos de revocação.

    Precisão interpolada: a maior precisão com revocação >= r. Sem anotadas, nan.
    """
    w = np.asarray(photo_weights, dtype=float)
    total = w[m.gt_photo].sum()
    if total == 0:
        return float("nan")
    if len(m.scores) == 0:
        return 0.0  # há anotadas, mas nenhuma prevista
    order = np.argsort(-m.scores, kind="stable")
    pw = w[m.pred_photo[order]]
    tp_w = np.cumsum(pw * m.tp[order])
    all_w = np.cumsum(pw)
    recall = tp_w / total
    precision = np.divide(tp_w, all_w, out=np.zeros_like(tp_w), where=all_w > 0)
    # Envelope: precisão máxima à direita de cada ponto
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    idx = np.searchsorted(recall, RECALL_POINTS, side="left")
    values = np.where(idx < len(envelope), envelope[np.minimum(idx, len(envelope) - 1)], 0.0)
    return float(values.mean())


def selection_score(per_class_ap: dict[str, float]) -> float:
    """Métrica de escolha (D2 a): média das AP das três classes usadas pelo leitor."""
    return float(np.mean([per_class_ap[c] for c in SELECTION_CLASSES]))


def ap_table(photos, preds: list[dict]) -> tuple[dict[str, float], dict[str, ClassMatches]]:
    """AP ponderada de cada classe e os casamentos, que o bootstrap reaproveita."""
    from meter_reader.detector import DETECTOR_CLASSES
    weights = np.array([p.weight for p in photos])
    matches = {c: match_class(photos, preds, i + 1) for i, c in enumerate(DETECTOR_CLASSES)}
    return {c: weighted_ap(m, weights) for c, m in matches.items()}, matches


def coverage_summary(matches: dict[str, ClassMatches]) -> pd.DataFrame:
    """Por classe: anotadas casadas, p5 e mediana da cobertura do campo (protocolo, 1.6)."""
    rows = []
    for cls, m in matches.items():
        cov = m.coverage[~np.isnan(m.coverage)]
        rows.append({"classe": cls, "casadas": len(cov), "anotadas": len(m.coverage),
                     "cobertura_p5": round(float(np.quantile(cov, 0.05)), 3) if len(cov) else np.nan,
                     "cobertura_p50": round(float(np.median(cov)), 3) if len(cov) else np.nan})
    return pd.DataFrame(rows).set_index("classe")


def bootstrap_selection_diff(photos, matches_a: dict, matches_b: dict, n_boot: int = N_BOOT,
                             seed: int = BOOT_SEED) -> np.ndarray:
    """Diferenças (A − B) da métrica de escolha em reamostragens pareadas por foto.

    Cada reamostragem multiplica o peso de cada foto pelo número de vezes em que
    ela foi sorteada; as previstas e anotadas seguem a sua foto.
    """
    weights = np.array([p.weight for p in photos])
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for r in range(n_boot):
        counts = np.bincount(rng.integers(0, len(photos), len(photos)), minlength=len(photos))
        w = weights * counts
        a = np.mean([weighted_ap(matches_a[c], w) for c in SELECTION_CLASSES])
        b = np.mean([weighted_ap(matches_b[c], w) for c in SELECTION_CLASSES])
        out[r] = a - b
    return out


# ---------------------------------------------------------------------
# Segunda porta (protocolo, 1.7; D3)
# ---------------------------------------------------------------------

def photo_score(pred: dict) -> float:
    """Escore da foto: a maior confiança entre as caixas detectadas, de qualquer classe."""
    scores = np.asarray(pred["scores"])
    return float(scores.max()) if len(scores) else 0.0


def gate_curve(scores: np.ndarray, has_field: np.ndarray, weights: np.ndarray,
               alpha: float = ALPHA) -> pd.DataFrame:
    """Uma linha por limiar t (rejeita-se com escore < t), do menor ao maior.

    Perda ponderada das fotos com campo, o seu limite superior de Kish e a
    cobertura ponderada das fotos sem campo. O primeiro limiar (0) não rejeita nada.
    """
    scores, has, w = (np.asarray(scores, float), np.asarray(has_field, bool),
                      np.asarray(weights, float))
    n_eff = kish_n(w[has])
    thresholds = np.unique(np.concatenate([[0.0], scores]))
    order = np.argsort(scores, kind="stable")
    s_sorted = scores[order]
    lost_w = np.cumsum(np.where(has[order], w[order], 0.0))
    caught_w = np.cumsum(np.where(~has[order], w[order], 0.0))
    k = np.searchsorted(s_sorted, thresholds, side="left")  # fotos com escore < t
    lost = np.where(k > 0, lost_w[np.maximum(k - 1, 0)], 0.0)
    caught = np.where(k > 0, caught_w[np.maximum(k - 1, 0)], 0.0)
    loss = lost / w[has].sum()
    events = loss * n_eff
    upper = np.where(events >= n_eff, 1.0,
                     beta.ppf(1 - alpha, events + 1, np.maximum(n_eff - events, 1e-9)))
    no_field = w[~has].sum()
    coverage = caught / no_field if no_field > 0 else np.full(len(thresholds), np.nan)
    return pd.DataFrame({"limiar": thresholds, "perda": loss, "perda_lim_sup": upper,
                         "cobertura": coverage})


def choose_gate_threshold(curve: pd.DataFrame, max_loss: float = GATE_MAX_LOSS) -> float | None:
    """Maior limiar que cumpre a meta (o de maior cobertura, D3 c β); None se nenhum cumpre."""
    ok = curve[curve["perda_lim_sup"] <= max_loss]
    return None if ok.empty else float(ok["limiar"].max())

def main() -> None:
    # Importações que leem os dados só aqui: os testes usam as funções acima sem dados reais
    from meter_reader.config import get_work_dir
    from training.box_labels import read_labels
    from training.box_report import SAMPLE_NAME, SAMPLE_SHA256
    from training.build_manifest import file_sha256

    work_dir = get_work_dir()
    if file_sha256(work_dir / SAMPLE_NAME) != SAMPLE_SHA256:
        raise SystemExit("A amostra difere da gravada na T5.2.")
    sample = pd.read_csv(work_dir / SAMPLE_NAME, float_precision="round_trip")
    _, boxes = read_labels(work_dir, sample=set(sample["pseudonimo"]))
    with_field = photos_with_field(boxes)

    # Só o gabarito (caixas e pesos); nenhuma predição é lida (D6)
    rows = []
    for subset, part in sample.groupby("conjunto"):
        has = part["pseudonimo"].isin(with_field).to_numpy()
        w = part["peso"].to_numpy(dtype=float)
        n_eff = kish_n(w[has])
        rows.append({"conjunto": subset, "fotos": len(part), "com_campo": int(has.sum()),
                     "sem_campo": int((~has).sum()), "n_efetivo_com_campo": round(n_eff, 1),
                     "n_efetivo_sem_campo": round(kish_n(w[~has]), 1),
                     "orcamento_de_perdas": loss_budget(n_eff)})
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).set_index("conjunto"))


if __name__ == "__main__":
    main()
