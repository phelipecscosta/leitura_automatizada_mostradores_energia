"""Comparação das cabeças novas (F-b1 e D1) com as antigas, fora da dobra (protocolo
do leitor, revisão R11). Só números.

- Função: revocação média por classe (classes pelo prefixo), média das sementes, com
  bootstrap pareado por recorte, estratificado pela classe verdadeira. Adotada se a
  diferença tiver o IC acima de zero.
- Duas leituras: AP, média das sementes, com bootstrap pareado por recorte,
  estratificado pela marca; limiar pelo F1, com validação aninhada por lote.
  Adotada se o AP subir com o IC acima de zero e a revocação no limiar for de pelo
  menos 0,5.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from meter_reader.reader_text import FUNCTION_ALPHABET, greedy_decode
from training.reader_heads_train import (CONFIG, FUNCTION_CLASSES, HEADS_DIR, function_class)
from training.reader_rule import BOOT_SEED, N_BOOT

MIN_RECALL_TWO = 0.5


def stratified_weights(groups: np.ndarray, n_boot: int, rng) -> np.ndarray:
    """Pesos de reamostragem (n_boot x itens), com o tamanho de cada grupo mantido."""
    w = np.zeros((n_boot, len(groups)))
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        w[:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1 / len(idx)), size=n_boot)
    return w


def boot_macro_recall(y: np.ndarray, hits: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Revocação média por classe em cada reamostragem. `hits`: (sementes, itens)."""
    out = np.zeros((weights.shape[0], hits.shape[0]))
    classes = np.unique(y)
    for c in classes:
        sel = y == c
        out += (weights[:, sel] @ hits[:, sel].T) / weights[:, sel].sum(1, keepdims=True)
    return (out / len(classes)).mean(axis=1)


def f1_threshold(y: np.ndarray, s: np.ndarray) -> float:
    """Limiar que maximiza o F1."""
    precision, recall, thresholds = precision_recall_curve(y, s)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[int(np.argmax(f1))])


def load(items: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Uma linha por recorte e semente: rótulos e saídas antigas e novas."""
    from meter_reader.config import get_work_dir
    from training.reader_train import META_NAME, PREDICTIONS_NAME, RUNS_DIR

    work_dir = get_work_dir()
    index = items.set_index(items["pseudonimo"] + "|" + items["classe"] + "|" + items["recorte"])
    func, two = [], []
    for meta_path in sorted((work_dir / RUNS_DIR / CONFIG).glob(f"*/{META_NAME}")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        old = np.load(meta_path.parent / PREDICTIONS_NAME)
        new = np.load(work_dir / HEADS_DIR / CONFIG / meta_path.parent.name / "predicoes_cabecas.npz")
        new_f = dict(zip(new["chave"], new["funcao"]))
        new_t = dict(zip(new["chave"], new["duas"]))
        base = {"lote": meta["lote_avaliado"], "semente": meta["semente"]}
        for key, lp in zip(old["chave_texto"], old["funcao_0"]):
            r = index.loc[key]
            if r["usa_funcao"]:
                old_read = greedy_decode(lp.astype(np.float32).argmax(-1).tolist(),
                                         FUNCTION_ALPHABET)
                func.append({**base, "chave": key, "y": function_class(r["funcao"]),
                             "antiga": function_class(old_read),
                             "nova": int(np.argmax(new_f[key]))})
        for key, logit in zip(old["chave"], old["duas"]):
            r = index.loc[key]
            if r["usa_duas"]:
                two.append({**base, "chave": key, "y": bool(r["duas_leituras"]),
                            "antiga": float(logit), "nova": float(new_t[key])})
    return pd.DataFrame(func), pd.DataFrame(two)


def main() -> None:
    from training.reader_dataset import load_items

    items, _ = load_items()
    func, two = load(items)
    rng = np.random.default_rng(BOOT_SEED)
    names = list(FUNCTION_CLASSES)

    # Função
    wide = func.pivot(index="chave", columns="semente", values=["y", "antiga", "nova"])
    y = wide["y"].iloc[:, 0].to_numpy()
    hits = {m: (wide[m].to_numpy().T == y) for m in ("antiga", "nova")}
    w = stratified_weights(y, N_BOOT, rng)
    boot = {m: boot_macro_recall(y, hits[m], w) for m in hits}
    point = {m: boot_macro_recall(y, hits[m], np.ones((1, len(y))))[0] for m in hits}
    low, high = np.percentile(boot["nova"] - boot["antiga"], [2.5, 97.5])
    print("Função, revocação média por classe (média das sementes):")
    print({m: round(float(v), 4) for m, v in point.items()},
          f"dif {point['nova'] - point['antiga']:.4f} IC [{low:.4f}; {high:.4f}]",
          "adotada:", bool(low > 0))
    for m in ("antiga", "nova"):
        is_103 = func["y"] == names.index("103")
        as_03 = func.loc[is_103, m].isin([names.index("03"), names.index("003")]).mean()
        print(f"  {m}: acerto por classe {float((func[m] == func['y']).mean()):.4f}; "
              f"103 como 03/003 {float(as_03):.4f}")
    print("\nFunção nova, matriz de confusão (3 sementes; linhas: rótulo):")
    print(pd.crosstab(func["y"].map(dict(enumerate(names))),
                      func["nova"].map(dict(enumerate(names)))).to_string())

    # Duas leituras
    wide = two.pivot(index="chave", columns="semente", values=["y", "antiga", "nova"])
    lots = two.drop_duplicates("chave").set_index("chave").loc[wide.index, "lote"].to_numpy()
    yt = wide["y"].iloc[:, 0].to_numpy(bool)
    w = stratified_weights(yt.astype(int), N_BOOT, rng)
    ap = {m: np.zeros(N_BOOT) for m in ("antiga", "nova")}
    ap_point = {}
    for m in ap:
        scores = wide[m].to_numpy().T                               # (sementes, recortes)
        ap_point[m] = float(np.mean([average_precision_score(yt, s) for s in scores]))
        for b in range(N_BOOT):
            ap[m][b] = np.mean([average_precision_score(yt, s, sample_weight=w[b])
                                for s in scores])
    low, high = np.percentile(ap["nova"] - ap["antiga"], [2.5, 97.5])
    rec, prec = [], []
    for lot in np.unique(lots):
        fit, test = lots != lot, lots == lot
        scores = wide["nova"].to_numpy().T
        t = f1_threshold(np.tile(yt[fit], len(scores)), scores[:, fit].ravel())
        fired = scores[:, test] >= t
        rec.append(fired[:, yt[test]].sum() / max(1, yt[test].sum() * len(scores)))
        prec.append(fired[:, yt[test]].sum() / max(1, fired.sum()))
        print(f"Duas leituras, lote {lot}: limiar (logit) {t:.2f}; revocação {rec[-1]:.3f}; "
              f"precisão {prec[-1]:.3f}")
    recall = float(np.mean(rec))
    print(f"\nDuas leituras, AP (média das sementes): "
          f"{ {m: round(v, 4) for m, v in ap_point.items()} } "
          f"dif {ap_point['nova'] - ap_point['antiga']:.4f} IC [{low:.4f}; {high:.4f}]; "
          f"revocação no limiar {recall:.3f}; adotada: {bool(low > 0 and recall >= MIN_RECALL_TWO)}")


if __name__ == "__main__":
    main()
