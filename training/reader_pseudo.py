"""Pseudo-rótulos do autotreinamento do leitor (protocolo do leitor, seção 1.10).

Na dobra k (lote k avaliado): o modelo da configuração adotada, treinado sem o lote
k (semente 0), lê os recortes dos lotes de treino que não têm alvo de texto. O
pseudo-rótulo é aceito quando a leitura livre é igual ao CSV (número inteiro na
leitura; bloco mais à esquerda no número de série) e a probabilidade é de pelo menos
o limiar alto da classe, calculado (seção 1.7) só com as predições fora da dobra dos
dois lotes de treino. O alvo é a leitura livre, com os zeros que o display mostra
(Z-a). Ficam de fora: a função (sem pseudo-rótulo), os digitais marcados com duas
leituras pelo professor e o lote avaliado. Só contagens são exibidas.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from meter_reader.reader_text import TEXT_ALPHABET, greedy_decode
from training.reader_dataset import ITEM_COLUMNS
from training.reader_eval import same_value, value_logprob
from training.reader_metrics import CLASSES, high_threshold

PSEUDO_DIR = Path("experimentos") / "leitor" / "pseudo"
SOURCE_SEED = 0
BATCH = 128


def fold_thresholds(scores: pd.DataFrame, held_out_lot: str) -> dict[str, float]:
    """Limiar alto (log) por classe, só com as predições fora da dobra dos lotes de
    treino da dobra (semente de origem)."""
    part = scores[(scores["semente"] == SOURCE_SEED) & (scores["lote"] != held_out_lot)]
    return {c: high_threshold(part[part["classe"] == c])["log_limiar"] for c in CLASSES}


def candidates(crops: pd.DataFrame, lots: dict[str, str], held_out_lot: str,
               targets: set[str], two_lines: set[str]) -> pd.DataFrame:
    """Recortes dos lotes de treino sem alvo de texto e sem marca de duas leituras."""
    df = crops[["pseudonimo", "classe"]].copy()
    df["lote"] = df["pseudonimo"].map(lots)
    key = df["pseudonimo"] + "|" + df["classe"]
    keep = (df["classe"].isin(CLASSES) & df["lote"].notna() & (df["lote"] != held_out_lot)
            & ~key.isin(targets) & ~key.isin(two_lines))
    return df[keep].reset_index(drop=True)


def decide(log_probs: np.ndarray, expected: str, field_class: str,
           log_t: float) -> str | None:
    """Pseudo-rótulo (o texto lido) ou None, pela regra da seção 1.10."""
    text = greedy_decode(np.asarray(log_probs).argmax(-1).tolist(), TEXT_ALPHABET)
    if not text or not same_value(text, expected, field_class):
        return None
    return text if value_logprob(log_probs, expected, field_class) >= log_t else None


def label_fold(model, cands: pd.DataFrame, expected: dict[tuple, str], cache,
               thresholds: dict[str, float], device: str) -> pd.DataFrame:
    """Lê os candidatos com o modelo da dobra e devolve os pseudo-rótulos aceitos."""
    from meter_reader.reader_input import reader_tensor
    from training.reader_cache import crop_key

    rows = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(cands), BATCH):
            part = cands.iloc[start:start + BATCH]
            x = torch.stack([reader_tensor(Image.fromarray(cache[crop_key(p, c)]))
                             for p, c in zip(part["pseudonimo"], part["classe"])])
            lp = model(x.to(device))["texto"].float().log_softmax(-1).cpu().numpy()
            for k, r in enumerate(part.itertuples(index=False)):
                exp = expected.get((r.pseudonimo, r.classe))
                if exp is None:
                    continue
                text = decide(lp[k], exp, r.classe, thresholds[r.classe])
                if text is not None:
                    rows.append({"pseudonimo": r.pseudonimo, "classe": r.classe,
                                 "lote": r.lote, "texto": text})
    return pd.DataFrame(rows, columns=["pseudonimo", "classe", "lote", "texto"])


def as_items(pseudo: pd.DataFrame) -> pd.DataFrame:
    """Pseudo-rótulos no formato dos itens de treino: só a perda do texto."""
    out = pseudo.assign(recorte="inteiro", funcao="", duas_leituras=False, usa_texto=True,
                        usa_funcao=False, usa_duas=False, origem="pseudo")
    return out[list(ITEM_COLUMNS)]


def build(source_config: str, items: pd.DataFrame, device: str = "cuda") -> dict:
    """Pseudo-rótulos de cada dobra, gravados em `pseudo/<configuração>_<lote>.csv`
    (lidos se já existirem). Devolve lote avaliado -> pseudo-rótulos."""
    from meter_reader.config import get_work_dir
    from meter_reader.reader import Reader
    from training.group_dataset import LABEL_VERSIONS, load_group_photos
    from training.reader_cache import CACHE_DIR, CACHE_NAME, load_cache
    from training.reader_eval import score_config
    from training.reader_labels import CROPS_NAME, CROPS_SHA256, labels_name
    from training.reader_pilot import csv_rows
    from training.reader_train import RUNS_DIR, WEIGHTS_NAME
    from training.teacher import MODEL

    work_dir = get_work_dir()
    out, missing = {}, []
    for lot in sorted(items["lote"].unique()):
        path = work_dir / PSEUDO_DIR / f"{source_config}_{lot}.csv"
        if path.exists():
            out[lot] = pd.read_csv(path, dtype={"texto": str}, keep_default_na=False)
        else:
            missing.append((lot, path))
    if not missing:
        return out

    photos = load_group_photos(LABEL_VERSIONS["r4"])
    lots = {p.pseudonym: p.lot for p in photos}
    rows = csv_rows(photos)
    crops = pd.read_csv(work_dir / CROPS_NAME, float_precision="round_trip")
    expected = {}
    for r in crops[["pseudonimo", "classe"]].itertuples(index=False):
        row = rows.get(r.pseudonimo)
        if row is not None:
            expected[(r.pseudonimo, r.classe)] = (row.meter_numbers[0] if r.classe == CLASSES[2]
                                                  else str(row.reading))
    raw = pd.read_csv(work_dir / labels_name(MODEL), keep_default_na=False, dtype=str)
    two_lines = set((raw.loc[raw["duas_leituras"] == "True", "pseudonimo"] + "|"
                     + raw.loc[raw["duas_leituras"] == "True", "classe"]))
    text_items = items[items["usa_texto"]]
    targets = set(text_items["pseudonimo"] + "|" + text_items["classe"])
    scores = score_config(work_dir, source_config, items)
    cache = load_cache(work_dir / CACHE_DIR / CACHE_NAME, CROPS_SHA256)
    candidate = source_config.split("+")[0]
    for lot, path in missing:
        model = Reader(candidate, pretrained=False).to(device)
        weights = work_dir / RUNS_DIR / source_config / f"{lot}_s{SOURCE_SEED}" / WEIGHTS_NAME
        model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
        cands = candidates(crops, lots, lot, targets, two_lines)
        pseudo = label_fold(model, cands, expected, cache,
                            fold_thresholds(scores, lot), device)
        path.parent.mkdir(parents=True, exist_ok=True)
        pseudo.to_csv(path, index=False)
        out[lot] = pseudo
        print(f"Dobra {lot}: {len(pseudo)} pseudo-rótulos de {len(cands)} candidatos")
    return out


def main() -> None:
    from training.reader_dataset import load_items

    items, _ = load_items()
    pseudo = build("L2-12", items)
    for lot, df in pseudo.items():
        counts = df["classe"].value_counts().reindex(list(CLASSES), fill_value=0)
        print(f"Dobra {lot}: " + ", ".join(f"{c} {n}" for c, n in counts.items()))


if __name__ == "__main__":
    main()
