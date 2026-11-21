"""Diagnóstico dos ciclométricos antes da revisão R8 (protocolo do leitor).

1. Onde está o erro: acerto exato por classe nos itens de treino (já vistos), na
   divisão interna (fotos não vistas, dos mesmos lotes) e no lote fora da dobra, com
   os pesos da configuração dada. Treino alto e o resto baixo indica variância
   (sobreajuste); treino baixo indica viés (o modelo erra até o que já viu); interna
   alta e fora da dobra baixa indica mudança entre lotes.
2. Tipo do erro nos ciclométricos fora da dobra, comparando os números sem os zeros
   à esquerda (Z-a).
3. Reserva: recortes ciclométricos dos lotes de desenvolvimento sem alvo de texto,
   por status do professor e lote.

Só contagens e proporções; nenhum texto, valor ou pseudônimo é exibido.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from meter_reader.reader_text import TEXT_ALPHABET, greedy_decode
from training.reader_dataset import CYCLOMETRIC
from training.reader_metrics import CLASSES

CONFIG = "L2-12"
PARTS = {"treino": "treino (vistos)", "interno": "divisão interna",
         "avaliacao": "fora da dobra"}


def error_kind(read: str, label: str) -> tuple[str, int | None, str | None]:
    """Tipo do erro de uma leitura de display: (tipo, posição do dígito trocado contada
    da direita, par "rótulo>lido"). Os números são comparados sem os zeros à esquerda."""
    if not read:
        return "vazio", None, None
    if not read.isdigit():
        return "com_letra", None, None
    a, b = read.lstrip("0") or "0", label.lstrip("0") or "0"
    if a == b:
        return "certo", None, None
    if len(a) == len(b) - 1 and any(b[k] == b[k + 1] and b[:k] + b[k + 1:] == a
                                    for k in range(len(b) - 1)):
        return "falta_repetido", None, None   # dois dígitos iguais lidos como um (CTC)
    if len(a) != len(b):
        return ("falta_digito" if len(a) < len(b) else "sobra_digito"), None, None
    diff = [k for k in range(len(a)) if a[k] != b[k]]
    if len(diff) > 1:
        return "varios_digitos", None, None
    k = diff[0]
    return "um_digito", len(a) - k, f"{b[k]}>{a[k]}"


def _hits(model, part: pd.DataFrame, cache, list_sha: str, gray: bool,
          device: str) -> np.ndarray:
    """Acerto exato (M-a) de cada item, na ordem de `part`."""
    from training.reader_torch import ReaderDataset, collate
    from training.reader_train import EVAL_BATCH, text_matches

    loader = DataLoader(ReaderDataset(part, cache, list_sha, train=False, gray=gray),
                        batch_size=EVAL_BATCH, collate_fn=collate, num_workers=2)
    hits = np.zeros(len(part), dtype=bool)
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["imagem"].to(device))["texto"].float().cpu()
            for row, idx in enumerate(batch["indice"].tolist()):
                r = part.iloc[idx]
                text = greedy_decode(logits[row].argmax(-1).tolist(), TEXT_ALPHABET)
                hits[idx] = text_matches(text, r["texto"], r["classe"])
    return hits


def split_accuracy(config: str, items: pd.DataFrame, device: str = "cuda") -> pd.DataFrame:
    """Acerto por treino (lote avaliado x semente), parte e classe."""
    from meter_reader.config import get_work_dir
    from meter_reader.reader import Reader
    from training.reader_cache import CACHE_DIR, CACHE_NAME
    from training.reader_dataset import fold_masks
    from training.reader_labels import CROPS_SHA256
    from training.reader_train import META_NAME, RUNS_DIR, WEIGHTS_NAME

    work_dir = get_work_dir()
    cache = work_dir / CACHE_DIR / CACHE_NAME
    text = items["usa_texto"].to_numpy(bool)
    rows = []
    for meta_path in sorted((work_dir / RUNS_DIR / config).glob(f"*/{META_NAME}")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        model = Reader(meta["candidato"], pretrained=False).to(device)
        model.load_state_dict(torch.load(meta_path.parent / WEIGHTS_NAME, map_location=device,
                                         weights_only=True))
        model.eval()
        masks = fold_masks(items, meta["lote_avaliado"])
        for part_name in PARTS:
            part = items[masks[part_name] & text].reset_index(drop=True)
            hits = _hits(model, part, cache, CROPS_SHA256, meta["cinza"], device)
            for cls in CLASSES:
                sel = (part["classe"] == cls).to_numpy()
                rows.append({"lote": meta["lote_avaliado"], "semente": meta["semente"],
                             "parte": part_name, "classe": cls, "n": int(sel.sum()),
                             "acerto": float(hits[sel].mean()) if sel.any() else np.nan})
        print(f"{meta['lote_avaliado']} s{meta['semente']}: concluído", flush=True)
    return pd.DataFrame(rows)


def error_profile(scores: pd.DataFrame, items: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Tipos de erro dos ciclométricos fora da dobra (todas as sementes)."""
    labels = items.set_index(items["pseudonimo"] + "|" + items["classe"] + "|"
                             + items["recorte"])["texto"]
    cyc = scores[scores["classe"] == CYCLOMETRIC].copy()
    kinds = [error_kind(str(read) if pd.notna(read) else "", labels[key])
             for read, key in zip(cyc["livre"], cyc["chave"])]
    cyc["tipo"], cyc["posicao"], cyc["par"] = zip(*kinds)
    wrong = cyc[cyc["tipo"] != "certo"]
    one = wrong[wrong["tipo"] == "um_digito"]
    return {
        "tipos": cyc["tipo"].value_counts().to_frame("leituras"),
        "acerto_por_origem": cyc.groupby("origem")["acerto"].agg(["size", "mean"]).round(3),
        "posicao_um_digito": one["posicao"].astype(int).value_counts().sort_index()
        .to_frame("leituras (1 = último tambor)"),
        "pares_um_digito": one["par"].value_counts().head(10).to_frame("leituras"),
    }


def reserve(items: pd.DataFrame) -> pd.DataFrame:
    """Recortes ciclométricos dos lotes de desenvolvimento, com e sem alvo de texto,
    por status do professor (Haiku) e lote."""
    from meter_reader.config import get_work_dir
    from training.group_dataset import LABEL_VERSIONS, load_group_photos
    from training.reader_labels import CROPS_NAME, labels_name
    from training.teacher import MODEL

    work_dir = get_work_dir()
    crops = pd.read_csv(work_dir / CROPS_NAME, float_precision="round_trip")
    raw = pd.read_csv(work_dir / labels_name(MODEL), keep_default_na=False, dtype=str)
    lots = {p.pseudonym: p.lot for p in load_group_photos(LABEL_VERSIONS["r4"])}
    cyc = crops.loc[crops["classe"] == CYCLOMETRIC, ["pseudonimo", "classe"]].copy()
    key = cyc["pseudonimo"] + "|" + cyc["classe"]
    status = dict(zip(raw["pseudonimo"] + "|" + raw["classe"], raw["status"]))
    text = items[items["usa_texto"]]
    targets = set(text["pseudonimo"] + "|" + text["classe"])
    cyc["lote"] = cyc["pseudonimo"].map(lots).fillna("sem_lote")
    cyc["situacao"] = np.where(key.isin(targets), "com_alvo", "sem_alvo")
    cyc["status_professor"] = key.map(status).fillna("sem_resposta")
    return pd.crosstab([cyc["situacao"], cyc["status_professor"]], cyc["lote"], margins=True,
                       margins_name="total")


def main() -> None:
    from meter_reader.config import get_work_dir
    from training.reader_dataset import load_items
    from training.reader_eval import score_config

    config = sys.argv[1] if len(sys.argv) > 1 else CONFIG
    items, _ = load_items()
    pd.set_option("display.width", 160)

    acc = split_accuracy(config, items)
    table = (acc.groupby(["classe", "parte"])["acerto"].agg(["mean", "min", "max"])
             .reindex(pd.MultiIndex.from_tuples([(c, p) for c in CLASSES for p in PARTS]))
             .rename(index=PARTS, level=1).round(3))
    print(f"\n1. Acerto exato de {config} (média, mínimo e máximo dos 9 treinos):")
    print(table.to_string())
    print("\nItens por parte (média por treino):")
    print(acc.groupby(["classe", "parte"])["n"].mean().round(0).unstack()
          .reindex(index=list(CLASSES), columns=list(PARTS)).rename(columns=PARTS).to_string())

    scores = score_config(get_work_dir(), config, items)
    print(f"\n2. Ciclométricos fora da dobra ({config}, 3 sementes):")
    for name, df in error_profile(scores, items).items():
        print(f"\n{name}:\n{df.to_string()}")

    print("\n3. Reserva de ciclométricos nos lotes de desenvolvimento:")
    print(reserve(items).to_string())


if __name__ == "__main__":
    main()
    