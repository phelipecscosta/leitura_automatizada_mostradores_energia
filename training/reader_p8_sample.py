"""Amostras da análise de erros (protocolo do leitor, seção 1.13, P8) e dois grupos
informativos, que não entram no gatilho:

- p8: 50 erros de transcrição fora da dobra da configuração final (semente 0),
  sorteados com semente em texto, proporcionais aos erros de cada classe;
- funcao_103: 10 displays com função 103 lidos como consumo pela cabeça da R12
  (semente 0), para ver se o "1" está visível (hipótese da resolução, F-x4);
- confirmacao_falsa: os casos reais em que o leitor confirmaria o CSV com o valor
  verdadeiro diferente (T6b.8d), para ver quem está certo.

Grava a lista na pasta de trabalho, para a ferramenta da `local/` da pesquisa. Na
tela, só contagens.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from training.reader_heads_train import CONFIG, FUNCTION_CLASSES, HEADS_DIR, function_class

SAMPLE_NAME = Path("experimentos") / "leitor" / "p8_amostra.csv"
P8_SIZE, P8_SEED = 50, "p8-erros"
F103_SIZE, F103_SEED = 10, "p8-funcao-103"


def _order(keys: pd.Series, seed: str) -> pd.Series:
    return keys.map(lambda k: hashlib.sha256(f"{seed}|{k}".encode()).hexdigest())


def proportional(counts: dict[str, int], total: int) -> dict[str, int]:
    """Divisão de `total` proporcional a `counts`, pelos maiores restos."""
    n = sum(counts.values())
    raw = {k: total * v / n for k, v in counts.items()}
    out = {k: int(np.floor(v)) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - out[k], reverse=True)[:total - sum(out.values())]:
        out[k] += 1
    return out


def main() -> None:
    from meter_reader.config import get_work_dir
    from meter_reader.reader_text import TEXT_ALPHABET, greedy_decode
    from training.group_dataset import LABEL_VERSIONS, load_group_photos
    from training.reader_cache import CACHE_DIR, CACHE_NAME
    from training.reader_dataset import fold_masks, load_items
    from training.reader_eval import score_config, value_logprob
    from training.reader_function_r12 import R12_HEADS, extended_items, predict_function
    from training.reader_labels import CROPS_SHA256
    from training.reader_metrics import config_table
    from training.reader_pilot import csv_rows
    from training.reader_train import META_NAME, PREDICTIONS_NAME, RUNS_DIR, WEIGHTS_NAME

    work_dir = get_work_dir()
    items, _ = load_items()
    index = items.set_index(items["pseudonimo"] + "|" + items["classe"] + "|" + items["recorte"])
    rows = csv_rows(load_group_photos(LABEL_VERSIONS["r4"]))
    out = []

    # P8: erros de transcrição da semente 0
    scores = score_config(work_dir, CONFIG, items)
    errors = scores[(scores["semente"] == 0) & ~scores["acerto"].astype(bool)].copy()
    errors["ordem"] = _order(errors["chave"], P8_SEED)
    quota = proportional(errors["classe"].value_counts().to_dict(), P8_SIZE)
    for cls, n in quota.items():
        for r in errors[errors["classe"] == cls].sort_values("ordem").head(n).to_dict("records"):
            it = index.loc[r["chave"]]
            out.append({"grupo": "p8", "chave": r["chave"], "rotulo": it["texto"],
                        "lido": "" if pd.isna(r["livre"]) else str(r["livre"])})

    # Função 103 lida como consumo (cabeças da R12, semente 0)
    ext = extended_items()
    cache = work_dir / CACHE_DIR / CACHE_NAME
    cons = [FUNCTION_CLASSES.index("03"), FUNCTION_CLASSES.index("003")]
    cands = []
    for meta_path in sorted((work_dir / RUNS_DIR / CONFIG).glob(f"*_s0/{META_NAME}")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        part = ext[fold_masks(ext, meta["lote_avaliado"])["avaliacao"]
                   & ext["usa_funcao"]].reset_index(drop=True)
        probs = predict_function(meta_path.parent / WEIGHTS_NAME,
                                 work_dir / HEADS_DIR / R12_HEADS / meta_path.parent.name
                                 / "pesos_cabecas.pt", meta["candidato"], meta["cinza"], part,
                                 cache, CROPS_SHA256)
        hit = (part["funcao"].map(function_class) == FUNCTION_CLASSES.index("103")).to_numpy() \
            & np.isin(probs.argmax(1), cons)
        for r, p in zip(part[hit].to_dict("records"), probs[hit]):
            cands.append({"grupo": "funcao_103",
                          "chave": f"{r['pseudonimo']}|{r['classe']}|{r['recorte']}",
                          "rotulo": r["funcao"], "lido": FUNCTION_CLASSES[int(p.argmax())]})
    cands = pd.DataFrame(cands)
    cands["ordem"] = _order(cands["chave"], F103_SEED)
    out += cands.sort_values("ordem").head(F103_SIZE).drop(columns="ordem").to_dict("records")

    # Confirmações falsas reais (todas as sementes, recortes únicos)
    highs = config_table(scores).set_index(["semente", "classe"])["log_limiar"]
    seen = set()
    for meta_path in sorted((work_dir / RUNS_DIR / CONFIG).glob(f"*/{META_NAME}")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        pred = np.load(meta_path.parent / PREDICTIONS_NAME)
        for key, lp in zip(pred["chave_texto"], pred["texto_0"]):
            it = index.loc[key]
            if it["classe"] == "numero_serie" or it["pseudonimo"] not in rows or key in seen:
                continue
            expected = int(rows[it["pseudonimo"]].reading)
            if int(it["texto"]) == expected:
                continue
            lp = lp.astype(np.float32)
            free = greedy_decode(lp.argmax(-1).tolist(), TEXT_ALPHABET)
            if (free.isdigit() and int(free) == expected and value_logprob(
                    lp, str(expected), it["classe"]) >= highs.loc[(meta["semente"], it["classe"])]):
                seen.add(key)
                out.append({"grupo": "confirmacao_falsa", "chave": key, "rotulo": it["texto"],
                            "lido": free})

    df = pd.DataFrame(out)
    df[["pseudonimo", "classe", "recorte"]] = df["chave"].str.split("|", expand=True)
    df["csv"] = df["pseudonimo"].map(lambda p: str(rows[p].reading) if p in rows else "")
    path = work_dir / SAMPLE_NAME
    df.drop(columns="chave").to_csv(path, index=False)
    print("Cotas da P8 por classe:", quota)
    print("Amostra gravada, por grupo:", df["grupo"].value_counts().to_dict())


if __name__ == "__main__":
    main()
