"""Cabeças da função e de duas leituras do leitor (protocolo do leitor, seção 1.8).

Função (digitais com alvo de função): acerto exato da leitura livre; matriz de
confusão entre 03, 003, 103, 24, 55, vazio e outros; gatilho da F-b (acerto exato
abaixo de 95%, ou mais de 2% dos 103 lidos como 03 ou 003).
Duas leituras (digitais com marca): AP e revocação no limiar de 0,5 (logit 0), por
lote. Predições fora da dobra da configuração final, 3 sementes somadas. Só números.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from meter_reader.reader_text import FUNCTION_ALPHABET, greedy_decode

CONFIG = "L2-12+auto1"
CODES = ("03", "003", "103", "24", "55")
MIN_FUNCTION_ACCURACY = 0.95
MAX_103_AS_03 = 0.02


def function_category(code: str) -> str:
    """Categoria da matriz de confusão da seção 1.8."""
    if code == "":
        return "vazio"
    return code if code in CODES else "outro"


def fb_trigger(labels: list[str], reads: list[str]) -> dict:
    """Gatilho da F-b: acerto exato abaixo de 95%, ou mais de 2% dos 103 lidos como
    03 ou 003."""
    labels, reads = np.array(labels, dtype=object), np.array(reads, dtype=object)
    accuracy = float((labels == reads).mean())
    is_103 = labels == "103"
    as_03 = float(np.isin(reads[is_103], ["03", "003"]).mean()) if is_103.any() else 0.0
    return {"acerto": round(accuracy, 4), "itens": int(len(labels)),
            "103_como_03": round(as_03, 4), "itens_103": int(is_103.sum()),
            "dispara": accuracy < MIN_FUNCTION_ACCURACY or as_03 > MAX_103_AS_03}


def main() -> None:
    from meter_reader.config import get_work_dir
    from training.reader_dataset import load_items
    from training.reader_train import META_NAME, PREDICTIONS_NAME, RUNS_DIR

    config = sys.argv[1] if len(sys.argv) > 1 else CONFIG
    work_dir = get_work_dir()
    items, _ = load_items()
    index = items.set_index(items["pseudonimo"] + "|" + items["classe"] + "|"
                            + items["recorte"])
    func_rows, two_rows = [], []
    for meta_path in sorted((work_dir / RUNS_DIR / config).glob(f"*/{META_NAME}")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        pred = np.load(meta_path.parent / PREDICTIONS_NAME)
        rows = index.loc[pred["chave_texto"]]
        keep = rows["usa_funcao"].to_numpy(bool)
        lp = pred["funcao_0"].astype(np.float32)[keep]
        for label, row_lp in zip(rows.loc[keep, "funcao"], lp):
            func_rows.append({"lote": meta["lote_avaliado"], "semente": meta["semente"],
                              "rotulo": label,
                              "lido": greedy_decode(row_lp.argmax(-1).tolist(),
                                                    FUNCTION_ALPHABET)})
        rows_all = index.loc[pred["chave"]]
        keep = rows_all["usa_duas"].to_numpy(bool)
        for label, logit in zip(rows_all.loc[keep, "duas_leituras"], pred["duas"][keep]):
            two_rows.append({"lote": meta["lote_avaliado"], "semente": meta["semente"],
                             "marca": bool(label), "logit": float(logit)})
    func, two = pd.DataFrame(func_rows), pd.DataFrame(two_rows)

    print("Função, gatilho da F-b (3 sementes somadas):")
    print(fb_trigger(func["rotulo"].tolist(), func["lido"].tolist()))
    print("\nFunção, acerto por lote e semente:")
    print((func["rotulo"] == func["lido"]).groupby([func["lote"], func["semente"]]).mean()
          .round(3).unstack().to_string())
    print("\nFunção, matriz de confusão (linhas: rótulo; colunas: lido):")
    print(pd.crosstab(func["rotulo"].map(function_category), func["lido"].map(function_category),
                      margins=True, margins_name="total").to_string())

    print("\nDuas leituras, por lote (3 sementes somadas):")
    out = []
    for lot, part in two.groupby("lote"):
        y, s = part["marca"].to_numpy(), part["logit"].to_numpy()
        fired = s > 0
        out.append({"lote": lot, "itens": len(part), "positivos": int(y.sum()),
                    "AP": round(float(average_precision_score(y, s)), 3) if y.any() else None,
                    "revocacao_0.5": round(float(fired[y].mean()), 3) if y.any() else None,
                    "disparos": int(fired.sum()),
                    "logit_p50_positivos": round(float(np.median(s[y])), 2) if y.any() else None,
                    "logit_p50_negativos": round(float(np.median(s[~y])), 2)})
    print(pd.DataFrame(out).to_string(index=False))


if __name__ == "__main__":
    main()
