"""Temperatura e ECE do leitor (protocolo do leitor, seção 1.6).

Uma temperatura por cabeça de CTC (texto e função), sobre as predições fora da dobra
da configuração final. A temperatura T divide os logits; a leitura livre (gulosa) não
muda, só a sua confiança. T é escolhida para que a confiança da leitura livre seja a
probabilidade de ela estar certa: mínimo da entropia cruzada binária entre a
confiança e o acerto exato (as 3 sementes juntas).

Validação aninhada por lote (como na E31): para cada lote, T é ajustada nos outros
dois e o ECE é medido nele. A temperatura final usa os três lotes e é gravada na
pasta de trabalho. Os limiares não dependem dela. Só números.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.optimize import minimize_scalar

from meter_reader.reader_text import FUNCTION_ALPHABET, TEXT_ALPHABET, encode, greedy_decode

CONFIG = "L2-12+auto1"
BINS = 10
TEMPERATURE_NAME = Path("experimentos") / "leitor" / "temperaturas.json"
LOG_T_BOUNDS = (np.log(0.25), np.log(8.0))


def free_readings(log_probs: np.ndarray, alphabet: str) -> list[str]:
    """Leitura gulosa de cada item; não depende da temperatura."""
    return [greedy_decode(row.argmax(-1).tolist(), alphabet) for row in log_probs]


def free_confidence(log_probs: np.ndarray, texts: list[str], alphabet: str,
                    temperature: float = 1.0) -> np.ndarray:
    """Probabilidade de cada leitura livre pela CTC, com as saídas divididas por T.
    `log_probs`: (N, passos, classes). Dividir as log-probabilidades por T equivale a
    dividir os logits, porque a diferença é uma constante por passo."""
    lp = torch.from_numpy(np.asarray(log_probs, dtype=np.float32) / temperature)
    lp = lp.log_softmax(-1).transpose(0, 1)                      # (passos, N, classes)
    targets = [torch.tensor(encode(t, alphabet), dtype=torch.long) for t in texts]
    lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    flat = torch.cat(targets) if int(lengths.sum()) else torch.zeros(0, dtype=torch.long)
    loss = F.ctc_loss(lp, flat, torch.full((len(texts),), lp.shape[0], dtype=torch.long),
                      lengths, blank=0, reduction="none", zero_infinity=False)
    return np.exp(-loss.numpy())


def ece(confidence: np.ndarray, correct: np.ndarray, bins: int = BINS) -> float:
    """Erro de calibração esperado: média ponderada, por faixas iguais de confiança,
    da distância entre a confiança média e o acerto."""
    confidence, correct = np.asarray(confidence, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(confidence, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        sel = idx == b
        if sel.any():
            total += sel.mean() * abs(confidence[sel].mean() - correct[sel].mean())
    return float(total)


def fit_temperature(log_probs: np.ndarray, texts: list[str], correct: np.ndarray,
                    alphabet: str) -> float:
    """T que minimiza a entropia cruzada binária entre confiança e acerto."""
    correct = np.asarray(correct, float)

    def nll(log_t: float) -> float:
        p = np.clip(free_confidence(log_probs, texts, alphabet, float(np.exp(log_t))),
                    1e-6, 1 - 1e-6)
        return float(-np.mean(correct * np.log(p) + (1 - correct) * np.log(1 - p)))

    return float(np.exp(minimize_scalar(nll, bounds=LOG_T_BOUNDS, method="bounded").x))


def head_data(config: str, items: pd.DataFrame, head: str) -> dict:
    """Saídas fora da dobra de uma cabeça ("texto" ou "funcao"), das 3 sementes, com
    o lote, a leitura livre e o acerto de cada item."""
    from meter_reader.config import get_work_dir
    from training.reader_train import META_NAME, PREDICTIONS_NAME, RUNS_DIR, text_matches

    work_dir = get_work_dir()
    index = items.set_index(items["pseudonimo"] + "|" + items["classe"] + "|"
                            + items["recorte"])
    alphabet = TEXT_ALPHABET if head == "texto" else FUNCTION_ALPHABET
    lps, lots, texts, correct = [], [], [], []
    for meta_path in sorted((work_dir / RUNS_DIR / config).glob(f"*/{META_NAME}")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        pred = np.load(meta_path.parent / PREDICTIONS_NAME)
        rows = index.loc[pred["chave_texto"]]
        keep = np.ones(len(rows), bool) if head == "texto" else rows["usa_funcao"].to_numpy(bool)
        lp = pred[f"{head}_0"].astype(np.float32)[keep]
        rows = rows[keep]
        free = free_readings(lp, alphabet)
        if head == "texto":
            ok = [text_matches(f, t, c) for f, t, c in zip(free, rows["texto"], rows["classe"])]
        else:
            ok = [f == t for f, t in zip(free, rows["funcao"])]
        lps.append(lp)
        lots += [meta["lote_avaliado"]] * len(rows)
        texts += free
        correct += ok
    return {"lp": np.concatenate(lps), "lote": np.array(lots), "texto": texts,
            "acerto": np.array(correct, bool), "alfabeto": alphabet}


def main() -> None:
    from meter_reader.config import get_work_dir
    from training.reader_dataset import load_items

    config = sys.argv[1] if len(sys.argv) > 1 else CONFIG
    items, _ = load_items()
    final = {}
    for head in ("texto", "funcao"):
        d = head_data(config, items, head)
        rows = []
        for lot in sorted(set(d["lote"])):
            test, fit = d["lote"] == lot, d["lote"] != lot
            t = fit_temperature(d["lp"][fit], [x for x, m in zip(d["texto"], fit) if m],
                                d["acerto"][fit], d["alfabeto"])
            texts = [x for x, m in zip(d["texto"], test) if m]
            before = free_confidence(d["lp"][test], texts, d["alfabeto"], 1.0)
            after = free_confidence(d["lp"][test], texts, d["alfabeto"], t)
            rows.append({"lote": lot, "itens": int(test.sum()), "T": round(t, 3),
                         "acerto": round(float(d["acerto"][test].mean()), 3),
                         "conf_media_T1": round(float(before.mean()), 3),
                         "ECE_T1": round(ece(before, d["acerto"][test]), 4),
                         "conf_media_T": round(float(after.mean()), 3),
                         "ECE_T": round(ece(after, d["acerto"][test]), 4)})
        final[head] = round(fit_temperature(d["lp"], d["texto"], d["acerto"], d["alfabeto"]), 4)
        print(f"\nCabeça {head} (validação aninhada por lote):")
        print(pd.DataFrame(rows).to_string(index=False))
        print(f"Temperatura final ({head}, três lotes): {final[head]}")

    path = get_work_dir() / TEMPERATURE_NAME
    path.write_text(json.dumps({"configuracao": config, **final}, indent=1), encoding="utf-8")
    print(f"\nGravado em {TEMPERATURE_NAME}")


if __name__ == "__main__":
    main()
