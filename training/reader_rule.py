"""Regra de escolha do leitor (protocolo do leitor, seções 1.5, 1.8 e 1.9; R2).

1. Latência em CPU: 4 threads, lote de 1 recorte, com o pré-processamento, em 200
   recortes fixos (permutação com semente 0), depois de 20 de aquecimento; teto de
   50 ms no p95. Os pesos não mudam a latência; usa-se a rede sem treino.
2. Bootstrap pareado por recorte, estratificado por classe (2.000 reamostragens), com
   os limiares fixos de cada semente: os mesmos recortes valem para todos os
   candidatos e sementes.
3. Vence o mais rápido cujo limite inferior do IC de 95% da diferença de M-c para o
   melhor fique acima de -3 pontos (Delta = 3); sai quem passar do teto.
Só números.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch

from training.reader_metrics import CLASSES, config_table, covered

DELTA = 0.03
LATENCY_CEILING_MS = 50.0
N_BOOT, BOOT_SEED = 2000, 0
LATENCY_ITEMS, LATENCY_WARMUP, LATENCY_THREADS = 200, 20, 4


def coverage_matrix(scores: pd.DataFrame, table: pd.DataFrame) -> dict[str, np.ndarray]:
    """Por classe: matriz (sementes x recortes) de cobertura, com os recortes na ordem
    das chaves, para parear entre candidatos."""
    out = {}
    for cls in CLASSES:
        rows = []
        for seed in sorted(scores["semente"].unique()):
            part = scores[(scores["semente"] == seed) & (scores["classe"] == cls)]
            part = part.sort_values("chave")
            log_t = table.loc[(table["semente"] == seed) & (table["classe"] == cls),
                              "log_limiar"].iloc[0]
            rows.append(covered(part, log_t))
        out[cls] = np.vstack(rows).astype(float)
    return out


def bootstrap_mc(matrices: dict[str, dict[str, np.ndarray]], n_boot: int = N_BOOT,
                 seed: int = BOOT_SEED) -> dict[str, np.ndarray]:
    """M-c reamostrada de cada candidato, com os mesmos pesos de recorte para todos."""
    rng = np.random.default_rng(seed)
    configs = list(matrices)
    out = {c: np.zeros(n_boot) for c in configs}
    for cls in CLASSES:
        n = matrices[configs[0]][cls].shape[1]
        weights = rng.multinomial(n, np.full(n, 1 / n), size=n_boot)   # (n_boot, n)
        for c in configs:
            cov = matrices[c][cls]                                      # (sementes, n)
            per_seed = weights @ cov.T / n                              # (n_boot, sementes)
            out[c] += per_seed.mean(axis=1) / len(CLASSES)
    return out


def latency_ms(candidate: str, crops: list, threads: int = LATENCY_THREADS,
               warmup: int = LATENCY_WARMUP) -> dict:
    """p50 e p95 em CPU, lote de 1, com o pré-processamento (realce e 48 x 192)."""
    from meter_reader.reader import Reader
    from meter_reader.reader_input import reader_tensor

    torch.set_num_threads(threads)
    model = Reader(candidate, pretrained=False).eval()
    times = []
    with torch.no_grad():
        for i, crop in enumerate(crops):
            start = time.perf_counter()
            model(reader_tensor(crop).unsqueeze(0))
            if i >= warmup:
                times.append((time.perf_counter() - start) * 1000)
    return {"p50": float(np.percentile(times, 50)), "p95": float(np.percentile(times, 95))}


def choose(mc: dict[str, float], boot: dict[str, np.ndarray],
           p95: dict[str, float]) -> tuple[str, pd.DataFrame]:
    """Aplica a regra da seção 1.9 e devolve o vencedor e a tabela da decisão."""
    best = max(mc, key=mc.get)
    rows = []
    for c in sorted(mc, key=lambda k: p95[k]):
        diff = boot[c] - boot[best]
        low, high = np.percentile(diff, [2.5, 97.5])
        rows.append({"candidato": c, "p95_ms": round(p95[c], 1), "M-c": round(mc[c], 4),
                     "dif_para_melhor": round(mc[c] - mc[best], 4),
                     "ic_inferior": round(float(low), 4), "ic_superior": round(float(high), 4),
                     "dentro_do_teto": p95[c] <= LATENCY_CEILING_MS,
                     "nao_inferior": c == best or low > -DELTA})
    table = pd.DataFrame(rows)
    ok = table[table["dentro_do_teto"] & table["nao_inferior"]]
    return (ok.iloc[0]["candidato"] if len(ok) else ""), table


def main() -> None:
    from meter_reader.config import get_work_dir
    from meter_reader.reader import CANDIDATES
    from training.reader_cache import CACHE_DIR, CACHE_NAME
    from training.reader_dataset import load_items
    from training.reader_eval import score_config
    from training.reader_labels import CROPS_SHA256
    from training.reader_metrics import summary
    from training.reader_torch import ReaderDataset

    work_dir = get_work_dir()
    items, _ = load_items()
    mc, matrices = {}, {}
    for c in CANDIDATES:
        scores = score_config(work_dir, c, items)
        table = config_table(scores)
        mc[c] = summary(table)["M-c"]
        matrices[c] = coverage_matrix(scores, table)
    boot = bootstrap_mc(matrices)

    text = items[items["usa_texto"]].reset_index(drop=True)
    order = np.random.default_rng(0).permutation(len(text))[:LATENCY_ITEMS + LATENCY_WARMUP]
    data = ReaderDataset(text, work_dir / CACHE_DIR / CACHE_NAME, CROPS_SHA256, train=False)
    crops = [data.image(int(i)) for i in order]
    p95 = {}
    for c in CANDIDATES:
        lat = latency_ms(c, crops)
        p95[c] = lat["p95"]
        print(f"{c}: latência p50 {lat['p50']:.1f} ms; p95 {lat['p95']:.1f} ms")

    winner, table = choose(mc, boot, p95)
    with pd.option_context("display.width", 160):
        print("\n" + table.to_string(index=False))
    print(f"\nVencedor pela regra da seção 1.9: {winner or 'nenhum'}")


if __name__ == "__main__":
    main()

    
