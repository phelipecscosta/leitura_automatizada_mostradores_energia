"""Autotreinamento e ablações do leitor, em sequência, pelas regras do protocolo
(seções 1.9 e 1.10; revisões R1 a R9). Cada fase parte da configuração adotada até
ali e decide sozinha, pela regra fixada antes:

1. Autotreinamento (S-a): adotado se a diferença de M-c tiver o IC acima de zero.
   S-b: até duas rodadas a mais, só se a primeira ganhar pelo menos 5 pontos de
   cobertura dos ciclométricos com o IC acima de zero; para quando o ganho de M-c
   tiver o IC incluindo zero.
2. Cinza (PP-a): substitui o RGB se a diferença de M-c tiver o IC acima de zero.
3. Sem `sonnet_csv`: os 52 rótulos saem do treino (não da avaliação) se a diferença
   de M-c tiver o IC acima de zero.
4. Leitor por família (R8): ajuste fino da configuração adotada, só com
   ciclométricos, comparado nos ciclométricos; adotado com pelo menos 5 pontos de
   ganho e o IC acima de zero.

Os treinos concluídos são pulados (retomada). As decisões são gravadas em
`experimentos/leitor/fases.json`. O modo `ensaio` repete tudo com 1 época, 1
semente e o prefixo "ensaio-", para achar defeitos antes do lançamento longo.
Só números.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from training.reader_metrics import CLASSES, config_table
from training.reader_rule import BOOT_SEED, N_BOOT, bootstrap_mc, coverage_matrix
from training.reader_torch import TEXT_BATCH

CANDIDATE = "L2-12"                 # vencedor da regra da seção 1.9 (T6b.5)
CYCLOMETRIC = "display_ciclometrico"
CYC_MARGIN = 0.05
MAX_EXTRA_ROUNDS = 2
REPORT_NAME = "fases.json"


def make_items_for(items: pd.DataFrame, pseudo: dict | None = None, drop_sonnet: bool = False,
                   only_class: str | None = None):
    """Função lote avaliado -> itens da dobra, conforme o estado adotado."""
    from training.reader_pseudo import as_items

    def items_for(lot: str) -> pd.DataFrame:
        df = items
        if pseudo is not None:
            df = pd.concat([df, as_items(pseudo[lot])], ignore_index=True)
        if drop_sonnet:   # sai do treino, fica na avaliação
            df = df[~((df["origem"] == "sonnet_csv") & (df["lote"] != lot))]
        if only_class:
            df = df[(df["classe"] == only_class) & df["usa_texto"]]
        return df.reset_index(drop=True)
    return items_for


def compare(work_dir: Path, items: pd.DataFrame, new: str, ref: str,
            classes=CLASSES) -> dict:
    """Diferença de M-c (nova - referência) nas classes dadas, com o IC de 95% do
    bootstrap pareado (os mesmos recortes nos dois)."""
    from training.reader_eval import score_config

    mats, mc, keys = {}, {}, {}
    for c in (new, ref):
        scores = score_config(work_dir, c, items)
        scores = scores[scores["classe"].isin(classes)]
        table = config_table(scores)
        mats[c] = coverage_matrix(scores, table, classes)
        mc[c] = float(table.groupby("semente")["cobertura"].mean().mean())
        keys[c] = sorted(set(scores["chave"]))
    if keys[new] != keys[ref]:
        raise SystemExit(f"Recortes avaliados diferentes entre {new} e {ref}.")
    boot = bootstrap_mc(mats, N_BOOT, BOOT_SEED, classes)
    low, high = np.percentile(boot[new] - boot[ref], [2.5, 97.5])
    return {"nova": new, "referencia": ref, "dif": round(mc[new] - mc[ref], 4),
            "ic_inferior": round(float(low), 4), "ic_superior": round(float(high), 4),
            "M-c_nova": round(mc[new], 4), "M-c_referencia": round(mc[ref], 4)}


def run_phases(items: pd.DataFrame, prefix: str = "", max_epochs: int | None = None,
               seeds=None) -> dict:
    from meter_reader.config import get_work_dir
    from training.reader_dataset import fold_masks
    from training.reader_pseudo import build
    from training.reader_train import (FINE_TUNE_EPOCHS, FINE_TUNE_LR_FACTOR, LEARNING_RATES,
                                       MAX_EPOCHS, SEEDS, run_grid)

    work_dir = get_work_dir()
    fine_tune_epochs = max_epochs or FINE_TUNE_EPOCHS   # no ensaio, 1 época também
    max_epochs, seeds = max_epochs or MAX_EPOCHS, seeds or SEEDS
    report_path = work_dir / "experimentos" / "leitor" / (prefix + REPORT_NAME)
    log = []

    def note(entry: dict) -> None:
        log.append(entry)
        print(json.dumps(entry, ensure_ascii=False), flush=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")

    def train(name: str, items_for, gray: bool, steps=None, **extra) -> None:
        spec = {"candidato": CANDIDATE, "cinza": gray, **extra}
        if steps:
            spec["passos_por_epoca"] = steps
        run_grid({name: spec}, items, max_epochs=max_epochs, seeds=seeds, items_for=items_for)

    base = prefix + CANDIDATE
    train(base, None, False)                       # base com os rótulos R9
    state = {"config": base, "pseudo": None, "sem_sonnet": False, "cinza": False}

    def current(only_class=None):
        return make_items_for(items, state["pseudo"], state["sem_sonnet"], only_class)

    # 1. Autotreinamento
    source = base
    for rnd in range(1, MAX_EXTRA_ROUNDS + 2):
        pseudo = build(source, items, candidate=CANDIDATE)
        name = f"{base}+auto{rnd}"
        train(name, make_items_for(items, pseudo, state["sem_sonnet"]), state["cinza"])
        result = compare(work_dir, items, name, state["config"])
        adopted = result["ic_inferior"] > 0
        note({"fase": f"autotreinamento, rodada {rnd}", **result, "adotada": adopted,
              "pseudo_rotulos": {lot: len(df) for lot, df in pseudo.items()}})
        if not adopted:
            break
        state.update(config=name, pseudo=pseudo)
        if rnd == 1:
            cyc = compare(work_dir, items, name, base, (CYCLOMETRIC,))
            trigger = cyc["dif"] >= CYC_MARGIN and cyc["ic_inferior"] > 0
            note({"fase": "gatilho da S-b (ciclométricos)", **cyc, "dispara": trigger})
            if not trigger:
                break
        source = name

    # 2. Cinza
    name = state["config"] + "+cinza"
    train(name, current(), True)
    result = compare(work_dir, items, name, state["config"])
    note({"fase": "cinza", **result, "adotada": result["ic_inferior"] > 0})
    if result["ic_inferior"] > 0:
        state.update(config=name, cinza=True)

    # 3. Sem sonnet_csv
    name = state["config"] + "-sonnet"
    train(name, make_items_for(items, state["pseudo"], True), state["cinza"])
    result = compare(work_dir, items, name, state["config"])
    note({"fase": "sem sonnet_csv", **result, "adotada": result["ic_inferior"] > 0})
    if result["ic_inferior"] > 0:
        state.update(config=name, sem_sonnet=True)

    # 4. Leitor por família (R8): ajuste fino a partir da configuração adotada, só com
    # ciclométricos, com os passos por época dela, 1/10 da taxa e 60 épocas
    adopted_for = current()
    steps = {}
    for lot in sorted(items["lote"].unique()):
        fold = adopted_for(lot)
        train_text = fold_masks(fold, lot)["treino"] & fold["usa_texto"].to_numpy(bool)
        steps[lot] = max(1, int(train_text.sum()) // TEXT_BATCH)
    name = state["config"] + "+familia"
    train(name, current(CYCLOMETRIC), state["cinza"], steps, inicio=state["config"],
          taxa=LEARNING_RATES[CANDIDATE] * FINE_TUNE_LR_FACTOR, max_epocas=fine_tune_epochs)
    result = compare(work_dir, items, name, state["config"], (CYCLOMETRIC,))
    family = result["dif"] >= CYC_MARGIN and result["ic_inferior"] > 0
    note({"fase": "leitor por família (ciclométricos)", **result, "adotada": family})

    final = {"fase": "configuração final", "configuracao": state["config"],
             "autotreinamento": state["pseudo"] is not None, "cinza": state["cinza"],
             "sem_sonnet": state["sem_sonnet"], "leitor_por_familia": family}
    note(final)
    return final


def clean_rehearsal() -> None:
    """Apaga só os artefatos do ensaio (prefixo "ensaio-")."""
    from meter_reader.config import get_work_dir
    from training.reader_eval import EVAL_DIR
    from training.reader_pseudo import PSEUDO_DIR
    from training.reader_train import RUNS_DIR

    work_dir = get_work_dir()
    for folder in (work_dir / RUNS_DIR).glob("ensaio-*"):
        shutil.rmtree(folder)
    for pattern in (EVAL_DIR / "ensaio-*.csv", PSEUDO_DIR / "ensaio-*.csv"):
        for f in work_dir.glob(str(pattern)):
            f.unlink()
    report = work_dir / "experimentos" / "leitor" / ("ensaio-" + REPORT_NAME)
    if report.exists():
        report.unlink()
    print("Artefatos do ensaio apagados.")


def main() -> None:
    from training.reader_dataset import load_items

    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "limpar_ensaio":
        clean_rehearsal()
        return
    items, _ = load_items()
    if command == "ensaio":
        run_phases(items, prefix="ensaio-", max_epochs=1, seeds=(0,))
    elif command == "executar":
        run_phases(items)
    else:
        raise SystemExit("Use: ensaio | limpar_ensaio | executar")


if __name__ == "__main__":
    main()
