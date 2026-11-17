"""Amostra da anotação de caixas do detector (ET5a, T5.2).

População (A1 a): as fotos pareadas que passam pela triagem C4, ou seja, as que
o detector vê em produção.
- Desenvolvimento: logits fora da dobra da execução R3 do C4, com a calibração
  aninhada por lote e o limiar do artefato (estimativa para fotos não vistas).
- Teste 03/07: logits gravados na abertura única (E68), com a temperatura e o
  limiar do artefato. Nada é recalculado nem ajustado.
Saídas impressas: só contagens, sem nomes nem caminhos.
"""
from __future__ import annotations

import json
import random

import numpy as np
import pandas as pd

from meter_reader.config import get_work_dir
from training.build_manifest import file_sha256
from training.build_triage import ARTIFACT_NAME
from training.group_dataset import LABEL_VERSIONS, TEST_LABELS, load_group_photos
from training.run_final import FINAL_FOLDER, VERSION
from training.run_finetune import eval_order, run_name
from training.triage_eval import (DETERMINED, calibrate_nested, class_log_probs,
                                  load_oof, rejection_score)

TEST_LOGITS = "teste_0307_c4_final"  # gravado no Bloco código 21 da pesquisa (E68)

# Plano amostral (T5.2): A3 (b), A4 (a), A5 (c+), opção (iii)
DEV_SIZE = 3000
TEST_SIZE = 1200
MIN_PER_STRATUM = 60
FULL_GROUPS = ("cic_duas_leituras",)                    # única fonte do display_kvarh
DISTANT_CONDITIONS = ("distante_parede", "distante_poste")  # perdas da triagem (E68)
SAMPLE_NAME = "amostra_caixas.csv"
SAMPLE_COLUMNS = ("ordem", "pseudonimo", "conjunto", "lote", "grupo", "condicao",
                  "determinado", "estrato", "peso")
# Ampliação do teste antes da primeira leitura pelo detector (ET5b, D6 b): as
# próximas fotos da mesma ordem fixa dos determinados. A versão original nunca muda.
TEST_EXTENSION = 600
SAMPLE_V2_NAME = "amostra_caixas_v2.csv"
TEST_DETERMINED_SEED = "caixas:teste:determinados"
EXTENSION_ORDER_SEED = "caixas:ordem:ampliacao_teste"


def passes(score: np.ndarray, threshold: float) -> np.ndarray:
    """Segue para o detector quem não é rejeitado (rejeita-se com score >= limiar)."""
    return np.asarray(score) < threshold


def artifact_record(work_dir) -> dict:
    """Temperatura e limiar do artefato do produto (JSON de procedência)."""
    path = (work_dir / FINAL_FOLDER / ARTIFACT_NAME).with_suffix(".json")
    return json.loads(path.read_text(encoding="utf-8"))


def meter_type(probs: np.ndarray) -> np.ndarray:
    """Tipo da triagem de quem segue: o mais provável entre digital e ciclométrico, como no
    produto (meter_reader.triage); é a referência do Validador 2 (ET5b)."""
    return np.where(np.asarray(probs)[:, 0] >= np.asarray(probs)[:, 1], "digital", "ciclometrico")


def _frame(photos, passed: np.ndarray, subset: str, probs: np.ndarray) -> pd.DataFrame:
    labels = np.array([p.product_class for p in photos])
    return pd.DataFrame({
        "pseudonimo": [p.pseudonym for p in photos],
        "conjunto": subset,
        "lote": [p.lot for p in photos],
        "grupo": [p.group for p in photos],
        "condicao": [p.condition for p in photos],
        "determinado": np.isin(labels, DETERMINED),
        "passa": passed,
        "tipo_triagem": meter_type(probs),
    })


def population() -> pd.DataFrame:
    """Uma linha por foto pareada do desenvolvimento e do teste, com a coluna `passa`."""
    work_dir = get_work_dir()
    record = artifact_record(work_dir)
    threshold, temperature = record["limiar"], record["temperatura"]

    # Desenvolvimento: mesma ordem e mesmos logits da calibração do artefato
    dev = load_group_photos(LABEL_VERSIONS[VERSION])
    dev = [dev[i] for i in eval_order(dev)]
    labels = np.array([p.product_class for p in dev])
    lots = np.array([p.lot for p in dev])
    logits = load_oof(work_dir, run_name("C4", "W-b", VERSION), [p.pseudonym for p in dev])
    dev_probs = calibrate_nested(logits, labels, lots).probs
    dev_score = rejection_score(dev_probs)

    # Teste: logits antes da temperatura, gravados uma única vez
    test = [p for p in load_group_photos(TEST_LABELS) if p.paired]
    raw = load_oof(work_dir, TEST_LOGITS, [p.pseudonym for p in test])
    test_probs = np.exp(class_log_probs(raw, temperature))
    test_score = rejection_score(test_probs)

    return pd.concat([_frame(dev, passes(dev_score, threshold), "desenvolvimento", dev_probs),
                      _frame(test, passes(test_score, threshold), "teste", test_probs)],
                     ignore_index=True)


def check(pop: pd.DataFrame) -> pd.DataFrame:
    """Perdas e cobertura por conjunto: devem reproduzir a E64 (dev) e a E68 (teste)."""
    rows = {}
    for subset, part in pop.groupby("conjunto"):
        det, rej = part[part["determinado"]], part[~part["determinado"]]
        rows[subset] = {"determinados": len(det), "perdas": int((~det["passa"]).sum()),
                        "rejeitaveis": len(rej), "rejeitaveis_que_passam": int(rej["passa"].sum()),
                        "cobertura": round(1 - rej["passa"].mean(), 4)}
    return pd.DataFrame(rows).T


def strata_counts(pop: pd.DataFrame) -> pd.DataFrame:
    """Fotos que passam por grupo × condição, no desenvolvimento e no teste."""
    passed = pop[pop["passa"]]
    table = passed.pivot_table(index=["grupo", "condicao"], columns="conjunto",
                               values="pseudonimo", aggfunc="count", fill_value=0)
    return table.sort_values("desenvolvimento", ascending=False)


def stratum(group: str, condition: str) -> str:
    return f"{group}|{condition}"


def allocate(sizes: dict[str, int], budget: int, minimum: int, full: set[str]) -> dict[str, int]:
    """Estratos inteiros (os de `full` e os de até `minimum` fotos) e, nos demais,
    o mínimo mais a parte proporcional ao excedente, com arredondamento pelo maior resto."""
    taken = {s: n for s, n in sizes.items() if s in full or n <= minimum}
    rest = {s: n for s, n in sizes.items() if s not in taken}
    left = budget - sum(taken.values()) - minimum * len(rest)
    excess = {s: n - minimum for s, n in rest.items()}
    if left < 0 or left > sum(excess.values()):
        raise ValueError("Orçamento incompatível com os estratos inteiros e os mínimos.")
    quota = {s: left * e / sum(excess.values()) for s, e in excess.items()}
    alloc = {s: minimum + int(q) for s, q in quota.items()}
    short = budget - sum(taken.values()) - sum(alloc.values())
    # Maior resto; empate decidido pelo nome do estrato (determinístico)
    for s in sorted(quota, key=lambda s: (-(quota[s] - int(quota[s])), s))[:short]:
        alloc[s] += 1
    return {**taken, **alloc}


def shuffled(codes, seed: str) -> list[str]:
    """Ordem aleatória completa e fixa: ordena antes, para não depender da ordem de entrada."""
    codes = sorted(codes)
    random.Random(seed).shuffle(codes)
    return codes


def _take(frame: pd.DataFrame, n: int, seed: str) -> pd.DataFrame:
    """As n primeiras fotos da ordem fixa, com o peso população ÷ sorteadas."""
    codes = shuffled(frame["pseudonimo"], seed)[:n]
    return pd.DataFrame({"pseudonimo": codes, "peso": len(frame) / n})


def draw(pop: pd.DataFrame) -> pd.DataFrame:
    """Amostra do desenvolvimento (estratificada) e do teste (aleatória), com pesos e ordem."""
    pop = pop[pop["passa"]].copy()
    pop["estrato"] = [stratum(g, c) for g, c in zip(pop["grupo"], pop["condicao"])]
    parts = []
    for subset, size in (("desenvolvimento", DEV_SIZE), ("teste", TEST_SIZE)):
        part = pop[pop["conjunto"] == subset]
        det, rej = part[part["determinado"]], part[~part["determinado"]]
        parts.append(rej[["pseudonimo"]].assign(peso=1.0))  # A4 (a): todos os rejeitáveis
        budget = size - len(rej)
        if subset == "teste":  # sem mínimo por estrato: aleatória entre os determinados
            parts.append(_take(det, budget, TEST_DETERMINED_SEED))
            continue
        sizes = det["estrato"].value_counts().to_dict()
        full = {s for s in sizes if s.split("|")[0] in FULL_GROUPS
                or s.split("|")[1] in DISTANT_CONDITIONS}
        for s, n in allocate(sizes, budget, MIN_PER_STRATUM, full).items():
            parts.append(_take(det[det["estrato"] == s], n, f"caixas:{subset}:{s}"))
    sample = pop.merge(pd.concat(parts, ignore_index=True), on="pseudonimo")
    order = {c: i + 1 for i, c in enumerate(shuffled(sample["pseudonimo"], "caixas:ordem"))}
    sample["ordem"] = sample["pseudonimo"].map(order)
    return sample.sort_values("ordem")[list(SAMPLE_COLUMNS)].reset_index(drop=True)


def save_or_check(sample: pd.DataFrame, work_dir, name: str = SAMPLE_NAME) -> str:
    """Grava a amostra na pasta de trabalho; se já existir, confere que é idêntica."""
    path = work_dir / name
    if path.exists():
        # round_trip: o leitor padrão pode errar o último dígito dos pesos
        saved = pd.read_csv(path, dtype={"determinado": bool}, float_precision="round_trip")
        if not saved.equals(sample.astype(saved.dtypes.to_dict())):
            raise ValueError("A amostra gravada difere do sorteio; nada foi alterado.")
        print("Amostra já gravada; conferida: idêntica ao sorteio.")
    else:
        sample.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
        print("Amostra gravada.")
    return file_sha256(path)

def extend_test(sample: pd.DataFrame, pop: pd.DataFrame, extra: int) -> pd.DataFrame:
    """Amplia os determinados do teste com as próximas `extra` fotos da mesma ordem fixa.

    As fotos já sorteadas continuam, com o peso recalculado (população ÷ sorteadas);
    as novas entram no fim da ordem de anotação, numa ordem aleatória própria. O
    desenvolvimento e os rejeitáveis do teste não mudam.
    """
    passed = pop[pop["passa"] & (pop["conjunto"] == "teste") & pop["determinado"]]
    old = (sample["conjunto"] == "teste") & sample["determinado"]
    n_old = int(old.sum())
    if n_old + extra > len(passed):
        raise ValueError("Ampliação maior que a população de determinados do teste.")
    codes = shuffled(passed["pseudonimo"], TEST_DETERMINED_SEED)[: n_old + extra]
    if set(codes[:n_old]) != set(sample.loc[old, "pseudonimo"]):
        raise ValueError("A ordem fixa não reproduz o sorteio original; nada foi gravado.")
    weight = len(passed) / len(codes)

    out = sample.copy()
    out.loc[old, "peso"] = weight
    new = passed[passed["pseudonimo"].isin(codes[n_old:])].copy()
    new["estrato"] = [stratum(g, c) for g, c in zip(new["grupo"], new["condicao"])]
    new["peso"] = weight
    start = int(out["ordem"].max())
    order = shuffled(new["pseudonimo"], EXTENSION_ORDER_SEED)
    new["ordem"] = new["pseudonimo"].map({c: start + i + 1 for i, c in enumerate(order)})
    new = new.sort_values("ordem")[list(SAMPLE_COLUMNS)]
    return pd.concat([out, new], ignore_index=True)

if __name__ == "__main__":
    import sys

    pd.set_option("display.width", 200, "display.max_rows", 200)
    pop = population()
    print(check(pop), end="\n\n")
    work_dir = get_work_dir()
    if sys.argv[1:] == ["--ampliar-teste"]:
        # Parte da amostra gravada e confere que ela reproduz o sorteio original
        v1 = pd.read_csv(work_dir / SAMPLE_NAME, dtype={"determinado": bool},
                         float_precision="round_trip")
        if not v1.equals(draw(pop).astype(v1.dtypes.to_dict())):
            raise SystemExit("A amostra gravada difere do sorteio; nada foi alterado.")
        sample = extend_test(v1, pop, TEST_EXTENSION)
        name = SAMPLE_V2_NAME
    else:
        sample = draw(pop)
        name = SAMPLE_NAME
        summary = sample.groupby(["conjunto", "estrato"]).agg(fotos=("pseudonimo", "size"),
                                                              peso=("peso", "first"))
        print(summary, end="\n\n")
    print(sample.groupby("conjunto").agg(fotos=("pseudonimo", "size"), soma_pesos=("peso", "sum")))
    print(f"SHA-256 de {name}: {save_or_check(sample, work_dir, name)}")
