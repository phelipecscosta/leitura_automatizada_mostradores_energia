"""Gera a amostra de rotulagem do Lab02 a partir do manifesto (T2.2).

Regras (E08, E11 e decisões da T2.2):
- partição por lote, em ordem cronológica: os dois primeiros lotes são
  treino, o terceiro é validação e o último é teste;
- cada lote recebe uma ordem aleatória completa e fixa, e a rotulagem segue
  sempre do início da lista, para que uma ampliação futura continue aleatória;
- universo: só imagens íntegras; no treino, pareadas e órfãs; na validação e
  no teste, só pareadas (em produção, as órfãs são rejeitadas antes da triagem);
- conjunto de estresse do V4: zeros sem nota que não foram sorteados para a
  rotulagem, com ordem aleatória própria.
"""

from __future__ import annotations

import random

SEED = 42
SETS_BY_RANK = ("treino", "treino", "validacao", "teste")  # ordem cronológica
LABELS_PER_LOT = 75
BLIND_PER_LOT = (13, 13, 12, 12)  # soma 50, na ordem cronológica dos lotes
ZERO_STRESS_SIZE = 50

ROLE_BLIND = "cega"
ROLE_LABEL = "rotulagem"
ROLE_ZERO = "estresse_zero"


def assign_sets(lots: set[str]) -> dict[str, str]:
    """Associa cada lote (data AAAA-MM-DD) ao seu conjunto, pela ordem cronológica."""
    ordered = sorted(lots)  # o formato ISO ordena como data
    if len(ordered) != len(SETS_BY_RANK):
        raise ValueError("A partição exige exatamente 4 lotes.")
    return dict(zip(ordered, SETS_BY_RANK))


def _in_universe(row: dict, subset: str) -> bool:
    """Imagem elegível para o conjunto: íntegra; órfã só no treino."""
    if row["integridade"] != "":
        return False
    return subset == "treino" or row["pareada"] == "True"


def _shuffled(names: list[str], key: str) -> list[str]:
    """Ordem aleatória fixa. Ordena antes de embaralhar, para não depender da
    ordem de entrada; a semente em texto dá a cada lista a sua sequência."""
    ordered = sorted(names)
    random.Random(f"{SEED}:{key}").shuffle(ordered)
    return ordered


def build_sample(rows: list[dict]) -> list[dict]:
    """Uma linha por imagem do universo, com conjunto, posição e papel.

    `rows` são as linhas do manifesto, lidas como texto.
    """
    sets = assign_sets({r["lote"] for r in rows})
    sample = []
    for rank, (lot, subset) in enumerate(sets.items()):
        names = [r["arquivo"] for r in rows if r["lote"] == lot and _in_universe(r, subset)]
        for position, name in enumerate(_shuffled(names, lot), start=1):
            if position <= BLIND_PER_LOT[rank]:
                role = ROLE_BLIND
            elif position <= LABELS_PER_LOT:
                role = ROLE_LABEL
            else:
                role = ""  # não sorteada
            sample.append({
                "lote": lot, "conjunto": subset, "posicao": position,
                "arquivo": name, "papel": role, "posicao_estresse": "",
            })

    # Estresse do V4: zeros sem nota, íntegros e pareados, fora da rotulagem
    labeled = {s["arquivo"] for s in sample if s["papel"]}
    zeros = [
        r["arquivo"] for r in rows
        if r["integridade"] == "" and r["pareada"] == "True"
        and r["leitura"] == "0" and r["nota"] == "" and r["arquivo"] not in labeled
    ]
    zero_position = {n: i for i, n in enumerate(_shuffled(zeros, "zeros"), start=1)}
    for s in sample:
        position = zero_position.get(s["arquivo"])
        if position is not None:
            s["posicao_estresse"] = position
            if position <= ZERO_STRESS_SIZE:
                s["papel"] = ROLE_ZERO
    return sample
