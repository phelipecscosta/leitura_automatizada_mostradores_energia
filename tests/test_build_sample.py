"""Testes da amostra de rotulagem do Lab02 (training/build_sample.py)."""

import random

import pytest

from training import build_sample as bs

LOTS = ("2000-01-01", "2000-01-02", "2000-01-03", "2000-02-01")


def make_rows(per_lot: int = 100) -> list[dict]:
    """Manifesto sintético com corrompidas, órfãs, zeros com e sem nota."""
    rows = []
    for lot in LOTS:
        for i in range(per_lot):
            orphan = i % 10 == 1
            rows.append({
                "lote": lot,
                "arquivo": f"{lot}_{i:03d}.jpg",
                "integridade": "vazia: preta" if i % 25 == 0 else "",
                "pareada": str(not orphan),
                "leitura": "" if orphan else ("0" if i % 7 == 0 else "123"),
                "nota": "T181" if not orphan and i % 14 == 0 else "",
            })
    return rows


def test_assign_sets_requires_four_lots():
    with pytest.raises(ValueError):
        bs.assign_sets({"2000-01-01", "2000-01-02"})


def test_universe_rules():
    rows = make_rows()
    source = {r["arquivo"]: r for r in rows}
    for s in bs.build_sample(rows):
        assert source[s["arquivo"]]["integridade"] == ""  # corrompida nunca entra
        if s["conjunto"] != "treino":
            assert source[s["arquivo"]]["pareada"] == "True"  # órfã só no treino


def test_role_counts_per_lot():
    sample = bs.build_sample(make_rows())
    for rank, lot in enumerate(LOTS):
        lot_rows = [s for s in sample if s["lote"] == lot]
        blind = [s["posicao"] for s in lot_rows if s["papel"] == bs.ROLE_BLIND]
        labeled = [s for s in lot_rows if s["papel"] in (bs.ROLE_BLIND, bs.ROLE_LABEL)]
        assert blind == list(range(1, bs.BLIND_PER_LOT[rank] + 1))  # início da lista
        assert len(labeled) == bs.LABELS_PER_LOT


def test_independent_of_input_order():
    rows = make_rows()
    shuffled = rows[:]
    random.Random(0).shuffle(shuffled)
    assert bs.build_sample(rows) == bs.build_sample(shuffled)


def test_larger_sample_keeps_the_order(monkeypatch):
    # Ampliar a rotulagem não pode mudar a ordem já sorteada
    rows = make_rows()
    order = lambda: [(s["lote"], s["posicao"], s["arquivo"]) for s in bs.build_sample(rows)]
    before = order()
    monkeypatch.setattr(bs, "LABELS_PER_LOT", 80)
    assert order() == before


def test_zero_stress_set(monkeypatch):
    rows = make_rows()
    sample = bs.build_sample(rows)
    eligible = {
        r["arquivo"] for r in rows
        if r["integridade"] == "" and r["pareada"] == "True"
        and r["leitura"] == "0" and r["nota"] == ""
    }
    labeled = {s["arquivo"] for s in sample if s["papel"] in (bs.ROLE_BLIND, bs.ROLE_LABEL)}
    ranked = {s["arquivo"] for s in sample if s["posicao_estresse"] != ""}
    assert ranked == eligible - labeled  # só zeros sem nota, nunca os rotulados

    # O corte respeita o tamanho e pega o início da ordem própria dos zeros.
    # A guarda impede um teste vazio: precisa haver mais zeros que o corte.
    assert len(ranked) > 2
    monkeypatch.setattr(bs, "ZERO_STRESS_SIZE", 2)
    stress = [s for s in bs.build_sample(rows) if s["papel"] == bs.ROLE_ZERO]
    assert sorted(s["posicao_estresse"] for s in stress) == [1, 2]
