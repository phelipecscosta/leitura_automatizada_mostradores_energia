"""Testes da amostra da anotação de caixas (T5.2), com populações sintéticas."""
import pandas as pd
import pytest

from training import box_sample as bs


def _pop(strata: dict[tuple[str, str, bool], int], subset: str = "desenvolvimento",
         passa: bool = True, start: int = 0) -> pd.DataFrame:
    """Uma linha por foto: (grupo, condição, determinado) -> quantidade."""
    rows, i = [], start
    for (group, condition, determined), n in strata.items():
        for _ in range(n):
            rows.append({"pseudonimo": f"F-{i:012x}", "conjunto": subset, "lote": "2026-05-20",
                         "grupo": group, "condicao": condition, "determinado": determined,
                         "passa": passa})
            i += 1
    return pd.DataFrame(rows)


@pytest.fixture
def small(monkeypatch):
    """Orçamentos pequenos: 100 no desenvolvimento e 40 no teste, com mínimo de 10."""
    monkeypatch.setattr(bs, "DEV_SIZE", 100)
    monkeypatch.setattr(bs, "TEST_SIZE", 40)
    monkeypatch.setattr(bs, "MIN_PER_STRATUM", 10)
    dev = _pop({("dig1_quadrado_branco", "raiz", True): 300,
                ("dig1_quadrado_branco", "visor", True): 60,
                ("dig1_quadrado_branco", "distante_parede", True): 15,
                ("cic_duas_leituras", "raiz", True): 12,
                ("dig5_pequenino", "raiz", True): 4,
                ("indeterminado", "raiz", False): 9})
    blocked = _pop({("dig1_quadrado_branco", "raiz", True): 50}, passa=False, start=1000)
    test = _pop({("dig1_quadrado_branco", "raiz", True): 70,
                 ("multiplos", "raiz", False): 5}, subset="teste", start=2000)
    return pd.concat([dev, blocked, test], ignore_index=True)


def test_passes_uses_strict_threshold():
    # Rejeita-se com score >= limiar: o empate não passa
    assert bs.passes([0.1, 0.5, 0.9], 0.5).tolist() == [True, False, False]


def test_allocate_exact_total_minimum_and_full_strata():
    sizes = {"a": 1000, "b": 200, "c": 50, "d": 30}
    alloc = bs.allocate(sizes, budget=400, minimum=60, full={"c"})
    assert sum(alloc.values()) == 400
    assert alloc["c"] == 50 and alloc["d"] == 30          # inteiro e pequeno
    assert alloc["a"] >= 60 and alloc["b"] >= 60          # mínimo garantido
    assert alloc["a"] > alloc["b"]                        # proporcional ao excedente
    assert all(alloc[s] <= sizes[s] for s in sizes)


def test_allocate_largest_remainder_is_deterministic():
    sizes = {"a": 101, "b": 101, "c": 101}
    first = bs.allocate(sizes, budget=50, minimum=10, full=set())
    assert sum(first.values()) == 50
    assert first == bs.allocate(dict(reversed(list(sizes.items()))), 50, 10, set())


def test_allocate_rejects_impossible_budget():
    with pytest.raises(ValueError):
        bs.allocate({"a": 100, "b": 100}, budget=10, minimum=60, full=set())
    with pytest.raises(ValueError):
        bs.allocate({"a": 100}, budget=500, minimum=60, full=set())


def test_shuffled_ignores_input_order():
    codes = [f"F-{i:012x}" for i in range(50)]
    assert bs.shuffled(codes, "x") == bs.shuffled(list(reversed(codes)), "x")
    assert bs.shuffled(codes, "x") != bs.shuffled(codes, "y")


def test_draw_sizes_weights_and_rules(small):
    sample = bs.draw(small)
    dev = sample[sample["conjunto"] == "desenvolvimento"]
    test = sample[sample["conjunto"] == "teste"]
    assert len(dev) == 100 and len(test) == 40
    # Só fotos que passam pela triagem; a soma dos pesos reproduz a população
    assert not sample["pseudonimo"].isin(small.loc[~small["passa"], "pseudonimo"]).any()
    assert dev["peso"].sum() == pytest.approx(400)
    assert test["peso"].sum() == pytest.approx(75)
    # Inteiros, com peso 1: rejeitáveis, duas leituras, distantes e estratos pequenos
    for s in ("indeterminado|raiz", "cic_duas_leituras|raiz",
              "dig1_quadrado_branco|distante_parede", "dig5_pequenino|raiz"):
        part = dev[dev["estrato"] == s]
        assert (part["peso"] == 1.0).all()
    assert (dev["estrato"] == "indeterminado|raiz").sum() == 9
    assert (test["estrato"] == "multiplos|raiz").sum() == 5
    # Teste sem mínimo por estrato: todos os determinados com o mesmo peso
    assert test.loc[test["determinado"], "peso"].nunique() == 1


def test_draw_order_and_reproducibility(small):
    sample = bs.draw(small)
    assert sample["ordem"].tolist() == list(range(1, len(sample) + 1))
    assert sample["pseudonimo"].is_unique
    assert tuple(sample.columns) == bs.SAMPLE_COLUMNS
    shuffled_input = small.sample(frac=1, random_state=3)
    assert sample.equals(bs.draw(shuffled_input))


def test_save_or_check_round_trip_and_refusal(small, tmp_path):
    sample = bs.draw(small)
    digest = bs.save_or_check(sample, tmp_path)
    # A segunda chamada confere (inclusive os pesos, lidos sem erro de arredondamento)
    assert bs.save_or_check(sample, tmp_path) == digest
    changed = sample.copy()
    changed.loc[0, "peso"] += 1e-12
    with pytest.raises(ValueError):
        bs.save_or_check(changed, tmp_path)
    assert bs.save_or_check(sample, tmp_path) == digest  # o arquivo não foi alterado

def test_extend_test_keeps_the_draw_and_appends_in_order(small):
    sample = bs.draw(small)
    extended = bs.extend_test(sample, small, extra=20)
    old = sample[(sample["conjunto"] == "teste") & sample["determinado"]]
    new = extended[(extended["conjunto"] == "teste") & extended["determinado"]]
    assert set(old["pseudonimo"]) < set(new["pseudonimo"]) and len(new) == len(old) + 20
    # As linhas antigas continuam, na mesma ordem; só o peso dos determinados do teste muda
    pd.testing.assert_frame_equal(extended.iloc[:len(sample)].drop(columns="peso"),
                                  sample.drop(columns="peso"))
    assert extended["ordem"].tolist() == list(range(1, len(extended) + 1))
    assert new["peso"].nunique() == 1
    assert extended.loc[extended["conjunto"] == "teste", "peso"].sum() == pytest.approx(75)
    # As novas são as próximas da mesma ordem fixa
    passed = small[(small["conjunto"] == "teste") & small["determinado"]]
    expected = bs.shuffled(passed["pseudonimo"], bs.TEST_DETERMINED_SEED)[:len(new)]
    assert set(new["pseudonimo"]) == set(expected)


def test_extend_test_rejects_more_than_the_population(small):
    with pytest.raises(ValueError):
        bs.extend_test(bs.draw(small), small, extra=1000)
