import random

from training.reader_audit import BLOCK, TRAPS, build_blocks, make_trap, swap_char


def test_troca_um_caractere_de_preferencia_parecido():
    rng = random.Random(0)
    for value in ("12345", "AB12", "9999", "XYZW"):
        out = swap_char(value, rng)
        assert out != value and len(out) == len(value)
        assert sum(a != b for a, b in zip(out, value)) == 1
    assert swap_char("8", rng) == "0"


def test_armadilha_muda_so_um_campo():
    row = {"pseudonimo": "p", "leitura": 12345, "numero": "AB12", "funcao": "03"}
    t = make_trap(row, random.Random(1))
    changed = [k for k in ("leitura", "numero") if str(t[k]) != str(row[k])]
    assert changed == [t["campo_trocado"]] and t["tipo"] == "armadilha"


def test_blocos_com_tres_armadilhas_e_reserva():
    real = [{"pseudonimo": f"r{i}", "leitura": 100 + i, "numero": f"N{i}", "funcao": ""}
            for i in range(120)]
    pool = [{"pseudonimo": f"t{i}", "leitura": 500 + i, "numero": f"M{i}", "funcao": "03"}
            for i in range(40)]
    items, reserve = build_blocks(real, pool)
    blocks = {}
    for it in items:
        blocks.setdefault(it["bloco"], []).append(it)
    assert len(blocks) == 3
    for b in blocks.values():
        assert sum(i["tipo"] == "armadilha" for i in b) == TRAPS
        assert sum(i["tipo"] == "real" for i in b) <= BLOCK
    assert sorted(i["pseudonimo"] for i in items if i["tipo"] == "real") == \
        sorted(r["pseudonimo"] for r in real)
    assert len(reserve) == 3 * TRAPS * 2
    used = {i["pseudonimo"] for i in items if i["tipo"] == "armadilha"}
    assert not used & {r["pseudonimo"] for r in reserve}
    assert build_blocks(real, pool)[0] == items          # sementes fixas: mesmo sorteio
