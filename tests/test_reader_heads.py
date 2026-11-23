from training.reader_heads import fb_trigger, function_category


def test_categorias_da_funcao():
    assert [function_category(c) for c in ("", "03", "103", "0312", "07")] == [
        "vazio", "03", "103", "outro", "outro"]


def test_gatilho_da_fb():
    labels = ["03"] * 96 + ["103"] * 4
    ok = fb_trigger(labels, list(labels))
    assert ok["acerto"] == 1.0 and not ok["dispara"]
    reads = ["03"] * 96 + ["03", "103", "103", "103"]        # 1 de 4 dos 103 lido como 03
    out = fb_trigger(labels, reads)
    assert out["103_como_03"] == 0.25 and out["dispara"]
    low = fb_trigger(["03"] * 10, ["03"] * 9 + ["24"])        # acerto de 90%
    assert low["dispara"]
