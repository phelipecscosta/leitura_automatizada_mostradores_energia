import json

from training.reader_runs import read_runs, summarize


def _meta(root, config, lot, seed, epoch, ma):
    folder = root / config / f"{lot}_s{seed}"
    folder.mkdir(parents=True)
    (folder / "meta.json").write_text(json.dumps({
        "configuracao": config, "lote_avaliado": lot, "semente": seed,
        "epoca_escolhida": epoch, "epocas_max": 240, "ma_interna": ma,
        "segundos_total": 600}), encoding="utf-8")


def test_resumo_e_alarme(tmp_path):
    for seed, epoch in enumerate((220, 230, 100)):
        _meta(tmp_path, "L1", "A", seed, epoch, 0.1 + seed / 10)
    _meta(tmp_path, "L2-9", "A", 0, 150, 0.6)
    (tmp_path / "L2-9" / "B_s0").mkdir()   # treino sem meta.json: não concluído
    runs = read_runs(tmp_path)
    assert len(runs) == 4
    out = summarize(runs)
    assert out.loc["L1", "epocas_no_fim"] == 2 and bool(out.loc["L1", "alarme_subtreino"])
    assert not bool(out.loc["L2-9", "alarme_subtreino"])
    assert out.loc["L1", "ma_media"] == 0.2
