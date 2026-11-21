import math

from training.reader_train import lr_factor, run_dir, text_matches


def test_cronograma():
    assert lr_factor(0, 10, 100) == 0.1
    assert lr_factor(9, 10, 100) == 1.0
    assert math.isclose(lr_factor(10, 10, 100), 1.0)
    assert math.isclose(lr_factor(55, 10, 100), 0.5)
    assert lr_factor(100, 10, 100) < 1e-9


def test_acerto_numerico_e_serie():
    assert text_matches("0012345", "12345", "display_digital")
    assert text_matches("12345", "0012345", "display_ciclometrico")
    assert not text_matches("12346", "12345", "display_digital")
    assert not text_matches("", "0", "display_digital")
    assert not text_matches("0AB12", "AB12", "numero_serie")


def test_pasta_do_treino(tmp_path):
    assert run_dir(tmp_path, "L1", "2026-05-20", 2).parts[-2:] == ("L1", "2026-05-20_s2")



def test_ajuste_fino_sem_pesos_de_partida(tmp_path, monkeypatch):
    import pandas as pd
    import pytest

    import meter_reader.config
    from training.reader_train import run_grid

    monkeypatch.setattr(meter_reader.config, "get_work_dir", lambda: tmp_path)
    items = pd.DataFrame({"lote": ["A"]})
    spec = {"x+familia": {"candidato": "L2-12", "inicio": "x", "taxa": 5e-5, "max_epocas": 1}}
    with pytest.raises(SystemExit, match="Pesos de partida ausentes"):
        run_grid(spec, items, seeds=(0,))
