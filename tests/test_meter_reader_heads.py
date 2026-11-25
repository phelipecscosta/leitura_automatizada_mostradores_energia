import re
from pathlib import Path

import torch

from meter_reader.reader_heads import CONSUMPTION_CLASSES, FUNCTION_CLASSES, Heads


def test_chaves_dos_pesos_iguais_as_do_treino():
    # As cabeças gravadas no artefato (T6c.4) carregam por estes nomes
    names = {"f_project.0", "f_out", "two.0", "two.2"}
    lstm = {f"f_rnn.{w}_l0{s}" for w in ("weight_ih", "weight_hh", "bias_ih", "bias_hh")
            for s in ("", "_reverse")}
    expected = {f"{n}.{p}" for n in names for p in ("weight", "bias")} | lstm
    assert set(Heads(64).state_dict()) == expected


def test_saidas_e_consumo():
    func, two = Heads(64)(torch.zeros(2, 5, 64))
    assert func.shape == (2, len(FUNCTION_CLASSES)) and two.shape == (2,)
    assert CONSUMPTION_CLASSES == FUNCTION_CLASSES[:3]


def test_treino_usa_as_cabecas_do_produto():
    from meter_reader import reader_heads
    from training import reader_heads_train
    assert reader_heads_train.Heads is reader_heads.Heads
    assert reader_heads_train.trunk_features is reader_heads.trunk_features
    assert reader_heads_train.FUNCTION_CLASSES is reader_heads.FUNCTION_CLASSES


def test_produto_nunca_importa_training():
    src = Path(__file__).resolve().parents[1] / "src" / "meter_reader"
    pattern = re.compile(r"^\s*(from|import)\s+training\b", re.MULTILINE)
    assert not [p.name for p in src.glob("*.py") if pattern.search(p.read_text(encoding="utf-8"))]
