import pandas as pd
import torch
from PIL import Image

from training.reader_function_x4 import (X4_HEIGHT, X4_WIDTH, _runs, build_model, direct_errors,
                                         left_band, x4_tensor)


def test_faixa_esquerda_e_entrada():
    crop = Image.new("RGB", (200, 60), (90, 90, 90))
    assert left_band(crop).size == (80, 60)
    assert tuple(x4_tensor(crop).shape) == (3, X4_HEIGHT, X4_WIDTH)


def test_saida_do_modelo():
    model = build_model(pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.zeros(2, 3, X4_HEIGHT, X4_WIDTH))
    assert tuple(out.shape) == (2, 7)


def test_nove_treinos():
    runs = _runs()
    assert len(runs) == 9 and len(set(runs)) == 9


def test_erros_diretos_por_origem_e_codigo():
    pop = pd.DataFrame({"alta": [True, True, True, False], "nc": [True, True, False, True],
                        "nao_julgado": [False, True, False, False], "s": [0.9, 0.9, 0.9, 0.9],
                        "origem": ["humano_f2", "haiku_csv", "humano_f2", "humano_f2"],
                        "funcao": ["103", "24", "03", "55"]})
    out = direct_errors(pop, "s", 0.5)
    assert out.to_dict() == {("humano_f2", "103"): 1}
