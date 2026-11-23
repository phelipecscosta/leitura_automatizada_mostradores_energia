import numpy as np
import torch

from training.reader_heads_train import (FUNCTION_CLASSES, Heads, class_weights,
                                         function_class, macro_recall, trunk_features)


def test_classe_pelo_prefixo():
    codes = ["", "03", "0312", "003", "103", "10312", "24", "55", "07"]
    assert [FUNCTION_CLASSES[function_class(c)] for c in codes] == [
        "vazio", "03", "03", "003", "103", "103", "24", "55", "outro"]


def test_pesos_das_classes():
    classes = np.array([1] * 8 + [3] * 2)
    w = class_weights(classes, n_classes=7)
    assert w[0] == 0 and float(w[3]) > float(w[1])            # 103 rara pesa mais
    assert abs(float(w[classes].mean()) - 1.0) < 1e-6          # peso médio por exemplo 1
    assert abs(float(w[1]) * 8 - float(w[3]) * 2) < 1e-6       # classes pesam o mesmo


def test_cabecas_sobre_o_tronco():
    from meter_reader.reader import Reader
    model = Reader("L2-12", pretrained=False).eval()
    with torch.no_grad():
        f = trunk_features(model, torch.zeros(2, 3, 48, 192))
    func, two = Heads(f.shape[-1])(f)
    assert func.shape == (2, len(FUNCTION_CLASSES)) and two.shape == (2,)


def test_revocacao_media():
    assert macro_recall(np.array([1, 1, 3, 3]), np.array([1, 1, 1, 3])) == 0.75
