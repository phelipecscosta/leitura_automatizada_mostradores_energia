import pandas as pd
import torch

from meter_reader.reader_text import TEXT_ALPHABET, encode
from training.reader_sanity import exact_match, overfit, sanity_items


def test_lote_fixo_por_classe():
    rows = [{"pseudonimo": f"F-{c}{i}", "classe": c, "recorte": "inteiro", "usa_texto": True}
            for c in ("display_digital", "display_ciclometrico", "numero_serie")
            for i in range(20)]
    picked = sanity_items(pd.DataFrame(rows))
    assert picked["classe"].value_counts().to_dict() == {
        "display_digital": 11, "display_ciclometrico": 11, "numero_serie": 10}


def test_acerto_exato():
    logits = torch.full((2, 4, len(TEXT_ALPHABET) + 1), -9.0)
    for t, k in enumerate([2, 0, 3, 0]):       # "12"
        logits[0, t, k] = 9.0
    logits[1, :, 0] = 9.0                      # só brancos
    batch = {"texto": torch.tensor(encode("12", TEXT_ALPHABET) + encode("5", TEXT_ALPHABET)),
             "texto_comprimento": torch.tensor([2, 1]),
             "usa_texto": torch.tensor([True, True])}
    assert exact_match(logits, batch, "texto", TEXT_ALPHABET) == 0.5


def test_sobreajuste_roda_em_cpu():
    batch = {"imagem": torch.randn(2, 3, 48, 192),
             "texto": torch.tensor(encode("12", TEXT_ALPHABET) + encode("3", TEXT_ALPHABET)),
             "texto_comprimento": torch.tensor([2, 1]),
             "funcao": torch.tensor([], dtype=torch.long),
             "funcao_comprimento": torch.tensor([0, 0]),
             "usa_texto": torch.tensor([True, True]),
             "usa_funcao": torch.tensor([False, False]),
             "usa_duas": torch.tensor([False, False]), "duas": torch.zeros(2)}
    r = overfit("L1", batch, "cpu", steps=2, pretrained=False)
    assert set(r) == {"inicial", "final", "acerto_texto", "acerto_funcao"}


