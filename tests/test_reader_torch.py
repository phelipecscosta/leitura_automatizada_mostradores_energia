import random

import numpy as np
import pandas as pd
from PIL import Image

from training.reader_augment import augment
from training.reader_torch import BalancedBatchSampler, ReaderDataset, collate


def _items():
    rows = []
    for k, (cls, n) in enumerate([("display_digital", 10), ("display_ciclometrico", 3),
                                  ("numero_serie", 10)]):
        for j in range(n):
            rows.append({"pseudonimo": f"F-{k}{j:02d}", "classe": cls, "lote": "L",
                         "recorte": "inteiro", "texto": "0123" if cls != "numero_serie"
                         else "AB12", "funcao": "03" if cls == "display_digital" else "",
                         "duas_leituras": False, "usa_texto": True,
                         "usa_funcao": cls == "display_digital",
                         "usa_duas": cls == "display_digital", "origem": "t"})
    for j in range(5):
        rows.append({"pseudonimo": f"F-9{j:02d}", "classe": "display_digital", "lote": "L",
                     "recorte": "inteiro", "texto": "", "funcao": "", "duas_leituras": True,
                     "usa_texto": False, "usa_funcao": False, "usa_duas": True,
                     "origem": "marca_professor"})
    return pd.DataFrame(rows)


def test_augment_mantem_tamanho_e_modo():
    image = Image.new("RGB", (120, 40), (200, 200, 200))
    out = augment(image, random.Random(0))
    assert out.size == (120, 40) and out.mode == "RGB"


def test_amostragem_equilibrada_com_fluxo_da_cabeca():
    items = _items()
    sampler = BalancedBatchSampler(items, seed=0, text_batch=6, head_batch=2)
    assert len(sampler) == 23 // 6
    batch = next(iter(sampler))
    chosen = items.iloc[batch]
    text = chosen[chosen["usa_texto"]]
    assert text["classe"].value_counts().tolist() == [2, 2, 2]
    assert (~chosen["usa_texto"]).sum() == 2


def test_conjunto_e_agrupamento(tmp_path):
    from training.reader_cache import crop_key

    items = _items().iloc[[0, 10, 13]]
    arrays = {"_lista_sha256": np.array("x")}
    for r in items.to_dict("records"):
        arrays[crop_key(r["pseudonimo"], r["classe"])] = np.full((40, 120, 3), 128, np.uint8)
    path = tmp_path / "cache.npz"
    np.savez(path, **arrays)
    data = ReaderDataset(items, path, "x", train=True)
    batch = collate([data[i] for i in range(len(data))])
    assert batch["imagem"].shape == (3, 3, 48, 384)
    assert batch["texto_comprimento"].tolist() == [4, 4, 4]
    assert batch["funcao_comprimento"].tolist() == [2, 0, 0]
    assert batch["usa_funcao"].tolist() == [True, False, False]
    assert batch["classe"].tolist() == [0, 1, 2]
