import numpy as np
import pandas as pd
import pytest
from PIL import Image

from training.reader_cache import build_cache, crop_key, load_cache


def _setup(tmp_path):
    photo = tmp_path / "foto.png"
    Image.new("RGB", (200, 100), (10, 20, 30)).save(photo)
    crops = pd.DataFrame({
        "pseudonimo": ["F-a", "F-a"],
        "classe": ["display_digital", "numero_serie"],
        "x0": [50.0, 20.0], "y0": [40.0, 10.0], "x1": [150.0, 60.0], "y1": [60.0, 20.0],
    })
    return crops, {"F-a": photo}


def test_cache_guarda_os_recortes_com_a_margem(tmp_path):
    crops, paths = _setup(tmp_path)
    out = tmp_path / "trabalho" / "cache.npz"
    assert build_cache(crops, paths, out, "abc", every=0) == 2
    cache = load_cache(out, "abc")
    digital = cache[crop_key("F-a", "display_digital")]
    # margem do digital: 0,22 x 100 = 22 px por lado; 0,27 x 20 = 5,4 px por lado,
    # arredondada para fora: de 34 a 66 na altura (32 px) e de 28 a 172 na largura
    assert digital.shape == (32, 144, 3)
    assert digital.dtype == np.uint8
    assert tuple(digital[0, 0]) == (10, 20, 30)
    cache.close()
    assert not out.with_name(out.name + ".tmp").exists()


def test_cache_recusa_outra_lista(tmp_path):
    crops, paths = _setup(tmp_path)
    out = tmp_path / "cache.npz"
    build_cache(crops, paths, out, "abc", every=0)
    with pytest.raises(ValueError):
        load_cache(out, "outra")
