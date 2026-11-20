"""Cache dos recortes do leitor (K-b; protocolo do leitor, seção 1.4).

Cada campo da lista de recortes do desenvolvimento é recortado uma vez, com a
margem da classe (meter_reader.crops), em resolução original, sem realce nem
redimensionamento, e gravado num único .npz na pasta de trabalho. O realce, a
tela fixa e a aumentação continuam no código (E33). A foto é aberta sem a
orientação EXIF, como no detector (E92). Nada é exibido além de contagens.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from meter_reader.crops import crop_field
from training.build_manifest import file_sha256

CACHE_DIR = Path("experimentos") / "leitor"
CACHE_NAME = "cache_recortes_leitor.npz"
_LIST_KEY = "_lista_sha256"


def crop_key(pseudonym: str, field_class: str) -> str:
    """Chave de um recorte no cache: pseudônimo e classe."""
    return f"{pseudonym}__{field_class}"


def build_cache(crops: pd.DataFrame, paths: dict[str, Path], out_path: Path,
                list_sha256: str, every: int = 1000) -> int:
    """Recorta todos os campos da lista, abrindo cada foto uma vez, e grava o .npz.

    `paths`: pseudônimo -> caminho da foto. A gravação é atômica.
    """
    arrays: dict[str, np.ndarray] = {_LIST_KEY: np.array(list_sha256)}
    groups = crops.groupby("pseudonimo", sort=True)
    for i, (pseudonym, rows) in enumerate(groups, 1):
        with Image.open(paths[pseudonym]) as image:
            image = image.convert("RGB")
            for r in rows.to_dict("records"):
                box = (r["x0"], r["y0"], r["x1"], r["y1"])
                crop = crop_field(image, box, r["classe"])
                arrays[crop_key(pseudonym, r["classe"])] = np.asarray(crop, dtype=np.uint8)
        if every and i % every == 0:
            print(f"  fotos recortadas: {i} de {groups.ngroups}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    with tmp.open("wb") as file:
        np.savez(file, **arrays)
    os.replace(tmp, out_path)
    return len(arrays) - 1


def load_cache(path: Path, list_sha256: str):
    """Abre o cache (leitura preguiçosa) e confere a lista de recortes que o gerou."""
    cache = np.load(path)
    if str(cache[_LIST_KEY]) != list_sha256:
        cache.close()
        raise ValueError("O cache não foi gerado pela lista de recortes vigente.")
    return cache


def main() -> None:
    from meter_reader.config import get_work_dir
    from training.group_dataset import LABEL_VERSIONS, load_group_photos
    from training.reader_labels import CROPS_NAME, CROPS_SHA256

    work_dir = get_work_dir()
    if file_sha256(work_dir / CROPS_NAME) != CROPS_SHA256:
        raise SystemExit("A lista de recortes não é a gerada (SHA-256 diferente).")
    crops = pd.read_csv(work_dir / CROPS_NAME, float_precision="round_trip")
    paths = {p.pseudonym: p.path for p in load_group_photos(LABEL_VERSIONS["r4"])}
    out_path = work_dir / CACHE_DIR / CACHE_NAME
    n = build_cache(crops, paths, out_path, CROPS_SHA256)
    print(f"Recortes no cache: {n} de {len(crops)}")
    print(f"Tamanho: {out_path.stat().st_size / 1e6:.0f} MB")
    print("SHA-256 do cache:", file_sha256(out_path))


if __name__ == "__main__":
    main()
