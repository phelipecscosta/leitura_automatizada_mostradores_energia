"""Limiar de confiança do número de série (revisão R20 do protocolo do leitor; ET6c,
T6c.7, S1-a).

Recortes de número de série dos 3 lotes de desenvolvimento, lidos fora da dobra pelos
modelos de 2 lotes (comitê das 3 sementes, TF32 desligado). Legível: professor
"aceito" ou "discorda"; ilegível: "ilegivel" ou "parcial"; "erro_formato" fica fora.
Limiar pela regra da R10 (V = 10%, pelo menos 30 abaixo). Relata também o efeito nos
recortes cujo número lido difere do esperado, os únicos em que o limiar atua.
Grava `limiar_numero_leitor.json` na pasta de trabalho, sem sobrescrever. Só números.

Uso, a partir da raiz do repositório do produto:
    python -m training.reader_serial_threshold
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

OUT_NAME = Path("experimentos") / "leitor" / "limiar_numero_leitor.json"
LEGIBLE, ILLEGIBLE = ("aceito", "discorda"), ("ilegivel", "parcial")


def effect(log_conf, mismatch, legible, log_t: float) -> dict:
    """Nos recortes com número diferente do esperado: quantos ficam abaixo do limiar
    (passam a "não localizado") e, entre eles, quantos são legíveis."""
    c, m, y = (np.asarray(log_conf, float), np.asarray(mismatch, bool),
               np.asarray(legible, bool))
    below = m & (c < log_t)
    return {"divergentes": int(m.sum()), "viram_nao_localizado": int(below.sum()),
            "legiveis_entre_eles": int((below & y).sum()),
            "ilegiveis_que_seguem_divergentes": int((m & ~below & ~y).sum())}


def main(device: str = "cuda") -> None:
    import pandas as pd
    from PIL import Image

    from meter_reader.config import get_work_dir
    from training.group_dataset import LABEL_VERSIONS, load_group_photos
    from training.reader_bands import lower_threshold
    from training.reader_cache import CACHE_DIR, CACHE_NAME, crop_key, load_cache
    from training.reader_dataset import SERIAL
    from training.reader_labels import CROPS_NAME, CROPS_SHA256, labels_name
    from training.reader_pilot import csv_rows
    from training.reader_product_parity import set_tf32
    from training.reader_tf32_check import _fold_models, _read
    from training.teacher import MODEL

    work_dir = get_work_dir()
    out_path = work_dir / OUT_NAME
    if out_path.exists():
        raise SystemExit("O limiar do número de série já foi gravado e não é refeito.")
    set_tf32(False)
    photos = [p for p in load_group_photos(LABEL_VERSIONS["r4"])]
    lot = {p.pseudonym: p.lot for p in photos}
    expected = {c: r.expected_meter_number for c, r in csv_rows(photos).items()}
    raw = pd.read_csv(work_dir / labels_name(MODEL), dtype=str, keep_default_na=False)
    status = dict(raw.loc[raw["classe"] == SERIAL, ["pseudonimo", "status"]]
                  .drop_duplicates("pseudonimo").itertuples(index=False))
    crops = pd.read_csv(work_dir / CROPS_NAME, dtype=str)
    codes = [c for c in crops.loc[crops["classe"] == SERIAL, "pseudonimo"].unique()
             if status.get(c) in LEGIBLE + ILLEGIBLE and c in expected and c in lot]
    cache = load_cache(work_dir / CACHE_DIR / CACHE_NAME, CROPS_SHA256)

    confs, legible, mismatch = [], [], []
    for lt in sorted({lot[c] for c in codes}):
        part = [c for c in codes if lot[c] == lt]
        images = [Image.fromarray(cache[crop_key(c, SERIAL)]) for c in part]
        reads = _read(_fold_models(work_dir, lt, device), images, math.inf, False, device)
        for c, r in zip(part, reads):
            confs.append(r["log_conf"])
            legible.append(status[c] in LEGIBLE)
            mismatch.append(r["texto"].upper() != expected[c].upper())
        print(f"Lote {lt}: {len(part)} recortes de número de série", flush=True)

    low = lower_threshold(np.array(confs), np.array(legible))
    if low["log_limiar"] is None:
        raise SystemExit("Nenhum limiar cumpre a regra da R10.")
    result = {**low, "limiar": math.exp(low["log_limiar"]), "recortes": len(codes),
              "legiveis": int(np.sum(legible)),
              "efeito": effect(confs, mismatch, legible, low["log_limiar"])}
    print("Limiar do número de série (R20):", json.dumps(result))
    out_path.write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
