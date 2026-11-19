"""Piloto do professor da rotulagem do leitor (T6a.3).

Sorteia recortes do desenvolvimento a partir das caixas que o leitor recebe
(select_fields no corte adotado, predições fora da dobra da semente 0 do S1_coco).
Composição fixa por classe (D5), sorteio simples com sementes em texto, e uma
subamostra repetida para medir a estabilidade. A amostra fica na pasta de trabalho,
só com pseudônimos e coordenadas.
"""

from __future__ import annotations

import pandas as pd

from meter_reader.fields import select_fields
from training.box_sample import shuffled
from training.build_manifest import file_sha256

PILOT_COUNTS = {"display_digital": 80, "display_ciclometrico": 70, "numero_serie": 50}
STABILITY_N = 30  # recortes enviados duas vezes (D5)
PILOT_NAME = "amostra_piloto_professor.csv"
PILOT_COLUMNS = ("ordem", "pseudonimo", "classe", "x0", "y0", "x1", "y1", "estabilidade")


def candidates(photos, preds: list[dict], cut: float) -> pd.DataFrame:
    """Uma linha por campo que o leitor receberia: consumo (digital ou ciclométrico)
    e número de série, com a caixa escolhida, a 0,1 px."""
    rows = []
    for photo, pred in zip(photos, preds):
        sel = select_fields(pred["boxes"], pred["scores"], pred["labels"], cut)
        chosen = []
        if sel.consumption is not None:
            cls = "display_digital" if sel.family == "digital" else "display_ciclometrico"
            chosen.append((cls, sel.consumption))
        if sel.serial is not None:
            chosen.append(("numero_serie", sel.serial))
        for cls, box in chosen:
            x0, y0, x1, y1 = (round(float(v), 1) for v in box)
            rows.append({"pseudonimo": photo.pseudonym, "classe": cls,
                         "x0": x0, "y0": y0, "x1": x1, "y1": y1})
    return pd.DataFrame(rows, columns=["pseudonimo", "classe", "x0", "y0", "x1", "y1"])


def draw(cands: pd.DataFrame, counts: dict = PILOT_COUNTS,
         stability_n: int = STABILITY_N) -> pd.DataFrame:
    """Sorteio por classe, marca de estabilidade e ordem de envio, todos fixos."""
    parts = []
    for cls, n in counts.items():
        pool = cands[cands["classe"] == cls].set_index("pseudonimo")
        if len(pool) < n:
            raise ValueError(f"{cls}: só {len(pool)} candidatos para {n}.")
        codes = shuffled(pool.index, f"piloto:{cls}")[:n]
        parts.append(pool.loc[codes].reset_index())
    sample = pd.concat(parts, ignore_index=True)
    keys = sample["pseudonimo"] + "|" + sample["classe"]
    repeated = set(shuffled(keys, "piloto:estabilidade")[:stability_n])
    sample["estabilidade"] = keys.isin(repeated)
    order = {k: i + 1 for i, k in enumerate(shuffled(keys, "piloto:ordem"))}
    sample["ordem"] = keys.map(order)
    return sample.sort_values("ordem")[list(PILOT_COLUMNS)].reset_index(drop=True)


def save_or_check(sample: pd.DataFrame, work_dir) -> str:
    """Grava a amostra; se já existir, confere que é idêntica ao sorteio."""
    path = work_dir / PILOT_NAME
    if path.exists():
        saved = pd.read_csv(path, float_precision="round_trip")
        if not saved.equals(sample.astype(saved.dtypes.to_dict())):
            raise ValueError("A amostra gravada difere do sorteio; nada foi alterado.")
        print("Amostra já gravada; conferida: idêntica ao sorteio.")
    else:
        sample.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
        print("Amostra gravada.")
    return file_sha256(path)


def main() -> None:
    from meter_reader.config import get_work_dir
    from training.crop_margin import dev_predictions

    photos, preds, cut = dev_predictions()
    cands = candidates(photos, preds, cut)
    print("Candidatos por classe:", cands["classe"].value_counts().to_dict())
    sample = draw(cands)
    print("Sorteados por classe:", sample["classe"].value_counts().to_dict(),
          "; estabilidade:", int(sample["estabilidade"].sum()))
    print("SHA-256 da amostra:", save_or_check(sample, get_work_dir()))


if __name__ == "__main__":
    main()
