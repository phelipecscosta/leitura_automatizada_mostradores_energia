"""Rotulagem completa do desenvolvimento pelo professor (T6a.4).

Recortes: os campos que o leitor receberia (select_fields no corte adotado) em todas
as fotos pareadas do desenvolvimento que passam pela triagem (população da E75).
Caixas (B-b): fora da dobra (semente 0 do S1_coco) nas fotos anotadas; detector final
do produto, em CPU, nas demais. Tudo na pasta de trabalho, só com pseudônimos; as
saídas impressas são contagens.

Uso, a partir da raiz do repositório do produto:
    python -m training.reader_labels recortes
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pandas as pd

from training.build_manifest import file_sha256
from training.reader_pilot import candidates

CROPS_NAME = "recortes_leitor_desenvolvimento.csv"
CROP_COLUMNS = ("pseudonimo", "classe", "x0", "y0", "x1", "y1", "origem_caixa")


def final_predictions(paths, detector, every: int = 500) -> list[dict]:
    """Caixas do detector final, foto a foto, no formato das predições do treino."""
    preds = []
    for i, path in enumerate(paths, 1):
        r = detector.predict(path)
        preds.append({"boxes": r.boxes, "scores": r.scores, "labels": r.labels})
        if i % every == 0:
            print(f"  detector final: {i} de {len(paths)} fotos")
    return preds


def merge_crops(oof: pd.DataFrame, final: pd.DataFrame) -> pd.DataFrame:
    """Junta os recortes das duas origens; uma foto nunca tem caixas das duas."""
    if set(oof["pseudonimo"]) & set(final["pseudonimo"]):
        raise ValueError("Foto com caixas das duas origens.")
    out = pd.concat([oof.assign(origem_caixa="fora_da_dobra"),
                     final.assign(origem_caixa="final")], ignore_index=True)
    return out.sort_values(["pseudonimo", "classe"])[list(CROP_COLUMNS)].reset_index(drop=True)


def recortes() -> None:
    from meter_reader.config import get_detector_artifact, get_work_dir
    from meter_reader.detector import PRODUCT_DETECTOR_SHA256, load_detector
    from training.box_sample import population
    from training.crop_margin import dev_predictions
    from training.group_dataset import LABEL_VERSIONS, load_group_photos

    work_dir = get_work_dir()
    pop = population()
    codes = set(pop.loc[(pop["conjunto"] == "desenvolvimento") & pop["passa"], "pseudonimo"])

    photos, preds, cut = dev_predictions()
    keep = [i for i, p in enumerate(photos) if p.pseudonym in codes]
    oof = candidates([photos[i] for i in keep], [preds[i] for i in keep], cut)
    annotated = {photos[i].pseudonym for i in keep}

    detector = load_detector(get_detector_artifact(), PRODUCT_DETECTOR_SHA256)
    if abs(detector.field_cut - cut) > 1e-9:
        raise SystemExit("O corte dos campos do artefato difere do adotado.")
    paths = {p.pseudonym: p.path for p in load_group_photos(LABEL_VERSIONS["r4"])}
    rest = sorted(codes - annotated)
    print(f"População: {len(codes)} fotos; caixas fora da dobra em {len(annotated)}; "
          f"detector final em {len(rest)}")
    final = candidates([SimpleNamespace(pseudonym=c) for c in rest],
                       final_predictions([paths[c] for c in rest], detector), cut)

    crops = merge_crops(oof, final)
    path = work_dir / CROPS_NAME
    crops.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    print("Recortes por classe e origem:\n"
          + pd.crosstab(crops["classe"], crops["origem_caixa"]).to_string())
    print("SHA-256 da lista de recortes:", file_sha256(path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Rotulagem completa do leitor (T6a.4).")
    sub = parser.add_subparsers(dest="comando", required=True)
    sub.add_parser("recortes", help="gera a lista de recortes do desenvolvimento")
    args = parser.parse_args()
    if args.comando == "recortes":
        recortes()


if __name__ == "__main__":
    main()
