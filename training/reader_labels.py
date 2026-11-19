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
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from training.build_manifest import file_sha256
from training.reader_pilot import candidates

CROPS_NAME = "recortes_leitor_desenvolvimento.csv"
CROP_COLUMNS = ("pseudonimo", "classe", "x0", "y0", "x1", "y1", "origem_caixa")
CROPS_SHA256 = "9509481ec41f6637c0c44353460fc5902c647ded24a23ce174fea636d8a781f1"
LABEL_DIR = Path("experimentos") / "leitor" / "rotulagem"
CLASS_CODES = {"display_digital": "dig", "display_ciclometrico": "cic", "numero_serie": "ser"}
MAX_PART_BYTES = 180_000_000   # folga abaixo do limite de tamanho por lote da Batch API
MAX_PART_REQUESTS = 10_000


def custom_id(pseudonym: str, field_class: str) -> str:
    """Identificador do pedido no lote: só o pseudônimo e o código da classe."""
    return f"{pseudonym}__{CLASS_CODES[field_class]}"


def parse_custom_id(cid: str) -> tuple[str, str]:
    code, short = cid.rsplit("__", 1)
    return code, {v: k for k, v in CLASS_CODES.items()}[short]


def state_path(model: str) -> Path:
    """Estado das partes enviadas, relativo à pasta de trabalho."""
    return LABEL_DIR / f"lotes_{model}.json"


def load_state(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"partes": []}


def save_state(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)  # gravação atômica


def split_parts(sizes: list[int], max_bytes: int = MAX_PART_BYTES,
                max_requests: int = MAX_PART_REQUESTS) -> list[int]:
    """Índices em que cada nova parte começa, para que nenhuma passe dos limites."""
    starts, total, count = [0], 0, 0
    for i, n in enumerate(sizes):
        if count and (total + n > max_bytes or count >= max_requests):
            starts.append(i)
            total, count = 0, 0
        total += n
        count += 1
    return starts


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

def enviar(model: str) -> None:
    """Envia em partes, pela Batch API, os recortes ainda não enviados."""
    from anthropic import Anthropic
    from dotenv import load_dotenv
    from PIL import Image

    from meter_reader.config import get_work_dir
    from training.group_dataset import LABEL_VERSIONS, load_group_photos
    from training.teacher import build_request, prepare_crop

    load_dotenv()
    work_dir = get_work_dir()
    if file_sha256(work_dir / CROPS_NAME) != CROPS_SHA256:
        raise SystemExit("A lista de recortes não é a gerada (SHA-256 diferente).")
    crops = pd.read_csv(work_dir / CROPS_NAME, float_precision="round_trip")
    sp = work_dir / state_path(model)
    sp.parent.mkdir(parents=True, exist_ok=True)
    state = load_state(sp)
    sent = {cid for part in state["partes"] for cid in part["ids"]}
    pending = [r for r in crops.to_dict("records")
               if custom_id(r["pseudonimo"], r["classe"]) not in sent]
    print(f"Recortes: {len(crops)}; já enviados: {len(sent)}; a enviar: {len(pending)}")
    if not pending:
        return
    paths = {p.pseudonym: p.path for p in load_group_photos(LABEL_VERSIONS["r4"])}
    client = Anthropic()

    def submit(requests: list[dict], size: int) -> None:
        batch = client.messages.batches.create(requests=requests)
        state["partes"].append({"id": batch.id, "ids": [q["custom_id"] for q in requests],
                                "coletada": False})
        save_state(sp, state)
        print(f"  parte {len(state['partes'])}: {len(requests)} pedidos, "
              f"{size / 1e6:.0f} MB, enviada")

    requests, size = [], 0
    for i, r in enumerate(pending, 1):
        box = (r["x0"], r["y0"], r["x1"], r["y1"])
        with Image.open(paths[r["pseudonimo"]]) as image:
            png = prepare_crop(image, box, r["classe"])
        req = {"custom_id": custom_id(r["pseudonimo"], r["classe"]),
               "params": build_request(png, r["classe"], model)}
        n = len(json.dumps(req))
        if requests and (size + n > MAX_PART_BYTES or len(requests) >= MAX_PART_REQUESTS):
            submit(requests, size)
            requests, size = [], 0
        requests.append(req)
        size += n
        if i % 1000 == 0:
            print(f"  preparados {i} de {len(pending)}")
    if requests:
        submit(requests, size)


def main() -> None:
    from training.teacher import MODEL

    parser = argparse.ArgumentParser(description="Rotulagem completa do leitor (T6a.4).")
    sub = parser.add_subparsers(dest="comando", required=True)
    sub.add_parser("recortes", help="gera a lista de recortes do desenvolvimento")
    env = sub.add_parser("enviar", help="envia os recortes pendentes pela Batch API")
    env.add_argument("--modelo", default=MODEL)
    args = parser.parse_args()
    if args.comando == "recortes":
        recortes()
    else:
        enviar(args.modelo)

if __name__ == "__main__":
    main()
