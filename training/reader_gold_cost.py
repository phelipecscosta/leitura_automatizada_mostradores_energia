"""Custo da população-ouro da função (medição antes de qualquer decisão; ET6c).

Lê, fora da dobra e pelo fluxo do produto (comitê das 3 sementes de cada dobra;
faixa de cima quando a D1 dispara, com o limiar do F1 escolhido nos outros dois
lotes), todos os recortes digitais do desenvolvimento. Conta os aprováveis (leitura
livre igual ao CSV) pela origem do rótulo de função e estima o tempo de julgamento
dos que ainda não têm julgamento humano, pela mediana do tempo do Q-a. Não usa a
cabeça da função. Grava a tabela por foto na pasta de trabalho. Só números.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HUMAN_ORIGINS = ("humano_f2", "humano_r12")
TABLE_NAME = Path("experimentos") / "leitor" / "digitais_aprovaveis.csv"


def label_source(origins: set[str], judged: bool) -> str:
    """Origem do rótulo de função de uma foto, da mais forte para a mais fraca."""
    if judged or origins & set(HUMAN_ORIGINS):
        return "humano"
    if "haiku_csv" in origins:
        return "aceito_professor"
    if "marca_professor" in origins:
        return "marca_professor"
    return "sem_rotulo" if not origins else "outro"


def main(device: str = "cuda") -> None:
    import torch
    from PIL import Image

    from meter_reader.config import get_work_dir
    from meter_reader.reader import Reader
    from meter_reader.reader_input import upper_band
    from training.group_dataset import LABEL_VERSIONS, load_group_photos
    from training.reader_cache import CACHE_DIR, CACHE_NAME, crop_key, load_cache
    from training.reader_dataset import DIGITAL
    from training.reader_function_r12 import R12_HEADS, extended_items
    from training.reader_function_rule import OUT_DIR, _committee_read, _outputs
    from training.reader_heads_eval import f1_threshold
    from training.reader_heads_train import CONFIG, HEADS_DIR, Heads, trunk_features
    from training.reader_labels import CROPS_NAME, CROPS_SHA256
    from training.reader_pilot import csv_rows
    from training.reader_train import RUNS_DIR, WEIGHTS_NAME

    work_dir = get_work_dir()
    photos = load_group_photos(LABEL_VERSIONS["r4"])
    lot_of = {p.pseudonym: p.lot for p in photos}
    expected = {c: r.reading for c, r in csv_rows(photos).items()}
    crops = pd.read_csv(work_dir / CROPS_NAME, float_precision="round_trip")
    codes_all = sorted(set(crops.loc[crops["classe"] == DIGITAL, "pseudonimo"]) & set(expected))
    items = extended_items()
    dig = items[items["classe"] == DIGITAL]
    origins = dig.groupby("pseudonimo")["origem"].apply(set).to_dict()
    judged_qa = set(pd.read_csv(work_dir / OUT_DIR / "rotulagem_r15_aceitos.csv", dtype=str,
                                keep_default_na=False)["pseudonimo"])
    two = dig[dig["usa_duas"]].drop_duplicates("pseudonimo").set_index("pseudonimo")
    cache = load_cache(work_dir / CACHE_DIR / CACHE_NAME, CROPS_SHA256)

    rows, outputs = [], {}
    for lot in sorted({lot_of[c] for c in codes_all}):
        codes = [c for c in codes_all if lot_of[c] == lot]
        whole = [Image.fromarray(cache[crop_key(c, DIGITAL)]) for c in codes]
        upper = [upper_band(c) for c in whole]
        out = []
        for seed in (0, 1, 2):
            run = f"{lot}_s{seed}"
            model = Reader("L2-12", pretrained=False).to(device)
            model.load_state_dict(torch.load(work_dir / RUNS_DIR / CONFIG / run / WEIGHTS_NAME,
                                             map_location=device, weights_only=True))
            model.eval()
            with torch.no_grad():
                dim = trunk_features(model, torch.zeros(1, 3, 48, 192, device=device)).shape[-1]
            heads = Heads(dim).to(device)
            heads.load_state_dict(torch.load(work_dir / HEADS_DIR / R12_HEADS / run
                                             / "pesos_cabecas.pt", map_location=device,
                                             weights_only=True))
            heads.eval()
            out.append({"inteiro": _outputs(model, heads, whole, device),
                        "cima": _outputs(model, heads, upper, device)})
        outputs[lot] = out
        logit = np.mean([o["inteiro"]["duas"] for o in out], axis=0)
        rows += [{"pseudonimo": c, "lote": lot, "indice": i, "logit_duas": float(logit[i])}
                 for i, c in enumerate(codes)]
        print(f"Lote {lot}: {len(codes)} recortes digitais", flush=True)

    df = pd.DataFrame(rows)
    # D1: limiar do F1 nos outros dois lotes, com as marcas de duas leituras
    marks = two["duas_leituras"].astype(bool)
    logit_of = dict(zip(df["pseudonimo"], df["logit_duas"]))
    d1 = {}
    for lot in sorted(df["lote"].unique()):
        fit = [c for c in marks.index if c in logit_of and lot_of[c] != lot]
        d1[lot] = f1_threshold(marks[fit].to_numpy(), np.array([logit_of[c] for c in fit]))
    high, fired = [], []
    for r in df.to_dict("records"):
        up = r["logit_duas"] >= d1[r["lote"]]
        key = "cima" if up else "inteiro"
        lps = [o[key]["texto"][r["indice"]] for o in outputs[r["lote"]]]
        text = _committee_read(lps)
        high.append(text.isdigit() and int(text) == int(expected[r["pseudonimo"]]))
        fired.append(up)
    df = df.drop(columns=["indice"])
    df["d1"], df["alta"] = fired, high
    df["rotulo"] = [label_source(origins.get(c, set()), c in judged_qa) for c in df["pseudonimo"]]
    df.to_csv(work_dir / TABLE_NAME, index=False)

    pd.set_option("display.width", 160)
    print(f"\nLimiares da D1 por lote: { {k: round(v, 3) for k, v in d1.items()} }")
    print("\nDigitais do desenvolvimento por origem do rótulo e aprovável (leitura = CSV):")
    print(pd.crosstab(df["rotulo"], df["alta"], margins=True, margins_name="total").to_string())
    todo = df["alta"] & (df["rotulo"] != "humano")
    secs = pd.read_csv(work_dir / OUT_DIR / "rotulagem_r15_aceitos.csv")["segundos"].astype(float)
    med = float(secs.median())
    print(f"\nA julgar (aprováveis sem julgamento humano): {int(todo.sum())}; por lote: "
          f"{df.loc[todo, 'lote'].value_counts().sort_index().to_dict()}")
    print(f"Tempo do Q-a: mediana {med:.1f} s por recorte (p75 {secs.quantile(0.75):.1f} s)")
    print(f"Estimativa: {todo.sum() * med / 60:.0f} min pela mediana; "
          f"{todo.sum() * secs.mean() / 60:.0f} min pela média")


if __name__ == "__main__":
    main()
