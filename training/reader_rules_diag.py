"""Diagnóstico das regras abertas do leitor (ET6c, T6c.1). Só contagens.

1. Campos que chegam ao leitor por foto (lista de recortes da ET6a): display e número
   de série, só display, só número de série ou nenhum, por família.
2. Situação da seleção dos campos nas fotos anotadas (predições fora da dobra da
   semente 0 do S1_coco, sem pesos): ambíguos por tipo (inclui o G1) e sem display.
3. Tambores em transição: alvos resolvidos e descartados pela T-a, e o acerto do
   comitê fora da dobra nos resolvidos, contra os demais ciclométricos.
4. Função: classes por origem do rótulo e os códigos combinados YY(Y)XX.
"""

from __future__ import annotations

from collections import Counter

import pandas as pd


def field_combinations(crops: pd.DataFrame, codes) -> pd.DataFrame:
    """Uma linha por foto de `codes`: a família do display recebido ("digital",
    "ciclometrico" ou "") e a combinação dos campos."""
    by_photo = crops.groupby("pseudonimo")["classe"].apply(set).to_dict()
    rows = []
    for code in sorted(codes):
        classes = by_photo.get(code, set())
        family = ("digital" if "display_digital" in classes
                  else "ciclometrico" if "display_ciclometrico" in classes else "")
        serial = "numero_serie" in classes
        combo = ("display_e_serie" if family and serial else "so_display" if family
                 else "so_serie" if serial else "nenhum")
        rows.append({"pseudonimo": code, "familia": family, "combinacao": combo})
    return pd.DataFrame(rows, columns=["pseudonimo", "familia", "combinacao"])


def selection_situation(sel) -> str:
    """Situação de uma `FieldSelection`: o tipo de ambiguidade, a falta de display
    (com ou sem número de série) ou o consumo identificado."""
    if sel.ambiguity is not None:
        return sel.ambiguity
    if sel.family is None:
        return "sem_display_com_serie" if sel.serial is not None else "sem_display_sem_serie"
    return "consumo_identificado"


def main() -> None:
    from meter_reader.config import get_work_dir
    from meter_reader.fields import select_fields
    from training.box_sample import population
    from training.crop_margin import dev_predictions
    from training.reader_dataset import CYCLOMETRIC, FULL, TARGET_STATUSES, load_items
    from training.reader_ensemble import ENSEMBLE
    from training.reader_eval import EVAL_DIR
    from training.reader_function_r12 import extended_items
    from training.reader_heads_train import FUNCTION_CLASSES, function_class
    from training.reader_labels import CROPS_NAME
    from training.reader_labels_r9 import R9_LABELS_NAME

    pd.set_option("display.width", 160)
    work_dir = get_work_dir()

    # 1. Campos recebidos pelo leitor, nas fotos do desenvolvimento que passam pela triagem
    pop = population()
    codes = set(pop.loc[(pop["conjunto"] == "desenvolvimento") & pop["passa"], "pseudonimo"])
    crops = pd.read_csv(work_dir / CROPS_NAME, float_precision="round_trip")
    combos = field_combinations(crops, codes)
    print(f"1. Campos recebidos pelo leitor ({len(combos)} fotos que passam pela triagem):")
    print(pd.crosstab(combos["combinacao"], combos["familia"].replace("", "sem_display"),
                      margins=True, margins_name="total").to_string())

    # 2. Situação da seleção nas fotos anotadas (fora da dobra)
    photos, preds, cut = dev_predictions()
    sits = Counter(selection_situation(select_fields(p["boxes"], p["scores"], p["labels"], cut))
                   for photo, p in zip(photos, preds) if photo.pseudonym in codes)
    total = sum(sits.values())
    print(f"\n2. Seleção dos campos nas fotos anotadas que passam ({total}; sem pesos):")
    for name, n in sits.most_common():
        print(f"  {name}: {n} ({n / total:.1%})")

    # 3. Tambores em transição (T-a) e acerto do comitê fora da dobra
    items, counts = load_items()
    n_cyc = int(((items["classe"] == CYCLOMETRIC) & items["usa_texto"]).sum())
    print(f"\n3. Transição: {counts['transicao_resolvida']} resolvidas e "
          f"{counts['transicao_descartada']} descartadas; {n_cyc} alvos ciclométricos")
    labels = pd.read_csv(work_dir / R9_LABELS_NAME, dtype=str, keep_default_na=False)
    trans = labels[(labels["classe"] == CYCLOMETRIC) & labels["status"].isin(TARGET_STATUSES)
                   & labels["texto"].str.contains("[", regex=False)]
    keys = set(trans["pseudonimo"] + "|" + CYCLOMETRIC + "|" + FULL)
    comite = pd.read_csv(work_dir / EVAL_DIR / f"{ENSEMBLE}.csv")
    cyc = comite[comite["classe"] == CYCLOMETRIC]
    in_trans = cyc["chave"].isin(keys)
    print(f"  Acerto do comitê fora da dobra: transição {int(cyc.loc[in_trans, 'acerto'].sum())}"
          f" de {int(in_trans.sum())}; demais ciclométricos "
          f"{cyc.loc[~in_trans, 'acerto'].mean():.3f} (de {int((~in_trans).sum())})")

    # 4. Função: classes por origem do rótulo e códigos combinados
    ext = extended_items()
    f = ext[ext["usa_funcao"]].copy()
    f["classe_funcao"] = [FUNCTION_CLASSES[function_class(c)] for c in f["funcao"]]
    print("\n4. Função: classe x origem do rótulo")
    print(pd.crosstab(f["classe_funcao"], f["origem"], margins=True,
                      margins_name="total").to_string())
    combined = f[f["funcao"].str.len() > 3]
    print(f"  Códigos combinados YY(Y)XX: {len(combined)}; por classe do prefixo: "
          + ", ".join(f"{k} {v}" for k, v in combined["classe_funcao"].value_counts().items()))


if __name__ == "__main__":
    main()
