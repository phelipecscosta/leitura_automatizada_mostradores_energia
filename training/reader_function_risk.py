"""F-x0 (protocolo do leitor, T6b.8c): tamanho do risco da função e da reserva da F-2.

1. Risco real: nos digitais julgados por humano na F-2 com função que não é de
   consumo (103, 24, 55, outro), em quantos o valor mostrado é igual ao do CSV. Só
   nesses a função é a última barreira; nos outros, o validador 13.2 já barra a foto.
2. Reserva: recortes da fila F-2 ainda sem julgamento, por grupo (tipo 1: o professor
   leu outra função; tipo 2: função de consumo com leitura discordante), pela função
   lida pelo professor e por lote, e a estimativa de quantos 103 há no tipo 1.

Só contagens e proporções.
"""

from __future__ import annotations

import pandas as pd
from scipy.stats import beta

from training.reader_heads_train import FUNCTION_CLASSES, function_class

NON_CONSUMPTION = ("103", "24", "55", "outro")


def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Intervalo exato de uma proporção."""
    a = (1 - level) / 2
    low = 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))
    high = 1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k))
    return low, high


def main() -> None:
    from meter_reader.config import get_work_dir
    from training.group_dataset import LABEL_VERSIONS, load_group_photos
    from training.reader_labels import F2_QUEUE_NAME, labels_name
    from training.reader_labels_r9 import R9_LABELS_NAME
    from training.reader_pilot import csv_rows
    from training.teacher import MODEL
    from training.teacher_eval import FUNCTIONS_OK, QUEUE_F2

    work_dir = get_work_dir()
    photos = load_group_photos(LABEL_VERSIONS["r4"])
    lots = {p.pseudonym: p.lot for p in photos}
    rows = csv_rows(photos)
    read = lambda name: pd.read_csv(work_dir / name, dtype=str, keep_default_na=False)
    labels, raw, queue = read(R9_LABELS_NAME), read(labels_name(MODEL)), read(F2_QUEUE_NAME)
    group = dict(zip(queue["pseudonimo"], queue["grupo"]))

    # 1. Risco real, nos julgados completos da F-2
    judged = labels[(labels["origem"] == "humano_f2") & (labels["status"] == "humano")].copy()
    judged["classe_funcao"] = judged["funcao"].map(lambda c: FUNCTION_CLASSES[function_class(c)])
    judged["grupo"] = judged["pseudonimo"].map(group)
    judged["igual_csv"] = [v != "" and p in rows and int(v) == int(rows[p].reading)
                           for v, p in zip(judged["valor"], judged["pseudonimo"])]
    print("1. Julgados completos da F-2: função (humano) x valor mostrado igual ao CSV")
    print(pd.crosstab([judged["grupo"], judged["classe_funcao"]], judged["igual_csv"],
                      margins=True, margins_name="total").to_string())
    other = judged[judged["classe_funcao"].isin(NON_CONSUMPTION)]
    k, n = int(other["igual_csv"].sum()), len(other)
    low, high = clopper_pearson(k, n)
    print(f"\nFunção que não é de consumo com o valor igual ao CSV: {k} de {n} "
          f"({k / max(1, n):.3f}; IC 95% [{low:.3f}; {high:.3f}])")

    # 2. Reserva da F-2 sem julgamento
    done = set(judged["pseudonimo"]) | set(
        labels.loc[labels["origem"] == "humano_f2", "pseudonimo"])
    f2 = raw[(raw["status"] == QUEUE_F2) & ~raw["pseudonimo"].isin(done)].copy()
    f2["grupo"] = f2["funcao"].map(lambda c: "tipo1" if c not in FUNCTIONS_OK else "tipo2")
    f2["funcao_professor"] = f2["funcao"].map(lambda c: FUNCTION_CLASSES[function_class(c)])
    f2["lote"] = f2["pseudonimo"].map(lots).fillna("sem_lote")
    print("\n2. Reserva da F-2 sem julgamento: grupo e função lida pelo professor x lote")
    print(pd.crosstab([f2["grupo"], f2["funcao_professor"]], f2["lote"], margins=True,
                      margins_name="total").to_string())
    t1 = judged[judged["grupo"] == "f2_tipo1"]
    share = float((t1["classe_funcao"] == "103").mean()) if len(t1) else 0.0
    n1 = int((f2["grupo"] == "tipo1").sum())
    print(f"\nNo tipo 1 julgado, fração de 103: {share:.3f} ({int((t1['classe_funcao'] == '103').sum())}"
          f" de {len(t1)}); 103 esperados na reserva do tipo 1: cerca de {share * n1:.0f} de {n1}")


if __name__ == "__main__":
    main()
