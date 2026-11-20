"""Perfil dos rótulos do leitor para as decisões prévias ao protocolo (T6b.1).

Só contagens e quantis; nenhum texto, valor ou pseudônimo é exibido.
Alvo de treino: status "aceito" ou "humano" (E95).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from meter_reader.crops import CROP_MARGINS

TARGET_STATUSES = ("aceito", "humano")
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def crop_geometry(labels: pd.DataFrame) -> pd.DataFrame:
    """Largura e altura do recorte com a margem, já em pé (rotação aplicada).

    Ignora o corte na borda da foto, então é um limite superior do tamanho real.
    Linhas sem rotação conhecida ficam com NaN.
    """
    mh = labels["classe"].map(lambda c: CROP_MARGINS[c][0])
    mv = labels["classe"].map(lambda c: CROP_MARGINS[c][1])
    w = (labels["x1"] - labels["x0"]) * (1 + 2 * mh)
    h = (labels["y1"] - labels["y0"]) * (1 + 2 * mv)
    rot = pd.to_numeric(labels["rotacao"], errors="coerce")
    turned = rot.isin([90, 270])
    out = pd.DataFrame({"largura": np.where(turned, h, w), "altura": np.where(turned, w, h)},
                       index=labels.index)
    out[rot.isna()] = np.nan
    return out


def leading_zero(text: pd.Series) -> pd.Series:
    """Texto com zero à esquerda (mais de um caractere e começa por 0)."""
    return (text.str.len() > 1) & text.str.startswith("0")


def profile(labels: pd.DataFrame, lots: pd.Series) -> dict[str, pd.DataFrame]:
    """Tabelas de contagens e quantis por classe. `lots`: pseudônimo -> lote."""
    df = labels.copy()
    df["lote"] = df["pseudonimo"].map(lots).fillna("sem_lote")
    target = df[df["status"].isin(TARGET_STATUSES)].copy()
    out = {"alvo_por_lote": pd.crosstab(target["classe"], target["lote"], margins=True)}

    out["rotacao"] = pd.crosstab(target["classe"], target["rotacao"].replace("", "vazia"))

    dig = target[target["classe"] == "display_digital"]
    out["digital"] = pd.DataFrame({
        "alvos": [len(dig)],
        "duas_leituras": [int((dig["duas_leituras"] == "True").sum())],
        "zero_a_esquerda": [int(leading_zero(dig["texto"]).sum())],
        "sem_funcao": [int((dig["funcao"] == "").sum())],
    })
    out["funcao"] = dig["funcao"].replace("", "vazia").value_counts().to_frame("recortes")

    cyc = target[target["classe"] == "display_ciclometrico"]
    out["ciclometrico"] = pd.DataFrame({
        "alvos": [len(cyc)],
        "com_transicao": [int(cyc["texto"].str.contains(r"\[", regex=True).sum())],
    })

    ser = target[target["classe"] == "numero_serie"]
    out["serie_letras"] = pd.DataFrame({
        "alvos": [len(ser)],
        "com_letra": [int(ser["texto"].str.contains(r"[A-Z]", regex=True).sum())],
    })

    # Número de caracteres lidos; uma transição [56] conta como um tambor
    n_chars = target["texto"].str.replace(r"\[[0-9]{2}\]", "#", regex=True).str.len()
    out["caracteres"] = n_chars.groupby(target["classe"]).describe()[["min", "50%", "max"]]

    geo = crop_geometry(target)
    geo["razao"] = geo["largura"] / geo["altura"]
    geo["classe"] = target["classe"]
    out["altura_px"] = geo.groupby("classe")["altura"].quantile(QUANTILES).unstack().round(1)
    out["razao_largura_altura"] = (geo.groupby("classe")["razao"].quantile(QUANTILES)
                                   .unstack().round(2))
    return out


def main() -> None:
    from meter_reader.config import get_work_dir
    from training.group_dataset import LABEL_VERSIONS, read_group_labels
    from training.reader_labels import FINAL_LABELS_NAME, labels_name
    from training.teacher import MODEL

    work_dir = get_work_dir()
    text_cols = {c: str for c in ("texto", "valor", "funcao", "duas_leituras", "rotacao")}

    def read(name: str) -> pd.DataFrame:
        return pd.read_csv(work_dir / name, keep_default_na=False, dtype=text_cols,
                           float_precision="round_trip")

    # Rótulos finais (alvos de treino) e lote de cada foto pela R4 do desenvolvimento
    labels = read(FINAL_LABELS_NAME)
    name, sha256 = LABEL_VERSIONS["r4"]
    lots = pd.Series({p.pseudonym: p.lot for p in read_group_labels(work_dir / name, sha256)})

    with pd.option_context("display.width", 140):
        for title, table in profile(labels, lots).items():
            print(f"\n== {title} ==")
            print(table.to_string())

        # Arquivo bruto do Haiku: todos os recortes, aceitos ou não (só contagens)
        raw = read(labels_name(MODEL))

        print("\n== rotacao_todos (Haiku, por status) ==")
        print(pd.crosstab([raw["classe"], raw["status"]],
                          raw["rotacao"].replace("", "vazia")).to_string())

        dig_raw = raw[raw["classe"] == "display_digital"]
        print("\n== digital: caracteres da funcao, por status (Haiku) ==")
        print(pd.crosstab(dig_raw["status"], dig_raw["funcao"].str.len()).to_string())

        print("\n== digital: caracteres da leitura, por status (Haiku) ==")
        n_chars = dig_raw["texto"].str.replace(r"\[[0-9]{2}\]", "#", regex=True).str.len()
        print(pd.crosstab(dig_raw["status"], n_chars).to_string())

if __name__ == "__main__":
    main()
