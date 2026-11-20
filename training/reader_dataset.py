"""Alvos do leitor e partição (protocolo do leitor, seções 1.1 a 1.3).

Cada item de treino é um recorte (ou a faixa de cima dele) com as máscaras das três
perdas: texto principal (CTC), função (CTC, só digital) e "duas leituras" (só
digital). Regras: Z-a (texto com os zeros), T-a (transição pela alternativa igual ao
CSV), V-a, F-c e D-b. Só contagens são exibidas.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np
import pandas as pd

from training.teacher_eval import reading_values

TARGET_STATUSES = ("aceito", "humano")
DIGITAL, CYCLOMETRIC, SERIAL = "display_digital", "display_ciclometrico", "numero_serie"
FULL, UPPER = "inteiro", "faixa_cima"
INTERNAL_FRACTION = 0.10   # divisão interna, só para a época (seção 1.2)
INTERNAL_SEED = "leitor-divisao-interna"
CACHE_SHA256 = "085aca5dc344c049c3d4cce3aaa6c07426bcac7a1299ad0e20cc246db036c58f"
# Rótulos finais da ET6a (seção 1.1 do protocolo): colar o valor completo
FINAL_LABELS_SHA256 = "577848233ea1eb76a99be3deda8d433ad09b0c764d1611b8e873b6b34cf8be13"

ITEM_COLUMNS = ("pseudonimo", "classe", "lote", "recorte", "texto", "funcao",
                "duas_leituras", "usa_texto", "usa_funcao", "usa_duas", "origem")
_TOKEN_RE = re.compile(r"\[[0-9]{2}\]|[0-9]")


def resolve_transition(text: str, expected: int) -> str | None:
    """T-a: troca cada tambor em transição [ab] pelo dígito que faz a leitura igual ao
    CSV (comparação numérica). None se nenhuma combinação bate ou se há mais de uma."""
    if "[" not in text:
        return text
    tokens = _TOKEN_RE.findall(text)
    if "".join(tokens) != text:
        return None
    options = [[t[1], t[2]] if t.startswith("[") else [t] for t in tokens]
    matches = set()
    for combo in np.array(np.meshgrid(*options), dtype=object).T.reshape(-1, len(options)):
        candidate = "".join(combo)
        if int(candidate) == expected:
            matches.add(candidate)
    return matches.pop() if len(matches) == 1 else None


def _item(r: dict, lot: str, kind: str, text, function, two, use_text, use_function,
          use_two) -> dict:
    return {"pseudonimo": r["pseudonimo"], "classe": r["classe"], "lote": lot,
            "recorte": kind, "texto": text, "funcao": function, "duas_leituras": two,
            "usa_texto": use_text, "usa_funcao": use_function, "usa_duas": use_two,
            "origem": r.get("origem", "professor")}


def build_items(labels: pd.DataFrame, expected: dict[str, int], lots: dict[str, str],
                raw: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Itens de treino a partir dos rótulos finais.

    `expected`: pseudônimo -> leitura do CSV (inteiro); `lots`: pseudônimo -> lote;
    `raw`: arquivo bruto do professor, para a cabeça "duas leituras" nos recortes
    digitais sem alvo de texto.
    """
    items, counts = [], {"transicao_resolvida": 0, "transicao_descartada": 0}
    target = labels[labels["status"].isin(TARGET_STATUSES)]
    for r in target.to_dict("records"):
        lot, cls, text = lots[r["pseudonimo"]], r["classe"], r["texto"]
        if cls != SERIAL and "[" in text:
            text = resolve_transition(text, expected[r["pseudonimo"]])
            counts["transicao_resolvida" if text else "transicao_descartada"] += 1
            if text is None:
                continue
        if cls != DIGITAL:
            items.append(_item(r, lot, FULL, text, "", False, True, False, False))
        elif r["duas_leituras"] == "True":
            # D-b: o recorte inteiro treina só a cabeça; a faixa de cima, as duas CTC
            items.append(_item(r, lot, FULL, "", "", True, False, False, True))
            items.append(_item(r, lot, UPPER, text, r["funcao"], False, True, True, False))
        else:
            items.append(_item(r, lot, FULL, text, r["funcao"], False, True, True, True))

    if raw is not None:
        have = set(target["pseudonimo"] + "|" + target["classe"])
        extra = raw[(raw["classe"] == DIGITAL)
                    & raw["duas_leituras"].isin(["True", "False"])
                    & ~(raw["pseudonimo"] + "|" + raw["classe"]).isin(have)]
        for r in extra.to_dict("records"):
            if r["pseudonimo"] in lots:
                items.append(_item({**r, "origem": "marca_professor"}, lots[r["pseudonimo"]],
                                   FULL, "", "", r["duas_leituras"] == "True",
                                   False, False, True))
        counts["so_duas_leituras"] = int(len(extra))
    return pd.DataFrame(items, columns=list(ITEM_COLUMNS)), counts


def internal_mask(items: pd.DataFrame, fraction: float = INTERNAL_FRACTION) -> np.ndarray:
    """Divisão interna fixa, por foto: True nas fotos sorteadas para escolher a época.

    O sorteio é pelo hash do pseudônimo com uma semente em texto, então não muda
    com a ordem dos itens nem entre configurações.
    """
    def draw(code: str) -> float:
        digest = hashlib.sha256(f"{INTERNAL_SEED}|{code}".encode()).digest()
        return int.from_bytes(digest[:8], "big") / 2 ** 64
    return items["pseudonimo"].map(draw).to_numpy() < fraction


def fold_masks(items: pd.DataFrame, held_out_lot: str) -> dict[str, np.ndarray]:
    """Máscaras da dobra: treino, divisão interna (dentro dos lotes de treino) e avaliação."""
    evaluate = (items["lote"] == held_out_lot).to_numpy()
    internal = internal_mask(items) & ~evaluate
    return {"treino": ~evaluate & ~internal, "interno": internal, "avaliacao": evaluate}


def load_items() -> tuple[pd.DataFrame, dict]:
    """Itens de treino a partir dos arquivos da pasta de trabalho, com os SHA-256
    conferidos. Devolve os itens e as contagens."""
    from meter_reader.config import get_work_dir
    from training.build_manifest import file_sha256
    from training.group_dataset import LABEL_VERSIONS, load_group_photos
    from training.reader_labels import FINAL_LABELS_NAME, labels_name
    from training.reader_pilot import csv_rows
    from training.teacher import MODEL

    work_dir = get_work_dir()
    if file_sha256(work_dir / FINAL_LABELS_NAME) != FINAL_LABELS_SHA256:
        raise SystemExit("Os rótulos do leitor não são os da ET6a (SHA-256 diferente).")
    text_cols = {c: str for c in ("texto", "valor", "funcao", "duas_leituras", "rotacao")}
    labels = pd.read_csv(work_dir / FINAL_LABELS_NAME, keep_default_na=False,
                         dtype=text_cols, float_precision="round_trip")
    raw = pd.read_csv(work_dir / labels_name(MODEL), keep_default_na=False,
                      dtype=text_cols, float_precision="round_trip")
    photos = load_group_photos(LABEL_VERSIONS["r4"])
    lots = {p.pseudonym: p.lot for p in photos}
    expected = {code: row.reading for code, row in csv_rows(photos).items()}
    return build_items(labels, expected, lots, raw)


def main() -> None:
    items, counts = load_items()
    print("Contagens:", counts)
    print(pd.crosstab([items["classe"], items["recorte"]],
                      [items["usa_texto"], items["usa_duas"]]).to_string())
    for lot in sorted(items["lote"].unique()):
        m = fold_masks(items, lot)
        print(f"Dobra {lot}: " + ", ".join(f"{k} {int(v.sum())}" for k, v in m.items()))


if __name__ == "__main__":
    main()
