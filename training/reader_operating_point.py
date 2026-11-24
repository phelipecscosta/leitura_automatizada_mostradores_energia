"""Ponto de operação e riscos do leitor final (protocolo do leitor, seções 1.7 e 1.11;
revisões R11 e R12). Só números.

1. Limiar alto por classe e semente: confirmação falsa dos impostores (limite
   superior de Clopper-Pearson), taxa dos uniformes, cobertura e acerto; e a
   cobertura por classe e lote.
2. Casos reais: alvos de leitura (digital e ciclométrico) cujo valor verdadeiro é
   diferente do CSV (F-2 e R9). Confirmação falsa real: a leitura livre igual ao CSV
   com a probabilidade do CSV acima do limiar alto.
3. Função (cabeças da R12): curva do limiar de consumo, P(03) + P(003). Para cada
   limiar, a fração das funções que não são de consumo aceita como consumo e a dos
   consumos mantida. Relato para o validador 13.3 da ET6c; nada é decidido aqui.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from training.reader_heads_train import CONFIG, FUNCTION_CLASSES, HEADS_DIR, function_class

MAX_NON_CONSUMPTION = 0.02          # seção 1.8: no máximo 2% de 103 lidos como consumo
CONSUMPTION = [FUNCTION_CLASSES.index("03"), FUNCTION_CLASSES.index("003")]
NON_CONSUMPTION = [FUNCTION_CLASSES.index(c) for c in ("103", "24", "55", "outro")]


def gate_curve(p_consumption: np.ndarray, y: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    """Para cada limiar t: fração das funções que não são de consumo com P(consumo) >= t
    e fração dos consumos mantidos."""
    cons, non = np.isin(y, CONSUMPTION), np.isin(y, NON_CONSUMPTION)
    rows = []
    for t in thresholds:
        passed = p_consumption >= t
        rows.append({"limiar": round(float(t), 4), "nao_consumo_aceito": int(passed[non].sum()),
                     "nao_consumo": int(non.sum()),
                     "taxa_nao_consumo": float(passed[non].mean()) if non.any() else 0.0,
                     "consumo_mantido": float(passed[cons].mean()) if cons.any() else 0.0})
    return pd.DataFrame(rows)


def lowest_safe(curve: pd.DataFrame, max_rate: float = MAX_NON_CONSUMPTION,
                upper: bool = False) -> pd.Series | None:
    """Menor limiar com a taxa de não consumo aceita de no máximo `max_rate` (pela
    estimativa ou, com `upper`, pelo limite superior de Clopper-Pearson)."""
    from training.reader_metrics import cp_upper

    for _, r in curve.sort_values("limiar").iterrows():
        rate = cp_upper(r["nao_consumo_aceito"], r["nao_consumo"]) if upper else r["taxa_nao_consumo"]
        if rate <= max_rate:
            return r
    return None


def main() -> None:
    from meter_reader.config import get_work_dir
    from training.group_dataset import LABEL_VERSIONS, load_group_photos
    from training.reader_cache import CACHE_DIR, CACHE_NAME
    from training.reader_dataset import fold_masks, load_items
    from training.reader_eval import score_config, value_logprob
    from training.reader_function_r12 import R12_HEADS, extended_items, predict_function
    from training.reader_labels import CROPS_SHA256
    from training.reader_metrics import config_table
    from training.reader_pilot import csv_rows
    from training.reader_train import META_NAME, PREDICTIONS_NAME, RUNS_DIR, WEIGHTS_NAME
    from meter_reader.reader_text import TEXT_ALPHABET, greedy_decode

    pd.set_option("display.width", 180)
    work_dir = get_work_dir()
    items, _ = load_items()
    scores = score_config(work_dir, CONFIG, items)
    table = config_table(scores)

    # 1. Ponto de operação
    print("1. Limiar alto por classe e semente:")
    print(table.round(4).to_string(index=False))
    highs = table.set_index(["semente", "classe"])["log_limiar"]
    from training.reader_metrics import covered
    cov = []
    for (seed, cls, lot), part in scores.groupby(["semente", "classe", "lote"]):
        cov.append({"classe": cls, "lote": lot, "semente": seed,
                    "cobertura": float(covered(part, highs.loc[(seed, cls)]).mean())})
    print("\nCobertura por classe e lote (média das sementes):")
    print(pd.DataFrame(cov).groupby(["classe", "lote"])["cobertura"].mean().round(3)
          .unstack().to_string())

    # 2. Casos reais de CSV diferente do valor verdadeiro
    rows = csv_rows(load_group_photos(LABEL_VERSIONS["r4"]))
    index = items.set_index(items["pseudonimo"] + "|" + items["classe"] + "|" + items["recorte"])
    real = []
    for meta_path in sorted((work_dir / RUNS_DIR / CONFIG).glob(f"*/{META_NAME}")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        pred = np.load(meta_path.parent / PREDICTIONS_NAME)
        for key, lp in zip(pred["chave_texto"], pred["texto_0"]):
            r = index.loc[key]
            if r["classe"] == "numero_serie" or r["pseudonimo"] not in rows:
                continue
            expected = int(rows[r["pseudonimo"]].reading)
            if int(r["texto"]) == expected:
                continue
            lp = lp.astype(np.float32)
            free = greedy_decode(lp.argmax(-1).tolist(), TEXT_ALPHABET)
            confirmed = (free.isdigit() and int(free) == expected
                         and value_logprob(lp, str(expected), r["classe"])
                         >= highs.loc[(meta["semente"], r["classe"])])
            real.append({"classe": r["classe"], "origem": r["origem"], "semente": meta["semente"],
                         "confirmado": bool(confirmed),
                         "le_o_valor_certo": free.isdigit() and int(free) == int(r["texto"])})
    real = pd.DataFrame(real)
    print("\n2. Casos reais (valor verdadeiro diferente do CSV), 3 sementes somadas:")
    print(real.groupby(["classe", "origem"]).agg(leituras=("confirmado", "size"),
                                                 confirmacao_falsa=("confirmado", "sum"),
                                                 le_o_valor_certo=("le_o_valor_certo", "sum"))
          .to_string())

    # 3. Função: curva do limiar de consumo
    ext = extended_items()
    cache = work_dir / CACHE_DIR / CACHE_NAME
    p_all, y_all = [], []
    for meta_path in sorted((work_dir / RUNS_DIR / CONFIG).glob(f"*/{META_NAME}")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        part = ext[fold_masks(ext, meta["lote_avaliado"])["avaliacao"]
                   & ext["usa_funcao"]].reset_index(drop=True)
        probs = predict_function(meta_path.parent / WEIGHTS_NAME,
                                 work_dir / HEADS_DIR / R12_HEADS / meta_path.parent.name
                                 / "pesos_cabecas.pt", meta["candidato"], meta["cinza"], part,
                                 cache, CROPS_SHA256)
        p_all.append(probs[:, CONSUMPTION].sum(1))
        y_all.append(part["funcao"].map(function_class).to_numpy())
    p, y = np.concatenate(p_all), np.concatenate(y_all)
    curve = gate_curve(p, y, np.quantile(p, np.linspace(0, 1, 201)))
    print("\n3. Função (R12), limiar de P(03) + P(003), 3 sementes somadas:")
    for label, kw in (("estimativa <= 2%", {}), ("limite superior <= 2%", {"upper": True})):
        r = lowest_safe(curve, **kw)
        print(f"  {label}: " + ("nenhum limiar" if r is None else
              f"limiar {r['limiar']}; não consumo aceito {int(r['nao_consumo_aceito'])} de "
              f"{int(r['nao_consumo'])}; consumo mantido {r['consumo_mantido']:.3f}"))
    print(curve.iloc[::20][["limiar", "taxa_nao_consumo", "consumo_mantido"]].round(3)
          .to_string(index=False))


if __name__ == "__main__":
    main()
