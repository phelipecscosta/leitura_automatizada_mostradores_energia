import numpy as np
import pandas as pd

from training.reader_metrics import config_table, cp_upper, high_threshold, summary


def test_clopper_pearson():
    assert cp_upper(0, 0) == 1.0
    assert abs(cp_upper(0, 59) - 0.0495) < 1e-3     # regra de três: 3/n
    assert cp_upper(1, 100) > cp_upper(0, 100)


def _scores(n, read_logps, acerto=0.8, logp_rotulo=-0.1):
    df = pd.DataFrame({"tem_imp_confundivel": [True] * n,
                       "imp_confundivel_lido": [False] * n,
                       "logp_imp_confundivel": [-50.0] * n,
                       "acerto": [i < int(acerto * n) for i in range(n)],
                       "logp_rotulo": [logp_rotulo] * n,
                       "imp_uniforme_lido": [False] * n, "logp_imp_uniforme": [-50.0] * n})
    for i, lp in enumerate(read_logps):
        df.loc[i, ["imp_confundivel_lido", "logp_imp_confundivel"]] = [True, lp]
    return df


def test_limiar_exclui_o_excesso_de_confirmacoes():
    # 100 impostores: com Y = 5% pelo limite superior, cabem até 1 confirmado
    assert cp_upper(1, 100) <= 0.05 < cp_upper(2, 100)
    t = high_threshold(_scores(100, [-0.5, -1.0, -2.0]))
    assert t["confirmados"] == 1 and -1.0 < t["log_limiar"] <= -0.5
    assert high_threshold(_scores(100, []))["log_limiar"] == -np.inf


def test_tabela_e_resumo():
    parts = []
    for seed in (0, 1):
        for cls in ("display_digital", "display_ciclometrico", "numero_serie"):
            df = _scores(100, [])
            df["semente"], df["classe"] = seed, cls
            parts.append(df)
    table = config_table(pd.concat(parts, ignore_index=True))
    s = summary(table)
    assert abs(s["M-c"] - 0.8) < 1e-9 and abs(s["M-a"] - 0.8) < 1e-9
