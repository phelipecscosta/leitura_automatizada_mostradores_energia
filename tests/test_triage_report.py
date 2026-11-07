"""Testes dos relatórios da triagem de cinco classes (T4g.6)."""

import numpy as np
import pytest

from meter_reader.contracts import PRODUCT_CLASSES
from training import triage_report as rp

D, C, I, M, O = range(5)  # digital, ciclométrico, indeterminado, múltiplos, outros


def onehot(classes, high=0.9):
    probs = np.full((len(classes), 5), (1 - high) / 4)
    probs[np.arange(len(classes)), classes] = high
    return probs


def test_summary_counts_losses_coverage_and_checks_the_target():
    labels = np.array([D] * 1000 + [O] * 10)
    rejected = np.zeros(1010, bool)
    rejected[:2] = True          # 2 determinados perdidos
    rejected[1000:1006] = True   # 6 de 10 rejeitáveis cobertos
    s = rp.summary(rejected, labels)
    assert (s["perdas"], s["determinados"], s["cobertos"], s["rejeitaveis"]) == (2, 1000, 6, 10)
    assert s["cobertura"] == pytest.approx(0.6) and s["cobertura_ic_inf"] < 0.6 < s["cobertura_ic_sup"]
    assert s["perda_limite_superior"] > 0.002 and not s["meta_cumprida"]  # 0,63% > 0,5%


def test_by_subset_and_confusion_keep_every_class():
    labels = np.array([D, D, M, O])
    rejected = np.array([False, True, True, False])
    table = rp.by_subset(rejected, labels, np.array(["a", "b", "a", "b"]))
    assert table.loc["b", "perdas"] == 1 and table.loc["a", "cobertos"] == 1
    conf = rp.confusion(onehot([D, D, O, O]), labels)
    assert list(conf.index) == list(PRODUCT_CLASSES) == list(conf.columns)
    assert conf.loc["multiplos", "outros"] == 1 and conf.values.sum() == 4


def test_separation_counts_destination_swaps_and_two_readings():
    labels = np.array([M, M, O, C, C])
    groups = np.array(["multiplos", "multiplos", "out_acesso", "cic_duas_leituras", "cic_duas_leituras"])
    probs = onehot([M, O, O, M, C])        # 2º múltiplos vai para outros; 1 duas leituras parece múltiplos
    rejected = np.array([True, True, True, True, False])
    s = rp.separation(probs, rejected, labels, groups)
    assert s["destino_trocado"] == 1 and s["rejeitaveis_cobertos"] == 3
    assert s["multiplos_rejeitados_como_multiplos"] == 1
    assert (s["duas_leituras_mais_provavel_multiplos"], s["duas_leituras_rejeitadas"]) == (1, 1)


def test_full_report_uses_the_fixed_threshold():
    labels = np.array([D, I, O])
    after = onehot([D, I, O])
    report = rp.full_report(after, after, 0.5, labels, np.array(["x"] * 3),
                            np.array(["dig1_quadrado_branco", "indeterminado", "out_acesso"]),
                            np.array(["raiz"] * 3))
    assert report["resumo"]["cobertos"] == 2 and report["resumo"]["perdas"] == 0
    assert set(report) == {"resumo", "por_lote", "por_condicao", "confusao", "separacao", "calibracao"}
