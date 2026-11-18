"""Testes da medição e da carga parcial dos pesos COCO na ablação do S1 (T5c.1).

Nenhum download: as origens são sintéticas ou outro S1 montado localmente.
"""
from collections import OrderedDict

import pytest
import torch

from meter_reader.detector import build_ssd
from training import detector_coco as coco

SHAPES = [[(16.0, 16.0)]] * 6  # uma forma por nível, só para montar a rede


def _state(items):
    """state_dict sintético, na ordem dada."""
    return OrderedDict((key, torch.zeros(shape)) for key, shape in items)


def test_group_of():
    assert coco.group_of("backbone.features.0.1.weight") == "espinha"
    assert coco.group_of("backbone.extra.2.0.weight") == "extras"
    assert coco.group_of("head.classification_head.module_list.0.1.bias") == "cabeca_classificacao"
    assert coco.group_of("head.regression_head.module_list.0.1.bias") == "cabeca_regressao"
    assert coco.group_of("anchor_generator.qualquer") == "outros"


def test_compare_state_counts():
    target = _state([("backbone.features.0.a", (2, 3)), ("backbone.features.0.b", (4,)),
                     ("backbone.extra.0.a", (5,))])
    source = _state([("backbone.features.0.a", (2, 3)), ("backbone.features.0.b", (2,))])
    out = coco.compare_state(target, source)
    assert (out["espinha"]["iguais"], out["espinha"]["forma"]) == (1, 1)
    assert (out["espinha"]["valores_iguais"], out["espinha"]["valores_total"]) == (6, 10)
    assert out["extras"]["ausentes"] == 1


def test_prefix_stops_at_first_divergence():
    # A terceira camada tem a mesma forma, mas vem depois da divergência: não entra
    target = _state([("backbone.features.0.a", (2,)), ("backbone.features.1.a", (3,)),
                     ("backbone.features.1.b", (4,)), ("backbone.extra.0.a", (5,))])
    source = _state([("backbone.features.0.a", (2,)), ("backbone.features.1.a", (9,)),
                     ("backbone.features.1.b", (4,)), ("backbone.extra.0.a", (5,))])
    assert coco.prefix_keys(target, source) == ["backbone.features.0.a"]


def test_prefix_stops_at_missing_key():
    target = _state([("backbone.features.0.a", (2,)), ("backbone.features.0.b", (2,))])
    source = _state([("backbone.features.0.b", (2,))])
    assert coco.prefix_keys(target, source) == []


def test_load_prefix_copies_only_the_prefix():
    model = build_ssd(SHAPES)
    donor = build_ssd(SHAPES).state_dict()
    backbone = [k for k in donor if coco.group_of(k) == "espinha"]
    cut = backbone[len(backbone) // 2]
    # Origem com valor 7 nos tensores reais e uma forma diferente no meio da espinha
    source = OrderedDict((k, torch.full_like(v, 7.0) if v.is_floating_point() else v)
                         for k, v in donor.items())
    source[cut] = torch.zeros(1)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    loaded = coco.load_prefix(model, source)
    after = model.state_dict()
    assert loaded == backbone[:backbone.index(cut)]
    assert all(torch.equal(after[k], source[k]) for k in loaded)
    # Tudo o que não está no prefixo fica intacto, inclusive extras e cabeças
    assert all(torch.equal(after[k], before[k]) for k in after if k not in loaded)


def test_load_prefix_rejects_no_match():
    with pytest.raises(ValueError):
        coco.load_prefix(build_ssd(SHAPES), {})
