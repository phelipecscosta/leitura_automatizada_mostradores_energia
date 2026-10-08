"""Testes do modelo final da triagem (training/final_model.py)."""

import numpy as np
import torch

from meter_reader.triage import TriageModel
from training.cross_validation import LinearHeads
from training.final_model import fold_heads


def test_folded_heads_equal_the_step_by_step_computation():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    dim = 6
    heads = [LinearHeads(dim) for _ in range(3)]
    mean, std = rng.normal(0, 1, dim), rng.uniform(0.5, 2, dim)
    log_w, log_pw, t_s, t_l = rng.normal(0, 1, 3), 0.7, 2.2, 1.9
    x = torch.as_tensor(rng.normal(0, 1, (10, dim)), dtype=torch.float32)

    # Passo a passo: padroniza, média das sementes, correção, temperatura
    z = (x - torch.as_tensor(mean, dtype=torch.float32)) / torch.as_tensor(std, dtype=torch.float32)
    with torch.no_grad():
        outs = [h(z) for h in heads]
        scene = (torch.stack([o[0] for o in outs]).mean(0) - torch.as_tensor(log_w, dtype=torch.float32)) / t_s
        leg = (torch.stack([o[1] for o in outs]).mean(0) - log_pw) / t_l
        folded_scene, folded_leg = fold_heads(heads, mean, std, log_w, log_pw, t_s, t_l)(x)

    assert torch.allclose(folded_scene, scene, atol=1e-5)
    assert torch.allclose(folded_leg, leg, atol=1e-5)


def test_folded_heads_load_into_the_product_model():
    heads = [LinearHeads(960) for _ in range(2)]
    folded = fold_heads(heads, np.zeros(960), np.ones(960), np.zeros(3), 0.0, 1.0, 1.0)
    model = TriageModel(pretrained=False)
    model.scene_head.load_state_dict(folded.scene_head.state_dict())
    model.legibility_head.load_state_dict(folded.legibility_head.state_dict())
