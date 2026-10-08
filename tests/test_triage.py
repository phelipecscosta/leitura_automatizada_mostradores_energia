"""Testes do modelo de triagem (meter_reader/triage.py)."""

import pytest
import torch

from meter_reader.triage import BACKBONES, SCENE_CLASSES, TriageModel


@pytest.mark.parametrize("backbone", list(BACKBONES))
def test_output_shapes_accept_non_square_input(backbone):
    model = TriageModel(backbone, pretrained=False).eval()
    x = torch.randn(2, 3, 299, 224)  # retrato 3:4, lado menor 224
    with torch.no_grad():
        scene, legibility = model(x)
    assert scene.shape == (2, len(SCENE_CLASSES))
    assert legibility.shape == (2,)


@pytest.mark.parametrize("backbone", list(BACKBONES))
def test_frozen_backbone_trains_only_the_heads(backbone):
    model = TriageModel(backbone, pretrained=False)
    model.freeze_backbone()
    dim = BACKBONES[backbone][2]
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable == (dim * 3 + 3) + (dim * 1 + 1)  # pesos e vieses das duas cabeças


def test_frozen_backbone_stays_in_eval_mode():
    model = TriageModel("mobilenet_v3_large", pretrained=False)
    model.freeze_backbone()
    model.train()
    assert model.training and not model.features.training


def test_unknown_backbone_fails():
    with pytest.raises(ValueError):
        TriageModel("resnet18", pretrained=False)

def test_rejection_score_is_total_probability():
    from meter_reader.triage import rejection_score
    assert abs(rejection_score(0.2, 0.5) - 0.6) < 1e-12
    assert rejection_score(1.0, 0.0) == 1.0   # certamente outros
    assert rejection_score(0.0, 0.0) == 0.0   # medidor certamente legível
