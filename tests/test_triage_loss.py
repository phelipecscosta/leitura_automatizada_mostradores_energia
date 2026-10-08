"""Testes da perda da triagem (training/triage_loss.py)."""

import math

import pytest
import torch
import torch.nn.functional as F

from training import triage_loss as tl
from training.labels import MASKED, SCENE_METER_ONLY

UNIFORM_W = torch.ones(tl.N_SCENE + 1)


def test_uniform_logits_give_ln3_for_exact_labels():
    logits = torch.zeros(4, 3)
    targets = torch.tensor([0, 1, 2, 0])
    assert tl.scene_loss(logits, targets, UNIFORM_W).item() == pytest.approx(math.log(3))


def test_partial_label_is_minus_log_of_meter_mass():
    logits = torch.tensor([[2.0, -1.0, 0.5]])
    targets = torch.tensor([SCENE_METER_ONLY])
    p = F.softmax(logits, dim=1)[0]
    expected = -math.log(p[0].item() + p[1].item())
    assert tl.scene_loss(logits, targets, UNIFORM_W).item() == pytest.approx(expected, rel=1e-5)


def test_partial_label_does_not_force_a_meter_type():
    # Mesma massa de medidor, distribuída de formas diferentes: mesma perda
    a = torch.tensor([[3.0, -9.0, 0.0]])
    b = torch.tensor([[-9.0, 3.0, 0.0]])
    t = torch.tensor([SCENE_METER_ONLY])
    assert tl.scene_loss(a, t, UNIFORM_W).item() == pytest.approx(tl.scene_loss(b, t, UNIFORM_W).item())


def test_scene_weights_balanced_and_partial_is_mean_meter_weight():
    targets = torch.tensor([0, 0, 0, 1, 2, SCENE_METER_ONLY])
    w = tl.scene_weights(targets)
    # 5 exemplos exatos, contagens [3, 1, 1]: pesos 5/9, 5/3, 5/3
    assert w[:3].tolist() == pytest.approx([5 / 9, 5 / 3, 5 / 3])
    # Indeterminado: (5/9·3 + 5/3·1) / 4
    assert w[3].item() == pytest.approx((5 / 9 * 3 + 5 / 3) / 4)


def test_legibility_is_masked_on_outros():
    logits = torch.tensor([1.0, 5.0, -2.0])
    targets = torch.tensor([1, MASKED, 0])
    pw = torch.tensor(1.0)
    expected = F.binary_cross_entropy_with_logits(torch.tensor([1.0, -2.0]), torch.tensor([1.0, 0.0]))
    assert tl.legibility_loss(logits, targets, pw).item() == pytest.approx(expected.item())


def test_pos_weight_is_negatives_over_positives():
    targets = torch.tensor([1, 0, 0, 0, MASKED])
    assert tl.legibility_pos_weight(targets).item() == pytest.approx(3.0)


def test_total_loss_backpropagates_through_partial_labels():
    scene_logits = torch.randn(3, 3, requires_grad=True)
    leg_logits = torch.randn(3, requires_grad=True)
    scene_t = torch.tensor([SCENE_METER_ONLY, 2, 0])
    leg_t = torch.tensor([1, MASKED, 0])
    total, _, _ = tl.triage_loss(scene_logits, leg_logits, scene_t, leg_t,
                                 tl.scene_weights(scene_t), tl.legibility_pos_weight(leg_t))
    total.backward()
    assert scene_logits.grad is not None and torch.isfinite(scene_logits.grad).all()
    assert leg_logits.grad[1].item() == 0.0  # a foto de outros não influencia a legibilidade
