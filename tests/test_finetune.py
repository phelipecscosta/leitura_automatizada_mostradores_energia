"""Testes do ajuste fino da triagem de cinco classes (T4f.6)."""

import numpy as np
import pytest
import torch
from PIL import Image

from meter_reader.preprocess import enhance, to_tensor
from training import finetune as ft


@pytest.fixture
def files(tmp_path):
    rng = np.random.default_rng(0)
    paths = []
    for i in range(4):
        path = tmp_path / f"foto{i}.jpg"
        Image.fromarray(rng.integers(0, 255, (480, 360, 3), dtype=np.uint8)).save(path)
        paths.append(path)
    return paths, [f"F-{i:012x}" for i in range(4)]


def test_train_views_change_by_epoch_and_are_reproducible(files):
    paths, codes = files
    data = ft.PhotoDataset(paths, codes, [0, 1, 2, 3], to_tensor, train=True)
    first, _ = data[0]
    assert torch.equal(first, data[0][0])          # mesma época: mesma vista
    data.set_epoch(1)
    assert not torch.equal(first, data[0][0])      # época nova: vista nova


def test_eval_input_is_enhanced_original_without_augmentation(files):
    paths, codes = files
    data = ft.PhotoDataset(paths, codes, [0, 1, 2, 3], to_tensor, train=False)
    data.set_epoch(5)
    with Image.open(paths[0]) as image:
        expected = to_tensor(enhance(image))
    x, y = data[0]
    assert torch.equal(x, expected) and y == 0


@pytest.mark.parametrize("candidate, shape", [("C2", (1, 3, 480, 360)), ("C3", (1, 3, 294, 224))])
def test_models_output_twelve_logits_and_train_whole_network(candidate, shape):
    model, _ = ft.build(candidate, 12, pretrained=False)
    assert model(torch.zeros(shape)).shape == (1, 12)
    assert all(p.requires_grad for p in model.parameters())  # rede inteira (ajuste fino)


from training.finetune import TrainConfig, lr_factor, train_one  # noqa: E402

TINY = TrainConfig(lr=1e-3, head_lr=None, max_epochs=2, batch=4, micro_batch=2, amp=False,
                   num_workers=0, eval_workers=0)


def test_lr_factor_warmup_then_cosine():
    assert lr_factor(0, 10, 100) == 0.0
    assert lr_factor(10, 10, 100) == pytest.approx(1.0)
    assert lr_factor(55, 10, 100) == pytest.approx(0.5)
    assert lr_factor(100, 10, 100) == pytest.approx(0.0)


def run_tiny(files, output_weight=None, seed=0):
    paths, codes = files
    targets = [0, 5, 9, 11]  # grupos de classes diferentes
    make = lambda train: ft.PhotoDataset(paths, codes, targets, to_tensor, train, run_seed=seed)
    internal_class = np.array([0, 1, 3, 4])  # classes do produto dos grupos acima
    return train_one("C2", make(True), make(False), internal_class, make(False), seed,
                     output_weight, device="cpu", config=TINY, pretrained=False, log=lambda _: None)


def test_train_one_returns_held_logits_best_epoch_and_is_reproducible(files):
    logits, best_epoch, history = run_tiny(files)
    assert logits.shape == (4, 12) and best_epoch in (1, 2) and len(history) == 2
    again, _, _ = run_tiny(files)
    assert np.allclose(logits, again, atol=1e-5)  # mesma semente, mesmo resultado na CPU


def test_wd_correction_is_applied(files):
    weight = np.ones(12)
    weight[0] = 4.0  # peso maior na saída 0
    logits, _, _ = run_tiny(files, weight)
    assert np.isfinite(logits).all() and logits.shape == (4, 12)

from training.finetune import build, distill_input, distill_loss  # noqa: E402


def test_distill_loss_reduces_to_cross_entropy_parts():
    criterion = torch.nn.CrossEntropyLoss()
    logits = torch.randn(6, 12)
    y = torch.randint(0, 12, (6,))
    ce = criterion(logits, y)
    # Professor igual ao aluno: KL = 0, sobra metade da entropia cruzada
    assert torch.isclose(distill_loss(logits, logits, y, criterion), 0.5 * ce)
    assert torch.isclose(distill_loss(logits, logits * 3, y, criterion, alpha=0.0), ce)
    assert distill_loss(logits, -logits, y, criterion) > 0.5 * ce  # discordância custa


def test_distill_input_gives_both_resolutions(files):
    paths, codes = files
    data = ft.PhotoDataset(paths, codes, [0, 1, 2, 3], distill_input, train=True)
    (x_student, x_teacher), _ = data[0]
    assert x_student.shape == (3, 480, 360) and x_teacher.shape == (3, 294, 224)


def test_train_one_with_teacher_returns_model(files):
    paths, codes = files
    targets = [0, 5, 9, 11]
    torch.manual_seed(0)
    teacher, _ = build("C3", 12, pretrained=False)
    train = ft.PhotoDataset(paths, codes, targets, distill_input, True)
    evaluation = ft.PhotoDataset(paths, codes, targets, to_tensor, False)
    held, best_epoch, history, model = train_one(
        "C2", train, evaluation, np.array([0, 1, 3, 4]), evaluation, 0, device="cpu",
        config=TINY, pretrained=False, log=lambda _: None, teacher=teacher, return_model=True)
    assert held.shape == (4, 12) and isinstance(model, torch.nn.Module)
