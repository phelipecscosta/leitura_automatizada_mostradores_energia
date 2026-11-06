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



from training.finetune import train_fixed  # noqa: E402


def fixed_tiny(files, stop_epoch, seed=0):
    paths, codes = files
    data = ft.PhotoDataset(paths, codes, [0, 5, 9, 11], to_tensor, True, run_seed=seed)
    return train_fixed("C2", data, stop_epoch, seed, device="cpu", config=TINY,
                       pretrained=False, log=lambda _: None)


def test_train_fixed_matches_train_one_at_the_same_epoch(files):
    # Até a época de parada, o treino final segue o mesmo caminho do treino por dobras:
    # mesma semente, mesmo cronograma (calculado para max_epochs) e mesmas vistas.
    # A rede do train_one tem os pesos da melhor época; o train_fixed parado nela é igual.
    paths, codes = files
    make = lambda train: ft.PhotoDataset(paths, codes, [0, 5, 9, 11], to_tensor, train)
    _, best_epoch, _, reference = train_one(
        "C2", make(True), make(False), np.array([0, 1, 3, 4]), make(False), 0, device="cpu",
        config=TINY, pretrained=False, log=lambda _: None, return_model=True)
    model = fixed_tiny(files, best_epoch)
    x = torch.stack([to_tensor(enhance(Image.open(p))) for p in paths])
    with torch.no_grad():
        assert torch.allclose(model(x), reference.eval()(x), atol=1e-5)


def test_train_fixed_is_reproducible_and_rejects_bad_stop(files):
    paths, _ = files
    x = torch.stack([to_tensor(enhance(Image.open(p))) for p in paths[:2]])
    first, second = fixed_tiny(files, 1), fixed_tiny(files, 1)
    with torch.no_grad():
        assert torch.allclose(first(x), second(x), atol=1e-5)
    for bad in (0, TINY.max_epochs + 1):
        with pytest.raises(ValueError):
            fixed_tiny(files, bad)
