"""Testes da sonda linear do C0 e do C1 (T4f.4)."""

import numpy as np
import pytest

from training import linear_probe as lp
from training.group_dataset import GroupPhoto
from training.group_labels import GROUPS, TRAINING_GROUP_CLASS, TRAINING_GROUPS
from training.triage_eval import BROAD_CLASS

LOTS = ("2026-05-20", "2026-05-21", "2026-05-22")


def dataset(per_group=6, views=2, dim=16, seed=0):
    """Fotos sintéticas: cada grupo de treino tem média própria; 1 órfã por grupo e lote."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=4.0, size=(len(TRAINING_GROUPS), dim))
    photos, feats, k = [], [], 0
    for lot in LOTS:
        for t, group in enumerate(TRAINING_GROUPS):
            for i in range(per_group):
                photos.append(GroupPhoto(f"F-{k:012x}", lot, "treino", i > 0, group, "raiz"))
                feats.append(centers[t] + rng.normal(size=(views, dim)))
                k += 1
    return photos, np.stack(feats).astype(np.float32)


def test_class_weights_average_one():
    y = np.array([0] * 8 + [1] * 2 + [2, 3, 4])
    w = lp.class_weights(y)
    assert np.isclose((w[y]).sum(), len(y))  # média dos pesos por foto = 1
    with pytest.raises(ValueError):
        lp.class_weights(np.array([0, 1, 2, 3]))  # classe 4 ausente


@pytest.mark.parametrize("weighting", lp.WEIGHTINGS)
def test_out_of_fold_logits_only_for_paired(weighting):
    photos, x = dataset()
    targets = np.array([p.target for p in photos])
    logits, index, folds = lp.cross_validate_probe(x, photos, targets,
                                                   TRAINING_GROUP_CLASS, weighting)
    assert logits.shape == (len(index), len(TRAINING_GROUPS))
    assert all(photos[i].paired for i in index)          # órfãs não são avaliadas
    assert sorted(index.tolist()) == [i for i, p in enumerate(photos) if p.paired]
    assert [f.held_out_lot for f in folds] == list(LOTS)
    # Dados separáveis: a sonda acerta quase tudo
    assert (logits.argmax(axis=1) == targets[index]).mean() > 0.9


def test_broad_ablation_has_five_outputs():
    photos, x = dataset()
    targets = np.array([p.product_class for p in photos])
    logits, _, _ = lp.cross_validate_probe(x, photos, targets, BROAD_CLASS)
    assert logits.shape[1] == 5


def test_lbfgs_ignores_random_state():
    # Evidência da revisão R1: duas sementes, coeficientes idênticos
    from sklearn.linear_model import LogisticRegression
    photos, x = dataset()
    rows, y = x[:, 0], np.array([p.target for p in photos])
    a = LogisticRegression(max_iter=2000, random_state=0).fit(rows, y).coef_
    b = LogisticRegression(max_iter=2000, random_state=99).fit(rows, y).coef_
    assert np.array_equal(a, b)


def test_missing_output_in_training_fold_is_refused():
    photos, x = dataset()
    targets = np.array([p.target for p in photos])
    # Um grupo só existe num lote: na dobra que deixa esse lote de fora, ele falta
    keep = [i for i, p in enumerate(photos)
            if not (p.group == "multiplos" and p.lot != LOTS[0])]
    with pytest.raises(ValueError, match="Alguma saída"):
        lp.cross_validate_probe(x[keep], [photos[i] for i in keep], targets[keep],
                                TRAINING_GROUP_CLASS)
    assert "multiplos" in GROUPS
