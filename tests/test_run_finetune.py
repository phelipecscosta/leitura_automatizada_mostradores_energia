"""Testes das partes puras do comando de ajuste fino (T4f.6)."""

import numpy as np

from training import run_finetune as rf
from training.group_dataset import GroupPhoto
from training.group_labels import TRAINING_GROUP_CLASS, TRAINING_GROUPS
from training.linear_probe import class_weights

LOTS = ("2026-05-20", "2026-05-21", "2026-05-22")


def photos():
    out, k = [], 0
    for lot in LOTS:
        for group in TRAINING_GROUPS:
            for paired in (True, True, False):
                out.append(GroupPhoto(f"F-{k:012x}", lot, "treino", paired, group, "raiz"))
                k += 1
    return out


def test_run_name_follows_c0_pattern_and_version():
    assert rf.run_name("C2", "W-b") == "c2_wb_fino"
    assert rf.run_name("C3", "W-d") == "c3_wd_fino"
    assert rf.run_name("C2", "W-b", "r3") == "c2_wb_fino_r3"


def test_output_weights_map_class_weight_to_each_group():
    ph = photos()
    train = np.arange(len(ph))
    w = rf.output_weights(ph, train)
    expected = class_weights(np.array([p.product_class for p in ph]))
    assert w.shape == (12,)
    assert np.allclose(w, expected[np.asarray(TRAINING_GROUP_CLASS)])


def test_eval_order_is_paired_only_lot_by_lot():
    ph = photos()
    order = rf.eval_order(ph)
    assert all(ph[i].paired for i in order)
    assert [ph[i].lot for i in order] == sorted(ph[i].lot for i in order)
    assert len(order) == sum(p.paired for p in ph)

def test_c4_name_and_save_part_roundtrip(tmp_path):
    assert rf.run_name("C4", "W-b", "r3") == "c4_wb_fino_r3"
    path = rf.part_path(tmp_path, "c4_wb_fino_r3_professor", "2026-05-20", 0)
    rf.save_part(path, ["F-00000000000a"], np.zeros((1, 12)), 3, [0.5, 0.4])
    data = np.load(path)
    assert list(data["codes"]) == ["F-00000000000a"] and int(data["best_epoch"]) == 3
