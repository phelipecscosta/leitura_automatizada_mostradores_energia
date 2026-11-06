"""Testes das partes puras do comando dos modelos finais (T4g.3)."""

import json

import pytest
import torch

from training import run_final as rf
from training.finetune import build
from training.triage_eval import OOF_FOLDER


def record(epochs):
    lots = ("2026-05-20", "2026-05-21", "2026-05-22")
    keys = [f"{lot}_s{s}" for s in (0, 1, 2) for lot in lots]
    return {"configuracao": {"epocas_escolhidas": dict(zip(keys, epochs))}}


def test_median_epoch_is_one_of_the_nine_epochs():
    assert rf.median_epoch(record([3, 11, 12, 9, 11, 14, 11, 2, 15])) == 11
    with pytest.raises(ValueError):
        rf.median_epoch(record([3, 4]))  # execução incompleta


def test_stop_epoch_reads_the_r3_run_of_the_candidate(tmp_path):
    folder = tmp_path / OOF_FOLDER
    folder.mkdir(parents=True)
    (folder / "c4_wb_fino_r3.json").write_text(json.dumps(record([5] * 9)), encoding="utf-8")
    assert rf.stop_epoch(tmp_path, "C4") == 5
    with pytest.raises(FileNotFoundError):
        rf.stop_epoch(tmp_path, "C3")  # sem a execução R3, não há época de parada


def test_save_and_load_roundtrip_checks_sha_and_never_overwrites(tmp_path):
    torch.manual_seed(0)
    model, _ = build("C2", 12, pretrained=False)
    path = rf.final_path(tmp_path, "C4", 1)
    assert path.name == "c4_final_s1.pt"
    digest = rf.save_final(path, model, {"candidato": "C4", "semente": 1})
    assert json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))["sha256_pesos"] == digest

    loaded = rf.load_final(path, "C2")
    x = torch.zeros(1, 3, 480, 360)
    with torch.no_grad():
        assert torch.equal(loaded(x), model.eval()(x))
    with pytest.raises(FileExistsError):
        rf.save_final(path, model, {})

    path.write_bytes(path.read_bytes() + b"0")  # pesos alterados depois de gravados
    with pytest.raises(ValueError):
        rf.load_final(path, "C2")
