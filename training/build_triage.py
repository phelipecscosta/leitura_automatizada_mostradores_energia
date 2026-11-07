"""Temperatura, limiar e artefato da triagem do produto (revisões R5 e R6; T4g.4).

Uso, a partir da raiz do repositório do produto:
    python -m training.build_triage

A temperatura e o limiar vêm da execução R3 de cada candidato (seção 1.6): a
temperatura de implantação é a ajustada nos 3 lotes, e o limiar é o da meta
sobre as predições calibradas de forma aninhada. Grava, em modelos/triagem5/:
- triagem_c4.pt: o artefato do produto (as 3 redes do C4 final, a temperatura
  e o limiar), com o JSON de procedência ao lado;
- triagem_c3.json: a temperatura e o limiar do C3, que é avaliado pelo código
  de treino (revisão R6) e não vai para o produto.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from meter_reader.config import get_work_dir
from meter_reader.contracts import TRIAGE_GROUPS
from meter_reader.triage import ARTIFACT_FORMAT, TriageEnsemble
from training.evaluation import _product_commit
from training.group_dataset import LABEL_VERSIONS, load_group_photos
from training.run_finetune import SEEDS, eval_order, run_name
from training.run_final import FINAL_FOLDER, VERSION, final_path, load_final, sha256_file
from training.triage_eval import (calibrate_nested, choose_threshold, load_oof,
                                  loss_coverage_curve, rejection_score)

ARTIFACT_NAME = "triagem_c4.pt"
C3_CALIBRATION_NAME = "triagem_c3.json"


def calibration(logits: np.ndarray, labels: np.ndarray, lots: np.ndarray) -> dict:
    """Temperatura de implantação e limiar da meta, a partir dos logits fora da dobra."""
    cal = calibrate_nested(logits, labels, lots)
    best = choose_threshold(loss_coverage_curve(rejection_score(cal.probs), labels))
    if best is None:
        raise ValueError("Nenhum limiar cumpre a meta de perda.")
    return {"temperatura": float(cal.final), "limiar": float(best["threshold"]),
            "cobertura": float(best["coverage"]), "perdas": int(best["losses"]),
            "perda": float(best["loss"]), "perda_limite_superior": float(best["loss_upper"]),
            "temperaturas_por_lote": cal.per_lot}


def r3_calibration(work_dir: Path, candidate: str) -> dict:
    """Calibração da execução R3 do candidato (W-b), nas fotos pareadas do desenvolvimento."""
    photos = load_group_photos(LABEL_VERSIONS[VERSION])
    order = eval_order(photos)
    codes = [photos[i].pseudonym for i in order]
    labels = np.array([photos[i].product_class for i in order])
    lots = np.array([photos[i].lot for i in order])
    name = run_name(candidate, "W-b", VERSION)
    return {"execucao_origem": name, **calibration(load_oof(work_dir, name, codes), labels, lots)}


def final_ensemble(work_dir: Path, candidate: str, temperature: float) -> TriageEnsemble:
    """As 3 redes finais do candidato (conferidas pelo SHA-256) com a temperatura."""
    arch = "C2" if candidate == "C4" else candidate
    nets = [load_final(final_path(work_dir, candidate, s), arch) for s in SEEDS]
    return TriageEnsemble(nets, temperature).eval()


def save_artifact(path: Path, ensemble: TriageEnsemble, threshold: float, info: dict) -> str:
    """Grava o artefato do produto e o JSON ao lado; nunca sobrescreve. Devolve o SHA-256."""
    if path.exists():
        raise FileExistsError("Artefato já gravado.")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Ida e volta pelo JSON: só tipos simples no artefato (um escalar do NumPy, como as
    # temperaturas por lote, impede o carregamento seguro com weights_only=True)
    info = json.loads(json.dumps(info))
    data = {"formato": ARTIFACT_FORMAT, "grupos": list(TRIAGE_GROUPS),
            "temperatura": float(ensemble.temperature), "limiar": float(threshold),
            "redes": [{k: v.detach().cpu() for k, v in net.state_dict().items()}
                      for net in ensemble.nets],
            "info": info}
    torch.save(data, path)
    digest = sha256_file(path)
    path.with_suffix(".json").write_text(
        json.dumps({**info, "sha256_artefato": digest}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    return digest


def run() -> None:
    work_dir = get_work_dir()
    folder = work_dir / FINAL_FOLDER
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for candidate in ("C4", "C3"):
        calib = r3_calibration(work_dir, candidate)
        info = {"candidato": candidate, "data": stamp, "commit_produto": _product_commit(),
                "sha256_rotulos": LABEL_VERSIONS[VERSION][1],
                "redes_sha256": [sha256_file(final_path(work_dir, candidate, s)) for s in SEEDS],
                **calib}
        print(f"{candidate}: T = {calib['temperatura']:.4f}; limiar = {calib['limiar']:.6f}; "
              f"cobertura no desenvolvimento = {calib['cobertura']:.1%}; "
              f"perdas = {calib['perdas']} (limite superior {calib['perda_limite_superior']:.3%})")
        if candidate == "C4":
            ensemble = final_ensemble(work_dir, "C4", calib["temperatura"])
            digest = save_artifact(folder / ARTIFACT_NAME, ensemble, calib["limiar"], info)
            print(f"C4: artefato gravado: {digest[:12]}…")
        else:
            path = folder / C3_CALIBRATION_NAME
            if path.exists():
                raise FileExistsError("Calibração do C3 já gravada.")
            path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
            print("C3: calibração gravada.")


if __name__ == "__main__":
    run()
    print("Concluído.")
