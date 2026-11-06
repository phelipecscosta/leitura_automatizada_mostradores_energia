"""Modelos finais da triagem de cinco classes: C3 e C4 (revisão R5; T4g.3).

Uso, a partir da raiz do repositório do produto:
    python -m training.run_final

Para cada semente, treina o C3 final e, com ele como professor, o C4 final,
com as fotos dos 3 lotes de desenvolvimento (pareadas e órfãs) e os rótulos
R3. A época de parada é a mediana das épocas escolhidas na execução R3 de
cada candidato. Cada rede é gravada assim que termina, em
modelos/triagem5/, com os metadados; uma nova chamada retoma do ponto em que
parou.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from meter_reader.config import get_work_dir
from training.evaluation import _product_commit
from training.finetune import CANDIDATES, CONFIGS, PhotoDataset, build, distill_input, train_fixed
from training.group_dataset import LABEL_VERSIONS, check_group_sizes, load_group_photos
from training.run_finetune import SEEDS, run_name
from training.triage_eval import OOF_FOLDER

FINAL_FOLDER = Path("modelos") / "triagem5"
VERSION = "r3"  # rótulos do modelo final (revisão R5)


def median_epoch(record: dict) -> int:
    """Mediana das épocas escolhidas (3 lotes × 3 sementes) de uma execução gravada."""
    epochs = list(record["configuracao"]["epocas_escolhidas"].values())
    if len(epochs) != 9:
        raise ValueError("A execução precisa ter 9 épocas escolhidas (3 lotes × 3 sementes).")
    return int(np.median(epochs))  # 9 valores: a mediana é um deles, sem arredondamento


def stop_epoch(work_dir: Path, candidate: str) -> int:
    """Época de parada do candidato, lida dos metadados da execução R3."""
    path = work_dir / OOF_FOLDER / f"{run_name(candidate, 'W-b', VERSION)}.json"
    return median_epoch(json.loads(path.read_text(encoding="utf-8")))


def final_path(work_dir: Path, candidate: str, seed: int) -> Path:
    return work_dir / FINAL_FOLDER / f"{candidate.lower()}_final_s{seed}.pt"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_final(path: Path, model: torch.nn.Module, metadata: dict) -> str:
    """Grava os pesos (CPU) e o JSON ao lado; nunca sobrescreve. Devolve o SHA-256."""
    if path.exists():
        raise FileExistsError("Modelo final já gravado.")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, path)
    digest = sha256_file(path)
    record = {**metadata, "sha256_pesos": digest,
              "data": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    path.with_suffix(".json").write_text(json.dumps(record, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    return digest


def load_final(path: Path, candidate_arch: str) -> torch.nn.Module:
    """Rede gravada, conferida pelo SHA-256 do JSON ao lado, em modo de avaliação."""
    record = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if sha256_file(path) != record["sha256_pesos"]:
        raise ValueError("Os pesos não correspondem ao SHA-256 gravado.")
    model, _ = build(candidate_arch, 12, pretrained=False)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.eval()


def run(device: str = "cuda") -> None:
    work_dir = get_work_dir()
    labels = LABEL_VERSIONS[VERSION]
    photos = load_group_photos(labels)  # confere o SHA-256 dos rótulos R3
    check_group_sizes(photos)
    stops = {c: stop_epoch(work_dir, c) for c in ("C3", "C4")}
    print(f"{len(photos)} fotos; épocas de parada: {stops}", flush=True)

    def dataset(seed, to_input):
        return PhotoDataset([p.path for p in photos], [p.pseudonym for p in photos],
                            [p.target for p in photos], to_input, True, run_seed=seed)

    def metadata(candidate, seed):
        cfg = CONFIGS["C2" if candidate == "C4" else candidate]
        return {"candidato": candidate, "semente": seed, "epoca_parada": stops[candidate],
                "execucao_origem": run_name(candidate, "W-b", VERSION),
                "treino": asdict(cfg), "fotos": len(photos), "versao_rotulos": VERSION,
                "sha256_rotulos": labels[1], "commit_produto": _product_commit()}

    for seed in SEEDS:
        c3_path, c4_path = final_path(work_dir, "C3", seed), final_path(work_dir, "C4", seed)
        if not c3_path.exists():
            model = train_fixed("C3", dataset(seed, CANDIDATES["C3"][1]), stops["C3"], seed,
                                device, log=lambda m, s=seed: print(f"[C3 s{s}] {m}", flush=True))
            print(f"[C3 s{seed}] gravado: {save_final(c3_path, model, metadata('C3', seed))[:12]}…")
            del model
            torch.cuda.empty_cache()
        if not c4_path.exists():
            teacher = load_final(c3_path, "C3")  # professor: o C3 final da mesma semente (R5)
            model = train_fixed("C2", dataset(seed, distill_input), stops["C4"], seed, device,
                                config=CONFIGS["C2"], teacher=teacher,
                                log=lambda m, s=seed: print(f"[C4 s{s}] {m}", flush=True))
            meta = {**metadata("C4", seed), "professor_sha256": sha256_file(c3_path)}
            print(f"[C4 s{seed}] gravado: {save_final(c4_path, model, meta)[:12]}…")
            del model, teacher
            torch.cuda.empty_cache()


if __name__ == "__main__":  # obrigatório no Windows: os processos auxiliares reimportam o módulo
    run(*sys.argv[1:2])
    print("Concluído.")
