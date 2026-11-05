"""Ajuste fino de um candidato em todas as dobras e sementes (T4f.6).

Uso, a partir da raiz do repositório do produto:
    python -m training.run_finetune C2 W-b

Cada par lote × semente é gravado assim que termina, em
experimentos/triagem5/partes/<execução>/, e uma nova chamada retoma do ponto
em que parou. Ao fim, as partes são reunidas na ordem de avaliação e gravadas
com save_oof, junto com as épocas escolhidas (protocolo, 1.2).
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from meter_reader.config import get_work_dir
from training.finetune import CANDIDATES, CONFIGS, PhotoDataset, train_one
from training.group_dataset import DEV_LABELS, GroupPhoto, fold_indices, internal_split, load_group_photos
from training.group_labels import TRAINING_GROUP_CLASS
from training.linear_probe import WEIGHTINGS, class_weights
from training.triage_eval import OOF_FOLDER, save_oof

SEEDS = (0, 1, 2)


def run_name(candidate: str, weighting: str) -> str:
    """Ex.: C2 e W-b -> c2_wb_fino (mesmo padrão dos nomes do C0)."""
    return f"{candidate.lower()}_{weighting.replace('-', '').lower()}_fino"


def output_weights(photos: list[GroupPhoto], train: np.ndarray) -> np.ndarray:
    """Peso W-d de cada uma das 12 saídas: o peso da sua classe do produto."""
    product = np.array([photos[i].product_class for i in train])
    return class_weights(product)[np.asarray(TRAINING_GROUP_CLASS)]


def eval_order(photos: list[GroupPhoto]) -> np.ndarray:
    """Índices avaliados, lote a lote em ordem: a mesma ordem do C0 e do C1."""
    lots = sorted({p.lot for p in photos})
    return np.concatenate([fold_indices(photos, lot)[1] for lot in lots])


def part_path(work_dir: Path, name: str, lot: str, seed: int) -> Path:
    return work_dir / OOF_FOLDER / "partes" / name / f"{lot}_s{seed}.npz"


def run(candidate: str, weighting: str, device: str = "cuda") -> Path:
    if candidate not in CANDIDATES or weighting not in WEIGHTINGS:
        raise SystemExit("Uso: python -m training.run_finetune {C2|C3} {W-b|W-d}")
    work_dir = get_work_dir()
    photos = load_group_photos()
    name = run_name(candidate, weighting)
    to_input = CANDIDATES[candidate][1]
    lots = sorted({p.lot for p in photos})

    def dataset(index, train_mode, seed):
        return PhotoDataset([photos[i].path for i in index], [photos[i].pseudonym for i in index],
                            [photos[i].target for i in index], to_input, train_mode, run_seed=seed)

    for seed in SEEDS:
        for lot in lots:
            path = part_path(work_dir, name, lot, seed)
            if path.exists():
                print(f"[{lot} s{seed}] já gravado, pulando", flush=True)
                continue
            train, held = fold_indices(photos, lot)
            fit, internal = internal_split(photos, train)
            weight = output_weights(photos, train) if weighting == "W-d" else None
            internal_class = np.array([photos[i].product_class for i in internal])
            logits, best_epoch, history = train_one(
                candidate, dataset(fit, True, seed), dataset(internal, False, seed),
                internal_class, dataset(held, False, seed), seed, weight, device,
                log=lambda m, tag=f"[{lot} s{seed}]": print(f"{tag} {m}", flush=True))
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, codes=np.array([photos[i].pseudonym for i in held]),
                     logits=logits, best_epoch=best_epoch, history=np.array(history))
            print(f"[{lot} s{seed}] gravado; melhor época {best_epoch}", flush=True)

    # Junta as partes: (sementes, fotos avaliadas, 12), na ordem de avaliação
    order = eval_order(photos)
    codes = [photos[i].pseudonym for i in order]
    per_seed, epochs = [], {}
    for seed in SEEDS:
        parts = [np.load(part_path(work_dir, name, lot, seed)) for lot in lots]
        if [c for p in parts for c in p["codes"]] != codes:
            raise ValueError("Partes fora da ordem de avaliação.")
        per_seed.append(np.concatenate([p["logits"] for p in parts]))
        epochs.update({f"{lot}_s{seed}": int(p["best_epoch"]) for lot, p in zip(lots, parts)})
    config = {"candidato": candidate, "ponderacao": weighting, "rotulo": "fino",
              "treino": asdict(CONFIGS[candidate]), "sementes": list(SEEDS),
              "epocas_escolhidas": epochs}
    return save_oof(work_dir, name, codes, [photos[i].lot for i in order],
                    np.stack(per_seed), config, DEV_LABELS[1])


if __name__ == "__main__":  # obrigatório no Windows: os processos auxiliares reimportam o módulo
    run(*sys.argv[1:3])
    print("Concluído.")
