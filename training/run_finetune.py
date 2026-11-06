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
import torch

from meter_reader.config import get_work_dir
from meter_reader.preprocess import to_tensor
from training.finetune import (CANDIDATES, CONFIGS, DISTILL_ALPHA, DISTILL_T, PhotoDataset,
                               distill_input, train_one)
from training.group_dataset import (LABEL_VERSIONS, GroupPhoto, check_group_sizes, fold_indices,
                                    internal_split, load_group_photos)
from training.group_labels import TRAINING_GROUP_CLASS
from training.linear_probe import WEIGHTINGS, class_weights
from training.triage_eval import OOF_FOLDER, save_oof

SEEDS = (0, 1, 2)


def run_name(candidate: str, weighting: str, version: str = "v1") -> str:
    """Ex.: C2, W-b -> c2_wb_fino; com os rótulos R3 -> c2_wb_fino_r3."""
    base = f"{candidate.lower()}_{weighting.replace('-', '').lower()}_fino"
    return base if version == "v1" else f"{base}_{version}"


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

def save_part(path: Path, codes: list[str], logits: np.ndarray, best_epoch: int,
              history) -> None:
    """Grava uma parte (lote × semente): pseudônimos, logits, época escolhida e histórico."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, codes=np.array(codes), logits=logits, best_epoch=best_epoch,
             history=np.array(history))

def run(candidate: str, weighting: str, version: str = "v1", device: str = "cuda") -> Path:
    if (candidate not in (*CANDIDATES, "C4") or weighting not in WEIGHTINGS
            or version not in LABEL_VERSIONS):
        raise SystemExit("Uso: python -m training.run_finetune {C2|C3|C4} {W-b|W-d} [v1|r3]")
    work_dir = get_work_dir()
    labels = LABEL_VERSIONS[version]
    photos = load_group_photos(labels)  # confere o SHA-256 da versão escolhida
    check_group_sizes(photos)           # a correção não pode esvaziar um grupo (mínimo 30)
    name = run_name(candidate, weighting, version)
    teacher_name = f"{name}_professor"  # C3 retreinado, para medir a reprodutibilidade
    lots = sorted({p.lot for p in photos})

    def dataset(index, train_mode, seed, to_input):
        return PhotoDataset([photos[i].path for i in index], [photos[i].pseudonym for i in index],
                            [photos[i].target for i in index], to_input, train_mode, run_seed=seed)

    for seed in SEEDS:
        for lot in lots:
            tag = f"[{lot} s{seed}]"
            path = part_path(work_dir, name, lot, seed)
            if path.exists():
                print(f"{tag} já gravado, pulando", flush=True)
                continue
            train, held = fold_indices(photos, lot)
            fit, internal = internal_split(photos, train)
            weight = output_weights(photos, train) if weighting == "W-d" else None
            internal_class = np.array([photos[i].product_class for i in internal])
            held_codes = [photos[i].pseudonym for i in held]

            teacher = None
            if candidate == "C4":
                # Professor: C3 da mesma dobra e semente (protocolo, 1.3), sempre W-b
                c3_input = CANDIDATES["C3"][1]
                t_logits, t_epoch, t_history, teacher = train_one(
                    "C3", dataset(fit, True, seed, c3_input), dataset(internal, False, seed, c3_input),
                    internal_class, dataset(held, False, seed, c3_input), seed, None, device,
                    log=lambda m, tag=tag: print(f"{tag} professor {m}", flush=True),
                    return_model=True)
                teacher_path = part_path(work_dir, teacher_name, lot, seed)
                if not teacher_path.exists():
                    save_part(teacher_path, held_codes, t_logits, t_epoch, t_history)
                arch, train_input, eval_input = "C2", distill_input, to_tensor
            else:
                arch = candidate
                train_input = eval_input = CANDIDATES[candidate][1]

            logits, best_epoch, history = train_one(
                arch, dataset(fit, True, seed, train_input), dataset(internal, False, seed, eval_input),
                internal_class, dataset(held, False, seed, eval_input), seed, weight, device,
                log=lambda m, tag=tag: print(f"{tag} {m}", flush=True), teacher=teacher)
            del teacher
            torch.cuda.empty_cache()  # libera o professor antes da próxima parte
            save_part(path, held_codes, logits, best_epoch, history)
            print(f"{tag} gravado; melhor época {best_epoch}", flush=True)

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
              "versao_rotulos": version,
              "treino": asdict(CONFIGS["C2" if candidate == "C4" else candidate]),
              "sementes": list(SEEDS), "epocas_escolhidas": epochs}
    if candidate == "C4":
        config["destilacao"] = {"professor": "C3", "temperatura": DISTILL_T, "alfa": DISTILL_ALPHA}
    return save_oof(work_dir, name, codes, [photos[i].lot for i in order],
                    np.stack(per_seed), config, labels[1])


if __name__ == "__main__":  # obrigatório no Windows: os processos auxiliares reimportam o módulo
    run(*sys.argv[1:4])
    print("Concluído.")
