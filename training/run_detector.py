"""Treino dos candidatos do detector em todas as dobras e sementes (ET5b, T5b.6).

Uso, a partir da raiz do repositório do produto:
    python -m training.run_detector tempo          (1 época de cada candidato; nada é gravado)
    python -m training.run_detector treinar S1     (um candidato: 3 dobras × 3 sementes)
    python -m training.run_detector treinar S1 coco   (ablação COCO, revisão R4: pasta S1_coco)
    python -m training.run_detector final S1 [coco]  (modelo final da semente 0, revisão R4)

Cada par lote × semente é gravado assim que termina, em
<pasta de trabalho>/experimentos/detector/<candidato>/, e uma nova chamada retoma
do ponto em que parou. Cada parte guarda as predições fora da dobra (só
pseudônimos), a época escolhida, o histórico, o corte de confiança e as
impressões digitais dos dados; os pesos da execução ficam ao lado, em .pt.


"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from meter_reader.config import get_work_dir
from training.build_manifest import file_sha256
from meter_reader.detector import SCORE_THRESH
from training.detector_anchors import ANCHORS_NAME, CANDIDATES
from training.detector_dataset import (BOXES_SHA256, DetectorDataset, DetectorPhoto, fold_indices,
                                       internal_split, load_detector_photos)
from training.detector_train import (DetectorTrainConfig, flatten_predictions, make_loader,
                                     predict, time_one_epoch, train_fixed, train_one)

SEEDS = (0, 1, 2)
OUTPUT_FOLDER = Path("experimentos") / "detector"

COCO_SUFFIX = "_coco"  # pasta da ablação COCO (revisão R4)


def variant_name(name: str, coco: bool) -> str:
    """Nome da pasta das partes: o candidato, ou o candidato com o sufixo da ablação.
    A ablação existe só para o vencedor (protocolo, 1.4)."""
    if not coco:
        return name
    if name != "S1":
        raise SystemExit("A ablação COCO existe só para o S1 (protocolo, 1.4).")
    return name + COCO_SUFFIX


def candidate(name: str):
    by_name = {c.name: c for c in CANDIDATES}
    if name not in by_name:
        raise SystemExit("Candidatos: " + ", ".join(by_name))
    return by_name[name]


def load_shapes(work_dir: Path) -> tuple[dict, str]:
    """Formas das âncoras gravadas na T5b.3 e a impressão digital do arquivo."""
    path = work_dir / ANCHORS_NAME
    return json.loads(path.read_text(encoding="utf-8")), file_sha256(path)


def part_path(work_dir: Path, name: str, lot: str, seed: int) -> Path:
    return work_dir / OUTPUT_FOLDER / name / f"{lot}_s{seed}.npz"


def save_part(path: Path, codes: list[str], preds: list[dict], meta: dict,
              state_dict: dict | None = None) -> None:
    """Gravação atômica: temporário na mesma pasta e os.replace. Os pesos, se dados, são
    gravados antes das predições, com o mesmo nome e extensão .pt (revisão R3): uma
    correção de inferência não exige retreinar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if state_dict is not None:
        weights = path.with_suffix(".pt")
        tmp_w = weights.with_name(weights.name + ".tmp")
        torch.save(state_dict, tmp_w)
        os.replace(tmp_w, weights)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as file:
        np.savez(file, meta=np.array(json.dumps(meta)), **flatten_predictions(codes, preds))
    os.replace(tmp, path)


def _folds(photos: list[DetectorPhoto]):
    lots = sorted({p.lot for p in photos if p.subset == "desenvolvimento"})
    for lot in lots:
        train, held = fold_indices(photos, lot)
        fit, internal = internal_split(photos, train)
        yield lot, [photos[i] for i in fit], [photos[i] for i in internal], [photos[i] for i in held]


def run_timing(device: str) -> None:
    """Uma época de cada candidato na primeira dobra e a projeção do custo total."""
    work_dir = get_work_dir()
    shapes, _ = load_shapes(work_dir)
    lot, fit, internal, held = next(_folds(load_detector_photos()))
    cfg = DetectorTrainConfig()
    for cand in CANDIDATES:
        train_s, internal_s = time_one_epoch(cand, shapes[cand.name]["formas"], fit, internal, device)
        per_run = cfg.max_epochs * (train_s + internal_s) + internal_s * len(held) / len(internal)
        print(f"{cand.name}: época {train_s:.0f} s, predição interna {internal_s:.0f} s; "
              f"projeção de {len(SEEDS) * 3} execuções: {len(SEEDS) * 3 * per_run / 3600:.1f} h",
              flush=True)
        torch.cuda.empty_cache()


def run_training(name: str, device: str, coco: bool = False) -> None:
    cand = candidate(name)
    folder = variant_name(name, coco)
    work_dir = get_work_dir()
    shapes, shapes_sha = load_shapes(work_dir)
    photos = load_detector_photos()
    cfg = DetectorTrainConfig()
    weights_meta = {"pesos_iniciais": "imagenet"}
    if coco:  # registro do que a ablação carregou (revisão R4, A-a)
        from training.detector_coco import COCO_WEIGHTS, coco_state, prefix_keys
        keys = prefix_keys(cand.build(shapes[name]["formas"]).state_dict(), coco_state())
        weights_meta = {"pesos_iniciais": "imagenet_e_prefixo_coco", "coco_tensores": len(keys),
                        "coco_primeira_fora": next(
                            k for k in cand.build(shapes[name]["formas"]).state_dict()
                            if k.startswith("backbone.features.") and k not in keys),
                        "coco_pesos": COCO_WEIGHTS.url}
    for seed in SEEDS:
        for lot, fit, internal, held in _folds(photos):
            tag = f"[{folder} {lot} s{seed}]"
            path = part_path(work_dir, folder, lot, seed)
            if path.exists():
                print(f"{tag} já gravado, pulando", flush=True)
                continue
            model, best_epoch, history = train_one(
                cand, shapes[name]["formas"], fit, internal, seed, device, cfg,
                log=lambda m, tag=tag: print(f"{tag} {m}", flush=True), coco_prefix=coco)
            preds = predict(model, make_loader(DetectorDataset(held, False), cfg, device, False),
                            device)
            meta = {"candidato": folder, "lote_avaliado": lot, "semente": seed,
                    "melhor_epoca": best_epoch, "historico": history, "treino": asdict(cfg),
                    "sha_caixas": BOXES_SHA256, "sha_formas": shapes_sha,
                    "corte_confianca": SCORE_THRESH, **weights_meta}
            save_part(path, [p.pseudonym for p in held], preds, meta,
                      {k: v.cpu() for k, v in model.state_dict().items()})
            print(f"{tag} gravado; melhor época {best_epoch}", flush=True)
            del model
            torch.cuda.empty_cache()

FINAL_FOLDER = Path("modelos") / "detector"
FINAL_SEED = 0  # revisão R2: o produto usa o modelo final da semente 0


def median_stop(work_dir: Path, folder: str) -> tuple[int, list[int]]:
    """Época de parada do modelo final (revisão R4): a mediana das épocas escolhidas nas
    9 partes (3 lotes × 3 sementes) do candidato. Devolve (mediana, épocas lidas)."""
    paths = sorted((work_dir / OUTPUT_FOLDER / folder).glob("*.npz"))
    if len(paths) != len(SEEDS) * 3:
        raise SystemExit(f"{folder}: são precisas 9 partes gravadas, há {len(paths)}.")
    epochs = []
    for path in paths:
        with np.load(path) as data:
            epochs.append(int(json.loads(str(data["meta"]))["melhor_epoca"]))
    return int(np.median(epochs)), epochs  # 9 valores: a mediana é um deles


def final_path(work_dir: Path, folder: str) -> Path:
    return work_dir / FINAL_FOLDER / f"{folder}_final_s{FINAL_SEED}.pt"


def run_final_model(name: str, device: str, coco: bool = False) -> None:
    """Treina e grava o modelo final do candidato com as fotos dos 3 lotes de
    desenvolvimento (nenhuma foto do teste). Nunca sobrescreve um modelo gravado."""
    from training.evaluation import _product_commit
    from training.run_final import save_final

    cand = candidate(name)
    folder = variant_name(name, coco)
    work_dir = get_work_dir()
    path = final_path(work_dir, folder)
    if path.exists():
        print(f"[{folder} final] já gravado, pulando", flush=True)
        return
    shapes, shapes_sha = load_shapes(work_dir)
    stop, epochs = median_stop(work_dir, folder)
    photos = [p for p in load_detector_photos() if p.subset == "desenvolvimento"]
    cfg = DetectorTrainConfig()
    print(f"[{folder} final] {len(photos)} fotos; épocas escolhidas {sorted(epochs)}; "
          f"parada na época {stop}", flush=True)
    model = train_fixed(cand, shapes[name]["formas"], photos, stop, FINAL_SEED, device, cfg,
                        log=lambda m: print(f"[{folder} final] {m}", flush=True),
                        coco_prefix=coco)
    meta = {"candidato": folder, "semente": FINAL_SEED, "epoca_parada": stop,
            "epocas_escolhidas": epochs, "fotos": len(photos),
            "lotes": sorted({p.lot for p in photos}), "treino": asdict(cfg),
            "sha_caixas": BOXES_SHA256, "sha_formas": shapes_sha,
            "corte_confianca": SCORE_THRESH,
            "pesos_iniciais": "imagenet_e_prefixo_coco" if coco else "imagenet",
            "commit_produto": _product_commit()}
    digest = save_final(path, model, meta)
    print(f"[{folder} final] gravado: SHA-256 {digest[:12]}…", flush=True)

if __name__ == "__main__":  # obrigatório no Windows: os processos auxiliares reimportam o módulo
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = sys.argv[1:]
    coco = len(args) == 3 and args[2] == "coco"
    if args == ["tempo"]:
        run_timing(device)
    elif args[:1] == ["treinar"] and (len(args) == 2 or coco):
        run_training(args[1], device, coco)
    elif args[:1] == ["final"] and (len(args) == 2 or coco):
        run_final_model(args[1], device, coco)
    else:
        raise SystemExit("Uso: python -m training.run_detector "
                         "{tempo | treinar S1|F1|F2 | treinar S1 coco | final S1 | final S1 coco}")
    print("Concluído.")
