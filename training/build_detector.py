"""Artefato do detector do produto (ET5c, T5c.4; E32, E66, E72).

Junta num arquivo, para o meter_reader.detector.load_detector: os pesos do modelo
final do S1_coco (semente 0, revisões R2 e R4), as formas das âncoras, o limiar da
porta (relatório da ablação) e o corte dos campos (relatório dos campos), com a
procedência. Só tipos simples na procedência, para a carga com weights_only=True.

Uso, a partir da raiz do repositório do produto:
    python -m training.build_detector
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from meter_reader.detector import ARTIFACT_FORMAT, DETECTOR_CLASSES
from training.build_manifest import file_sha256

ARTIFACT_NAME = "detector_s1.pt"
FOLDER = "S1_coco"  # vencedor da ablação (E-ablação, T5c.2)


def build_artifact(final_weights: Path, shapes, gate: float, cut: float, info: dict) -> dict:
    """Conteúdo do artefato. Confere os pesos finais pelo SHA-256 do JSON ao lado e exige
    0 < limiar da porta <= corte dos campos < 1."""
    final_weights = Path(final_weights)
    record = json.loads(final_weights.with_suffix(".json").read_text(encoding="utf-8"))
    if file_sha256(final_weights) != record["sha256_pesos"]:
        raise SystemExit("Os pesos finais não correspondem ao SHA-256 gravado.")
    if not 0 < gate <= cut < 1:
        raise ValueError("É preciso 0 < limiar da porta <= corte dos campos < 1.")
    return {"formato": ARTIFACT_FORMAT, "classes": list(DETECTOR_CLASSES),
            "formas": [[[float(w), float(h)] for w, h in level] for level in shapes],
            "pesos": torch.load(final_weights, map_location="cpu", weights_only=True),
            "limiar_porta": float(gate), "corte_campos": float(cut),
            "info": {**info, "candidato": str(record["candidato"]),
                     "epoca_parada": int(record["epoca_parada"]),
                     "sha256_pesos_finais": str(record["sha256_pesos"]),
                     "commit_treino": str(record["commit_produto"]),
                     "sha_caixas": str(record["sha_caixas"]),
                     "sha_formas": str(record["sha_formas"])}}


def save_artifact(path: Path, data: dict) -> str:
    """Gravação atômica que nunca sobrescreve. Devolve o SHA-256 do arquivo."""
    path = Path(path)
    if path.exists():
        raise FileExistsError("Artefato do detector já gravado.")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(data, tmp)
    os.replace(tmp, path)
    return file_sha256(path)


def main() -> None:
    from meter_reader.config import get_work_dir
    from training.detector_ablation import SUMMARY_NAME as ABLATION_SUMMARY
    from training.detector_fields_eval import SUMMARY_NAME as FIELDS_SUMMARY
    from training.run_detector import FINAL_FOLDER, OUTPUT_FOLDER, final_path, load_shapes

    work_dir = get_work_dir()
    shapes, shapes_sha = load_shapes(work_dir)
    ablation_path = work_dir / OUTPUT_FOLDER / ABLATION_SUMMARY
    fields_path = work_dir / OUTPUT_FOLDER / FIELDS_SUMMARY
    decision = json.loads(ablation_path.read_text(encoding="utf-8"))["decisao"]
    if decision["vencedor"] != FOLDER:
        raise SystemExit("A ablação não escolheu o S1_coco: reveja antes de empacotar.")
    cut = json.loads(fields_path.read_text(encoding="utf-8"))["corte_adotado"]
    data = build_artifact(final_path(work_dir, FOLDER), shapes["S1"]["formas"],
                          decision["limiar_porta"], cut,
                          {"sha_relatorio_ablacao": file_sha256(ablation_path),
                           "sha_relatorio_campos": file_sha256(fields_path)})
    if data["info"]["sha_formas"] != shapes_sha:
        raise SystemExit("O modelo final foi treinado com outro arquivo de formas.")
    digest = save_artifact(work_dir / FINAL_FOLDER / ARTIFACT_NAME, data)
    print(f"Artefato gravado em modelos/detector/{ARTIFACT_NAME}, na pasta de trabalho.")
    print(f"Limiar da porta {data['limiar_porta']:.6f}; corte dos campos {data['corte_campos']:.2f}; "
          f"época de parada {data['info']['epoca_parada']}")
    print(f"SHA-256: {digest}")


if __name__ == "__main__":
    main()
