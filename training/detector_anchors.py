"""Âncoras dos candidatos do detector escolhidas pelos dados, e latência (ET5b, T5b.3).

C4 b (revisada antes do treino): agrupamento k-means das formas (largura,
altura) das caixas do desenvolvimento, com distância 1 - IoU entre formas
centradas, já na escala da entrada de cada candidato; as formas são repetidas
em todos os níveis do detector, como o torchvision faz no Faster R-CNN. O número
de formas por posição é o da configuração-padrão (6 no SSDlite, 15 no Faster
R-CNN), para que a cabeça tenha o mesmo tamanho. A primeira versão dividia as
formas por igual entre os níveis e cobriu menos que as âncoras-padrão: as
formas grandes caíam em grades de passo grosso. Meta: em cada classe, >= 90% das
caixas com alguma âncora de IoU >= 0,5; se não for alcançada, o valor alcançado
é registrado, sem mudar a regra.

Também mede a latência em CPU dos três candidatos sem pesos treinados (a
latência depende da arquitetura e da resolução): é a ordem de simplicidade da
regra de escolha (C3 a 1). O teste 03/07 fica de fora. Saídas: só medidas.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

import numpy as np
import torch

from meter_reader.detector import LOW_LATENCY_RPN, build_frcnn, build_ssd

KMEANS_SEED = 0
KMEANS_ITERS = 300
COVERAGE_TARGET = 0.90   # fração das caixas de cada classe com IoU >= 0,5 (C4 b)
ANCHORS_NAME = "detector_ancoras.json"
LATENCY_WARMUP, LATENCY_RUNS, LATENCY_THREADS = 20, 200, 4  # como na E67


@dataclass(frozen=True)
class Candidate:
    """Candidato do protocolo (C1 b, C4 a): níveis, formas por nível e entrada."""
    name: str
    profile: str
    levels: int
    per_level: int
    input_size: tuple[int, int]  # (largura, altura) da entrada para uma foto 360 × 480

    def build(self, shapes, pretrained_backbone: bool = False):
        if self.name == "S1":
            return build_ssd(shapes, self.input_size, pretrained_backbone)
        width, height = self.input_size
        rpn = LOW_LATENCY_RPN if self.profile == "CPU" else {}
        return build_frcnn(shapes, min(width, height), max(width, height), pretrained_backbone,
                           **rpn)


CANDIDATES = (
    Candidate("S1", "CPU", 6, 6, (360, 480)),    # SSDlite, resolução nativa
    Candidate("F1", "CPU", 5, 15, (360, 480)),   # Faster R-CNN (N8), resolução nativa
    Candidate("F2", "GPU", 5, 15, (600, 800)),   # Faster R-CNN (N8), ampliado
)


def shape_iou(wh: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """IoU entre formas centradas na mesma origem: (n, 2) × (k, 2) -> (n, k)."""
    inter = (np.minimum(wh[:, None, 0], centers[None, :, 0])
             * np.minimum(wh[:, None, 1], centers[None, :, 1]))
    area = wh[:, 0] * wh[:, 1]
    return inter / (area[:, None] + (centers[:, 0] * centers[:, 1])[None, :] - inter)


def kmeans_shapes(wh: np.ndarray, k: int, seed: int = KMEANS_SEED,
                  iters: int = KMEANS_ITERS) -> np.ndarray:
    """k centros (largura, altura) pela distância 1 - IoU, ordenados por área.

    Início pelos k-means++ na mesma distância; os centros são a mediana do grupo
    (robusta às caixas extremas). Grupo vazio mantém o centro anterior.
    """
    wh = np.asarray(wh, dtype=np.float64)
    if len(wh) < k:
        raise ValueError("Menos caixas que formas pedidas.")
    rng = np.random.default_rng(seed)
    centers = [wh[rng.integers(len(wh))]]
    for _ in range(k - 1):
        dist = 1 - shape_iou(wh, np.array(centers)).max(axis=1)
        centers.append(wh[rng.choice(len(wh), p=dist ** 2 / (dist ** 2).sum())])
    centers = np.array(centers)
    for _ in range(iters):
        assign = shape_iou(wh, centers).argmax(axis=1)
        new = np.array([np.median(wh[assign == j], axis=0) if (assign == j).any() else centers[j]
                        for j in range(k)])
        if np.allclose(new, centers):
            break
        centers = new
    return centers[np.argsort(centers[:, 0] * centers[:, 1])]


def replicate_levels(centers: np.ndarray, levels: int) -> list[list[tuple[float, float]]]:
    """As mesmas formas, arredondadas a 0,01 px, em todos os níveis (C4 b revisada)."""
    shapes = [(round(float(w), 2), round(float(h), 2)) for w, h in centers]
    return [list(shapes) for _ in range(levels)]


def model_anchors(model, input_size: tuple[int, int]) -> tuple[torch.Tensor, tuple[int, int]]:
    """Âncoras de um modelo montado para uma foto de `input_size` (largura, altura), e o
    tamanho (altura, largura) da entrada depois da transformação do modelo."""
    width, height = input_size
    model.eval()
    with torch.no_grad():
        images, _ = model.transform([torch.zeros(3, height, width)])
        features = model.backbone(images.tensors)
        features = list(features.values()) if isinstance(features, dict) else [features]
        generator = model.anchor_generator if hasattr(model, "anchor_generator") \
            else model.rpn.anchor_generator
        anchors = generator(images, features)[0]
    return anchors, tuple(images.image_sizes[0])


def latency_ms(model, input_size: tuple[int, int], warmup: int = LATENCY_WARMUP,
               runs: int = LATENCY_RUNS, threads: int = LATENCY_THREADS) -> tuple[float, float]:
    """(p50, p95) em ms de uma foto por vez, em CPU, com a saída completa do modelo
    (inclusive a supressão de não máximos). Entrada aleatória em [0, 1]."""
    torch.set_num_threads(threads)
    model = model.eval().to("cpu")
    width, height = input_size
    generator = torch.Generator().manual_seed(0)
    x = [torch.rand(3, height, width, generator=generator)]
    times = []
    with torch.no_grad():
        for i in range(warmup + runs):
            start = time.perf_counter()
            model(x)
            if i >= warmup:
                times.append(1000 * (time.perf_counter() - start))
    return float(np.percentile(times, 50)), float(np.percentile(times, 95))


def main() -> None:
    # Importações que leem os dados só aqui: os testes usam as funções acima sem dados reais
    import pandas as pd

    from meter_reader.config import get_work_dir
    from training.box_labels import read_labels
    from training.box_report import SAMPLE_NAME, SAMPLE_SHA256
    from training.box_stats import PHOTO_SIZE, anchor_summary, best_iou, box_frame, to_input
    from training.build_manifest import file_sha256

    work_dir = get_work_dir()
    if file_sha256(work_dir / SAMPLE_NAME) != SAMPLE_SHA256:
        raise SystemExit("A amostra difere da registrada.")
    sample = pd.read_csv(work_dir / SAMPLE_NAME)
    dev = set(sample.loc[sample["conjunto"] == "desenvolvimento", "pseudonimo"])
    _, boxes = read_labels(work_dir, sample=set(sample["pseudonimo"]))
    frame = box_frame([b for b in boxes if b.pseudonym in dev], {c: PHOTO_SIZE for c in dev})

    pd.set_option("display.width", 200)
    record = {}
    for cand in CANDIDATES:
        scaled = to_input(frame, (cand.input_size[1], cand.input_size[0])).numpy()
        wh = np.stack([scaled[:, 2] - scaled[:, 0], scaled[:, 3] - scaled[:, 1]], axis=1)
        shapes = replicate_levels(kmeans_shapes(wh, cand.per_level), cand.levels)
        model = cand.build(shapes)
        anchors, (h, w) = model_anchors(model, cand.input_size)
        best = best_iou(to_input(frame, (h, w)), anchors)
        summary = anchor_summary(frame, best)
        p50, p95 = latency_ms(model, cand.input_size)
        reached = bool((summary["frac_iou_0_5"] >= COVERAGE_TARGET).all())
        print(f"{cand.name} (perfil {cand.profile}): entrada {w} x {h}, {len(anchors)} âncoras; "
              f"meta de {COVERAGE_TARGET:.0%} {'alcançada' if reached else 'NÃO alcançada'}; "
              f"latência p50 {p50:.1f} ms, p95 {p95:.1f} ms")
        print(summary, end="\n\n")
        record[cand.name] = {"formas": shapes, "entrada": [w, h], "latencia_p50_ms": round(p50, 1),
                             "latencia_p95_ms": round(p95, 1), "meta_alcancada": reached}
    (work_dir / ANCHORS_NAME).write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(f"Formas gravadas em {ANCHORS_NAME}, na pasta de trabalho.")


if __name__ == "__main__":
    main()

