"""Geometria das caixas anotadas, só no desenvolvimento (ET5b, T5b.1).

Mede, por classe, a largura, a altura e a proporção das caixas em pixels da
foto original, o tamanho das fotos, o número de caixas por foto e o IoU da
melhor âncora de cada configuração-padrão dos candidatos do torchvision. Serve
para fixar no protocolo a resolução de entrada e as âncoras antes de qualquer
treino. Sem pesos amostrais: as âncoras precisam cobrir os extremos, e a
amostra enriquecida (condições distantes inteiras) mostra justamente os menores
campos. O teste 03/07 fica de fora. Saídas: só contagens e medidas.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
import torch
from torchvision.models.detection import (fasterrcnn_mobilenet_v3_large_320_fpn,
                                          fasterrcnn_mobilenet_v3_large_fpn,
                                          ssdlite320_mobilenet_v3_large)
from torchvision.ops import box_iou

from training.box_labels import BOX_CLASSES, Box

QUANTILES = (0.0, 0.05, 0.5, 0.95, 1.0)
QUANTILE_NAMES = ("min", "p5", "p50", "p95", "max")
MEASURES = ("largura", "altura", "proporcao")
FRAME_COLUMNS = ("pseudonimo", "classe", "x0", "y0", "x1", "y1", "largura", "altura",
                 "foto_largura", "foto_altura")
PHOTO_SIZE = (360, 480)  # (largura, altura) de todas as fotos do desenvolvimento (T5b.1)
TINY_SIDE = 4.0          # lado abaixo do qual a caixa é suspeita de arraste acidental

# Configurações-padrão do torchvision: os dois candidatos da E07 e, só como
# referência, a variante de 320 px do Faster R-CNN
CANDIDATES = {
    "ssdlite320": ssdlite320_mobilenet_v3_large,
    "frcnn_fpn": fasterrcnn_mobilenet_v3_large_fpn,
    "frcnn_320_fpn": fasterrcnn_mobilenet_v3_large_320_fpn,
}


def box_frame(boxes: list[Box], sizes: dict[str, tuple[int, int]]) -> pd.DataFrame:
    """Uma linha por caixa: classe, coordenadas, largura, altura e proporção
    (largura ÷ altura), em pixels da foto original, e o tamanho da foto."""
    rows = [{"pseudonimo": b.pseudonym, "classe": b.cls,
             "x0": b.x0, "y0": b.y0, "x1": b.x1, "y1": b.y1,
             "largura": b.x1 - b.x0, "altura": b.y1 - b.y0,
             "foto_largura": sizes[b.pseudonym][0], "foto_altura": sizes[b.pseudonym][1]}
            for b in boxes]
    frame = pd.DataFrame(rows, columns=list(FRAME_COLUMNS))
    frame["proporcao"] = frame["largura"] / frame["altura"]
    return frame


def geometry_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Por classe e medida: número de caixas e os quantis (mínimo, p5, mediana, p95, máximo)."""
    rows = []
    for cls in BOX_CLASSES:
        part = frame[frame["classe"] == cls]
        for measure in MEASURES:
            values = part[measure].to_numpy(dtype=float)
            # Classe sem caixas aparece na tabela, com os quantis vazios
            qs = np.quantile(values, QUANTILES) if len(values) else [np.nan] * len(QUANTILES)
            row = {"classe": cls, "medida": measure, "caixas": len(values)}
            row.update({name: round(float(q), 2) for name, q in zip(QUANTILE_NAMES, qs)})
            rows.append(row)
    return pd.DataFrame(rows).set_index(["classe", "medida"])


def photo_size_counts(sizes: dict[str, tuple[int, int]]) -> pd.Series:
    """Fotos por tamanho (largura × altura): a orientação muda o redimensionamento fixo."""
    return pd.Series([f"{w} x {h}" for w, h in sizes.values()]).value_counts().rename("fotos")


def boxes_per_photo(codes: Iterable[str], frame: pd.DataFrame) -> pd.Series:
    """Fotos por número de caixas (0, 1, 2, 3 ou mais), contando as anotadas sem caixas."""
    counts = frame["pseudonimo"].value_counts().reindex(sorted(codes), fill_value=0)
    labels = counts.clip(upper=3).map(lambda n: "3 ou mais" if n == 3 else str(n))
    order = ["0", "1", "2", "3 ou mais"]
    return labels.value_counts().reindex(order, fill_value=0).rename("fotos")


def tiny_boxes(frame: pd.DataFrame, side: float = TINY_SIDE) -> pd.Series:
    """Caixas com algum lado menor que `side` px, por classe: possíveis arrastes acidentais."""
    small = frame[frame[["largura", "altura"]].min(axis=1) < side]
    return small["classe"].value_counts().reindex(list(BOX_CLASSES), fill_value=0).rename("caixas")


def default_anchors(name: str, photo_size: tuple[int, int] = PHOTO_SIZE
                    ) -> tuple[torch.Tensor, tuple[int, int]]:
    """Âncoras da configuração-padrão de um candidato, em pixels da entrada da rede,
    e o tamanho dessa entrada (altura, largura). Pesos aleatórios: só a geometria importa."""
    model = CANDIDATES[name](weights=None, weights_backbone=None,
                             num_classes=len(BOX_CLASSES) + 1).eval()
    width, height = photo_size
    with torch.no_grad():
        images, _ = model.transform([torch.zeros(3, height, width)])
        features = model.backbone(images.tensors)
        features = list(features.values()) if isinstance(features, dict) else [features]
        # No SSD, o gerador de âncoras fica no modelo; no Faster R-CNN, na RPN
        generator = (model.anchor_generator if hasattr(model, "anchor_generator")
                     else model.rpn.anchor_generator)
        anchors = generator(images, features)[0]
    return anchors, tuple(images.image_sizes[0])


def to_input(frame: pd.DataFrame, input_hw: tuple[int, int],
             photo_size: tuple[int, int] = PHOTO_SIZE) -> torch.Tensor:
    """Caixas da foto original levadas à entrada da rede. A escala é separada em x e
    em y, porque o SSDlite achata a foto num quadrado."""
    sx, sy = input_hw[1] / photo_size[0], input_hw[0] / photo_size[1]
    scale = np.array([sx, sy, sx, sy], dtype=np.float32)
    return torch.from_numpy(frame[["x0", "y0", "x1", "y1"]].to_numpy(dtype=np.float32) * scale)


def best_iou(boxes: torch.Tensor, anchors: torch.Tensor, chunk: int = 256) -> np.ndarray:
    """IoU da âncora que melhor encaixa em cada caixa, em blocos para limitar a memória."""
    parts = [box_iou(boxes[i:i + chunk], anchors).max(dim=1).values
             for i in range(0, len(boxes), chunk)]
    return torch.cat(parts).numpy() if parts else np.zeros(0)


def anchor_summary(frame: pd.DataFrame, best: np.ndarray) -> pd.DataFrame:
    """Por classe: caixas, p5 e mediana do IoU da melhor âncora e a fração com IoU >= 0,5."""
    part = frame.assign(iou=best)
    rows = []
    for cls in BOX_CLASSES:
        values = part.loc[part["classe"] == cls, "iou"].to_numpy(dtype=float)
        row = {"classe": cls, "caixas": len(values), "iou_p5": np.nan,
               "iou_p50": np.nan, "frac_iou_0_5": np.nan}
        if len(values):
            row.update(iou_p5=round(float(np.quantile(values, 0.05)), 3),
                       iou_p50=round(float(np.median(values)), 3),
                       frac_iou_0_5=round(float((values >= 0.5).mean()), 3))
        rows.append(row)
    return pd.DataFrame(rows).set_index("classe")


def main() -> None:
    # Importações que leem os dados só aqui: os testes usam as funções acima sem dados reais
    from PIL import Image

    from meter_reader.config import get_work_dir
    from training.box_labels import read_labels
    from training.box_report import SAMPLE_NAME, SAMPLE_SHA256
    from training.build_manifest import file_sha256
    from training.group_dataset import LABEL_VERSIONS, load_group_photos

    work_dir = get_work_dir()
    if file_sha256(work_dir / SAMPLE_NAME) != SAMPLE_SHA256:
        raise SystemExit("A amostra difere da gravada na T5.2.")
    sample = pd.read_csv(work_dir / SAMPLE_NAME)
    dev = set(sample.loc[sample["conjunto"] == "desenvolvimento", "pseudonimo"])

    # Os limites das caixas já foram conferidos na T5.4; aqui só se filtra o desenvolvimento
    _, boxes = read_labels(work_dir, sample=set(sample["pseudonimo"]))
    boxes = [b for b in boxes if b.pseudonym in dev]
    paths = {p.pseudonym: p.path for p in load_group_photos(LABEL_VERSIONS["r4"])
             if p.pseudonym in dev}
    if len(paths) != len(dev):
        raise SystemExit(f"{len(dev) - len(paths)} foto(s) da amostra sem caminho no manifesto.")
    sizes = {}
    for code, path in paths.items():
        with Image.open(path) as image:  # só o cabeçalho: o tamanho, sem decodificar
            sizes[code] = image.size
    if set(sizes.values()) != {PHOTO_SIZE}:
        raise SystemExit("Há fotos com tamanho diferente de 360 x 480: reveja a conversão.")

    frame = box_frame(boxes, sizes)
    pd.set_option("display.width", 200, "display.max_rows", 200)
    print(f"Desenvolvimento: {len(dev)} fotos, {len(frame)} caixas\n")
    print(photo_size_counts(sizes), end="\n\n")
    print(boxes_per_photo(dev, frame), end="\n\n")
    print(geometry_summary(frame), end="\n\n")
    print(f"Caixas com algum lado menor que {TINY_SIDE:.0f} px:")
    print(tiny_boxes(frame), end="\n\n")
    print(f"Caixas com proporção abaixo de 1 (mais altas que largas): "
          f"{int((frame['proporcao'] < 1).sum())}\n")
    for name in CANDIDATES:
        anchors, (h, w) = default_anchors(name)
        best = best_iou(to_input(frame, (h, w)), anchors)
        print(f"{name}: entrada {w} x {h} (largura x altura), {len(anchors)} âncoras")
        print(anchor_summary(frame, best), end="\n\n")


if __name__ == "__main__":
    main()
