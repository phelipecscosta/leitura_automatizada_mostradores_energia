"""Conjunto de dados do detector: alvos, dobras, aumentação com as caixas e pesos (ET5b, T5b.4).

Protocolo: exploracao/detector_README.md (pesquisa), seções 1.1, 1.2 e 1.5.
Uma única conversão das caixas anotadas (training.box_labels) para os alvos do
torchvision: caixas (x0, y0, x1, y1) em pixels da foto original e rótulos de 1 a
4 na ordem de meter_reader.detector.DETECTOR_CLASSES (0 é o fundo). As mensagens
mostram só contagens, nunca nomes nem caminhos.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms.functional import to_tensor

from meter_reader.detector import DETECTOR_CLASSES
from meter_reader.preprocess import enhance
from training.box_labels import Box
from training.detector_eval import MIN_FIELD_SIDE
from training.finetune import epoch_seed

# Impressões digitais do gabarito vigente (protocolo, 1.1): as caixas depois da
# revisita das fotos novas do teste ampliado (ET5b, D6 b)
BOXES_SHA256 = "0c79825ba04b56dce92213977579dbb9c64eb3a84ee72fa138f6b2f70e53b451"
PHOTOS_SHA256 = "d1e73df99d4bf3e52f580999014961f006cf482408d124a58f1c867c2a50b8be"

INTERNAL_FRACTION = 0.10  # divisão interna para escolher a época (protocolo, 1.2)
INTERNAL_SEED = 0         # a mesma para todos os candidatos
AUG_JITTER = 0.10         # brilho, contraste, translação e escala de ±10% (protocolo, 1.5)
MIN_BOX_SIDE = 1.0        # caixa com algum lado abaixo disso, depois da aumentação, sai do alvo


@dataclass(frozen=True)
class DetectorPhoto:
    """Uma foto da amostra. Nunca exibir `path` em saídas versionadas."""

    pseudonym: str
    subset: str          # "desenvolvimento" ou "teste"
    lot: str
    weight: float        # peso amostral da amostra v2
    group: str           # rótulo de grupo vigente (R4 no desenvolvimento, R5 no teste)
    boxes: tuple[tuple[float, float, float, float], ...]
    labels: tuple[int, ...]  # 1 a 4, na ordem de DETECTOR_CLASSES
    path: Path | None = field(default=None, compare=False)

    @property
    def has_field(self) -> bool:
        """Foto com campo (protocolo, 1.1; D3 a ii): alguma caixa com os dois lados >= 4 px."""
        return any(min(x1 - x0, y1 - y0) >= MIN_FIELD_SIDE for x0, y0, x1, y1 in self.boxes)

    def target(self) -> dict[str, torch.Tensor]:
        """Alvo do torchvision; foto sem caixas tem tensores vazios (o fundo inteiro)."""
        return {"boxes": torch.tensor(self.boxes, dtype=torch.float32).reshape(-1, 4),
                "labels": torch.tensor(self.labels, dtype=torch.int64)}


def label_of(cls: str) -> int:
    """Classe de caixa -> rótulo do detector (1 a 4)."""
    return DETECTOR_CLASSES.index(cls) + 1


def build_photos(sample_rows: list[dict], boxes: list[Box], groups: dict[str, str],
                 paths: dict[str, Path] | None = None) -> list[DetectorPhoto]:
    """Junta a amostra (linhas do CSV), as caixas e os grupos vigentes, na ordem da amostra.

    `groups` e `paths`: pseudônimo -> grupo e caminho. Falha se faltar grupo ou
    caminho para alguma foto, ou se houver caixa de foto fora da amostra.
    """
    codes = [r["pseudonimo"] for r in sample_rows]
    by_code: dict[str, list[Box]] = {c: [] for c in codes}
    stray = 0
    for b in boxes:
        if b.pseudonym in by_code:
            by_code[b.pseudonym].append(b)
        else:
            stray += 1
    missing_group = sum(1 for c in codes if c not in groups)
    missing_path = 0 if paths is None else sum(1 for c in codes if c not in paths)
    if stray or missing_group or missing_path:
        raise ValueError(f"Divergências: {stray} caixa(s) fora da amostra, {missing_group} foto(s) "
                         f"sem grupo, {missing_path} sem caminho.")
    photos = []
    for row in sample_rows:
        code = row["pseudonimo"]
        own = by_code[code]
        photos.append(DetectorPhoto(
            pseudonym=code, subset=row["conjunto"], lot=row["lote"], weight=float(row["peso"]),
            group=groups[code], boxes=tuple((b.x0, b.y0, b.x1, b.y1) for b in own),
            labels=tuple(label_of(b.cls) for b in own),
            path=None if paths is None else paths[code]))
    return photos


def fold_indices(photos: list[DetectorPhoto], held_out_lot: str) -> tuple[np.ndarray, np.ndarray]:
    """(treino, avaliação) de uma dobra, só no desenvolvimento (protocolo, 1.2)."""
    dev = np.array([p.subset == "desenvolvimento" for p in photos])
    lots = np.array([p.lot for p in photos])
    if held_out_lot not in set(lots[dev]):
        raise ValueError("Lote deixado de fora inexistente no desenvolvimento.")
    return np.flatnonzero(dev & (lots != held_out_lot)), np.flatnonzero(dev & (lots == held_out_lot))


def internal_split(photos: list[DetectorPhoto], train: np.ndarray,
                   fraction: float = INTERNAL_FRACTION,
                   seed: int = INTERNAL_SEED) -> tuple[np.ndarray, np.ndarray]:
    """(ajuste, interna) dentro do treino de uma dobra, por sorteio com semente fixa.

    Independe da ordem de `train`: o sorteio é feito sobre os pseudônimos ordenados.
    """
    train = np.asarray(train)
    order = train[np.argsort([photos[i].pseudonym for i in train])]
    rng = np.random.default_rng(seed)
    internal = np.sort(rng.choice(order, size=round(fraction * len(order)), replace=False))
    return np.setdiff1d(train, internal), internal


def augment(image: torch.Tensor, target: dict, seed: int) -> tuple[torch.Tensor, dict]:
    """Brilho, contraste, translação e escala de ±10%, com as caixas transformadas junto.

    Sem espelhamento e sem rotação (protocolo, 1.5). Os cantos expostos recebem a
    cor média da foto. Caixas que saem quase inteiras da foto são descartadas.
    Determinística pela semente.
    """
    fill = [float(v) for v in image.mean(dim=(1, 2))]
    transform = v2.Compose([
        v2.ColorJitter(brightness=AUG_JITTER, contrast=AUG_JITTER),
        v2.RandomAffine(degrees=0, translate=(AUG_JITTER, AUG_JITTER),
                        scale=(1 - AUG_JITTER, 1 + AUG_JITTER), fill=fill),
        v2.SanitizeBoundingBoxes(min_size=MIN_BOX_SIDE),
    ])
    height, width = image.shape[-2:]
    boxes = tv_tensors.BoundingBoxes(target["boxes"], format="XYXY", canvas_size=(height, width))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        out_image, out = transform(tv_tensors.Image(image), {"boxes": boxes,
                                                             "labels": target["labels"]})
    return out_image.as_subclass(torch.Tensor), {"boxes": out["boxes"].as_subclass(torch.Tensor),
                                                 "labels": out["labels"]}


def load_image(path: Path) -> torch.Tensor:
    """Foto com o realce fixo da triagem, em [0, 1] (protocolo, 1.5; E19, E33)."""
    with Image.open(path) as image:
        return to_tensor(enhance(image))


class DetectorDataset(Dataset):
    """Fotos e alvos do detector; no treino, uma vista nova a cada época."""

    def __init__(self, photos: list[DetectorPhoto], train: bool, run_seed: int = 0) -> None:
        self.photos, self.train, self.run_seed = list(photos), train, run_seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.photos)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, dict]:
        photo = self.photos[i]
        image, target = load_image(photo.path), photo.target()
        if self.train:
            image, target = augment(image, target,
                                    epoch_seed(photo.pseudonym, self.run_seed, self.epoch))
        return image, target


def collate(batch: list[tuple[torch.Tensor, dict]]) -> tuple[list[torch.Tensor], list[dict]]:
    """Lista de imagens e lista de alvos: o formato de entrada dos detectores do torchvision."""
    images, targets = zip(*batch)
    return list(images), list(targets)


def load_detector_photos() -> list[DetectorPhoto]:
    """Amostra v2, caixas e grupos vigentes da pasta de trabalho, conferidos, com os caminhos."""
    import csv

    from meter_reader.config import get_work_dir
    from training.box_labels import BOXES_NAME, PHOTOS_NAME, read_labels
    from training.box_report import SAMPLE_NAME, SAMPLE_SHA256
    from training.build_manifest import file_sha256
    from training.group_dataset import LABEL_VERSIONS, TEST_LABEL_VERSIONS, load_group_photos

    work_dir = get_work_dir()
    for name, sha256 in ((SAMPLE_NAME, SAMPLE_SHA256), (BOXES_NAME, BOXES_SHA256),
                         (PHOTOS_NAME, PHOTOS_SHA256)):
        if file_sha256(work_dir / name) != sha256:
            raise SystemExit(f"{name}: impressão digital diferente da registrada.")
    with (work_dir / SAMPLE_NAME).open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    codes = {r["pseudonimo"] for r in rows}
    _, boxes = read_labels(work_dir, sample=codes)
    groups, paths = {}, {}
    for labels in (LABEL_VERSIONS["r4"], TEST_LABEL_VERSIONS["r5"]):
        for p in load_group_photos(labels):
            if p.pseudonym in codes:
                groups[p.pseudonym], paths[p.pseudonym] = p.group, p.path
    return build_photos(rows, boxes, groups, paths)


def main() -> None:
    """Confere a montagem nos dados reais: só contagens."""
    from collections import Counter

    photos = load_detector_photos()
    for subset in ("desenvolvimento", "teste"):
        part = [p for p in photos if p.subset == subset]
        classes = Counter(lab for p in part for lab in p.labels)
        print(f"{subset}: {len(part)} fotos, {sum(p.has_field for p in part)} com campo, "
              f"soma dos pesos {sum(p.weight for p in part):.1f}; caixas por classe: "
              + ", ".join(f"{c} {classes[label_of(c)]}" for c in DETECTOR_CLASSES))
    lots = sorted({p.lot for p in photos if p.subset == "desenvolvimento"})
    for lot in lots:
        train, held = fold_indices(photos, lot)
        fit, internal = internal_split(photos, train)
        print(f"Dobra sem {lot}: ajuste {len(fit)}, interna {len(internal)}, avaliada {len(held)}")
    image, target = DetectorDataset(photos, train=True)[0]
    print(f"Primeira foto aumentada: {tuple(image.shape)}, {len(target['boxes'])} caixa(s)")


if __name__ == "__main__":
    main()
