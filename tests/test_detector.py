"""Testes do detector do produto (src/meter_reader/detector.py)."""
import pytest
import torch
from torchvision.models.detection.image_list import ImageList

from meter_reader import detector as det
from training.box_labels import BOX_CLASSES

SSD_SHAPES = [[(8.0 * (i + 1), 4.0 * (i + 1)), (6.0 * (i + 1), 6.0 * (i + 1))] for i in range(6)]

def test_detector_classes_are_the_annotation_classes():
    assert det.DETECTOR_CLASSES == BOX_CLASSES and det.NUM_CLASSES == 5


def test_generator_centers_shapes_on_each_cell_in_torchvision_order():
    gen = det.ShapeAnchorGenerator([[(10.0, 4.0), (6.0, 6.0)]])
    images = ImageList(torch.zeros(1, 3, 32, 64), [(32, 64)])
    anchors = gen(images, [torch.zeros(1, 8, 2, 4)])[0]  # passo de 16 px
    assert anchors.shape == (2 * 4 * 2, 4)
    # Primeira célula (canto superior esquerdo), primeira e segunda forma
    assert anchors[0].tolist() == [3.0, 6.0, 13.0, 10.0]
    assert anchors[1].tolist() == [5.0, 5.0, 11.0, 11.0]
    # A posição muda antes da forma: a terceira âncora é a segunda célula na horizontal
    assert anchors[2].tolist() == [19.0, 6.0, 29.0, 10.0]
    assert gen.num_anchors_per_location() == [2]


def test_generator_rejects_empty_level_and_wrong_level_count():
    with pytest.raises(ValueError):
        det.ShapeAnchorGenerator([[(1.0, 1.0)], []])
    gen = det.ShapeAnchorGenerator([[(1.0, 1.0)]])
    with pytest.raises(ValueError):
        gen(ImageList(torch.zeros(1, 3, 8, 8), [(8, 8)]), [torch.zeros(1, 1, 2, 2)] * 2)


@pytest.mark.parametrize("build", [
    lambda: det.build_ssd(SSD_SHAPES),
])
def test_models_train_and_predict_on_a_portrait_photo(build):
    torch.manual_seed(0)
    model = build()
    # Lote de 2: o último nível do SSDlite tem 1 × 1 posição, e a normalização em
    # lote não treina com uma foto só (o treino descarta o lote final incompleto)
    images = [torch.rand(3, 480, 360) for _ in range(2)]
    target = {"boxes": torch.tensor([[50.0, 60.0, 150.0, 90.0]]), "labels": torch.tensor([4])}
    model.train()
    losses = model(images, [target, target])
    assert all(torch.isfinite(v) for v in losses.values())
    model.eval()
    with torch.no_grad():
        out = model(images[:1])[0]
    assert set(out) == {"boxes", "labels", "scores"}
    assert out["labels"].numel() == 0 or int(out["labels"].min()) >= 1


def test_ssd_keeps_the_native_size_without_squashing():
    model = det.build_ssd(SSD_SHAPES).eval()
    images, _ = model.transform([torch.rand(3, 480, 360)])
    assert tuple(images.image_sizes[0]) == (480, 360)

def test_product_detector_has_no_faster_rcnn():
    # E82 e opção 3a da ET5c: o Faster R-CNN não entra no pacote do produto
    assert not hasattr(det, "build_frcnn") and not hasattr(det, "LOW_LATENCY_RPN")
