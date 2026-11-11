"""Testes do formato das caixas anotadas (T5.3), com dados sintéticos."""
import pytest

from training import box_labels as bl
from training.box_labels import Box, BoxLabelError, PhotoRecord

A, B, C = "F-" + "a" * 12, "F-" + "b" * 12, "F-" + "c" * 12
WHEN = "2026-11-11 12:00:00"


def _photos():
    # A com duas caixas; B anotada sem caixas (sem mostrador)
    return [PhotoRecord(A, 2, 3.25, WHEN), PhotoRecord(B, 0, 1.0, WHEN)]


def _boxes():
    return [Box(A, "display_digital", 10, 20, 110.04, 60), Box(A, "numero_serie", 5, 5, 9, 9)]


def test_round_trip_keeps_order_and_zero_box_photos(tmp_path):
    bl.write_labels(tmp_path, _photos(), _boxes())
    photos, boxes = bl.read_labels(tmp_path, sample={A, B}, sizes={A: (480, 360), B: (480, 360)})
    assert [p.pseudonym for p in photos] == [A, B]
    assert photos[1].n_boxes == 0
    assert [b.cls for b in boxes] == ["display_digital", "numero_serie"]
    assert boxes[0].x1 == 110.0  # arredondado a 0,1 px


def test_no_files_means_nothing_annotated(tmp_path):
    assert bl.read_labels(tmp_path) == ([], [])


def test_only_one_file_is_refused(tmp_path):
    bl.write_labels(tmp_path, _photos(), _boxes())
    (tmp_path / bl.PHOTOS_NAME).unlink()
    with pytest.raises(BoxLabelError):
        bl.read_labels(tmp_path)


def test_rewrite_replaces_previous_content(tmp_path):
    bl.write_labels(tmp_path, _photos(), _boxes())
    bl.write_labels(tmp_path, [PhotoRecord(B, 0, 1.0, WHEN)], [])
    photos, boxes = bl.read_labels(tmp_path)
    assert [p.pseudonym for p in photos] == [B] and boxes == []
    assert not list(tmp_path.glob("*.tmp"))  # nenhum temporário deixado para trás


@pytest.mark.parametrize("box", [
    Box(A, "placa", 1, 1, 5, 5),               # classe fora do esquema
    Box(A, "numero_serie", 5, 1, 5, 9),        # sem largura
    Box(A, "numero_serie", -1, 1, 5, 9),       # coordenada negativa
    Box(A, "numero_serie", 1.01, 1, 1.04, 9),  # perde a área no arredondamento
])
def test_write_refuses_invalid_boxes(tmp_path, box):
    with pytest.raises(BoxLabelError):
        bl.write_labels(tmp_path, [PhotoRecord(A, 1, 1.0, WHEN)], [box])
    assert not (tmp_path / bl.BOXES_NAME).exists()  # nada gravado


def test_write_refuses_inconsistent_counts_and_repeats(tmp_path):
    with pytest.raises(BoxLabelError):  # registro diz 1 caixa, há 2
        bl.write_labels(tmp_path, [PhotoRecord(A, 1, 1.0, WHEN)], _boxes())
    with pytest.raises(BoxLabelError):  # caixa de foto sem registro
        bl.write_labels(tmp_path, [PhotoRecord(B, 0, 1.0, WHEN)], _boxes())
    with pytest.raises(BoxLabelError):  # foto repetida
        bl.write_labels(tmp_path, [PhotoRecord(B, 0, 1.0, WHEN)] * 2, [])


def test_read_checks_photo_limits_and_sample(tmp_path):
    bl.write_labels(tmp_path, _photos(), _boxes())
    with pytest.raises(BoxLabelError):  # caixa além da largura da foto
        bl.read_labels(tmp_path, sizes={A: (100, 360), B: (480, 360)})
    with pytest.raises(BoxLabelError):  # foto fora da amostra
        bl.read_labels(tmp_path, sample={A})


def _edit(path, old, new):
    path.write_text(path.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")


@pytest.mark.parametrize("name, old, new", [
    (bl.BOXES_NAME, "110.0", "inf"),                 # número não finito
    (bl.BOXES_NAME, "display_digital", "placa"),     # classe fora do esquema
    (bl.BOXES_NAME, "pseudonimo,classe", "pseudonimo,tipo"),  # cabeçalho alterado
    (bl.PHOTOS_NAME, ",2,", ",3,"),                  # contagem diferente das linhas
    (bl.PHOTOS_NAME, "3.2", "-3.2"),                 # tempo negativo
    (bl.PHOTOS_NAME, WHEN, "11/11/2026"),            # data e hora fora do formato
    (bl.PHOTOS_NAME, B, A),                          # foto repetida
])
def test_read_refuses_edited_files(tmp_path, name, old, new):
    bl.write_labels(tmp_path, _photos(), _boxes())
    _edit(tmp_path / name, old, new)
    with pytest.raises(BoxLabelError):
        bl.read_labels(tmp_path)


def test_error_messages_do_not_show_values(tmp_path):
    bl.write_labels(tmp_path, _photos(), _boxes())
    _edit(tmp_path / bl.BOXES_NAME, "110.0", "abc")
    with pytest.raises(BoxLabelError) as error:
        bl.read_labels(tmp_path)
    assert "abc" not in str(error.value) and A not in str(error.value)
    assert "linha 2" in str(error.value)
