import pytest
import torch

from meter_reader.reader import Reader
from meter_reader.reader_text import FUNCTION_ALPHABET, TEXT_ALPHABET, encode
from training.reader_loss import reader_loss


@pytest.mark.parametrize("candidate", ["L1", "L2-9", "L2-12"])
def test_formas(candidate):
    model = Reader(candidate, pretrained=False).eval()
    out = model(torch.zeros(2, 3, 48, 192))
    assert out["texto"].shape == (2, 48, len(TEXT_ALPHABET) + 1)   # 192 / 4 passos
    assert out["funcao"].shape == (2, 48, len(FUNCTION_ALPHABET) + 1)
    assert out["duas"].shape == (2,)


def _batch():
    texts, functions = ["0012345", "AB12", ""], ["03", "", ""]
    return {
        "texto": torch.tensor([k for s in texts for k in encode(s, TEXT_ALPHABET)]),
        "texto_comprimento": torch.tensor([len(s) for s in texts]),
        "funcao": torch.tensor([k for s in functions for k in encode(s, FUNCTION_ALPHABET)]),
        "funcao_comprimento": torch.tensor([len(s) for s in functions]),
        "usa_texto": torch.tensor([True, True, False]),
        "usa_funcao": torch.tensor([True, False, False]),
        "usa_duas": torch.tensor([True, False, True]),
        "duas": torch.tensor([0.0, 0.0, 1.0]),
    }


def test_perda_finita_e_mascaras():
    model = Reader("L1")
    total, parts = reader_loss(model(torch.randn(3, 3, 48, 192)), _batch())
    assert torch.isfinite(total) and set(parts) == {"texto", "funcao", "duas"}
    total.backward()


def test_passo_limitado_no_l2():
    from meter_reader.reader import _l2_backbone

    features, channels = _l2_backbone(12, pretrained=False)
    assert channels == 96
    assert features(torch.zeros(1, 3, 48, 192)).shape[2:] == (6, 48)   # passo 8 x 4


def test_candidato_desconhecido():
    with pytest.raises(ValueError):
        Reader("L3")


def test_perda_sem_itens_de_uma_cabeca():
    batch = _batch()
    batch["usa_funcao"] = torch.tensor([False, False, False])
    _, parts = reader_loss(Reader("L1")(torch.randn(3, 3, 48, 192)), batch)
    assert "funcao" not in parts
