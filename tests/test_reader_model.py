import hashlib
import math

import numpy as np
import pytest
import torch
from PIL import Image

from meter_reader.reader import Reader
from meter_reader.reader_heads import Heads, trunk_features
from meter_reader.reader_input import upper_band
from meter_reader.reader_model import (READER_ARTIFACT_FORMAT, ReaderArtifactError, ReaderModel,
                                       committee_read, exact_gpu_math, load_reader,
                                       log_mean_exp, sequence_logprob)
from meter_reader.reader_text import FUNCTION_CLASSES, TEXT_ALPHABET

LIMITS = {"d1_logit": 3.0, "limiar_inferior_log": math.log(0.2), "temperatura_texto": 1.4,
          "limiar_funcao": 0.7258}
C = len(TEXT_ALPHABET) + 1


def _steps(*chars, p=0.9):
    """Log-probabilidades com um caractere por passo ("" = branco) com prob. p."""
    lp = torch.full((len(chars), C), math.log((1 - p) / (C - 1)))
    for t, ch in enumerate(chars):
        lp[t, 0 if ch == "" else TEXT_ALPHABET.index(ch) + 1] = math.log(p)
    return lp


def test_probabilidade_da_sequencia_pela_ctc():
    lp = _steps("1", p=0.5)
    assert math.isclose(sequence_logprob(lp, "1"), math.log(0.5), rel_tol=1e-5)
    assert sequence_logprob(lp, "") == -math.inf
    # a temperatura achata a distribuição: a leitura mais provável perde probabilidade
    assert sequence_logprob(lp, "1", 2.0) < sequence_logprob(lp, "1")


def test_comite_maior_media_e_sem_vazias():
    a, b, empty = _steps("1", "", "2"), _steps("1", "", "7"), _steps("", "", "")
    assert committee_read([a, a, b]) == "12"
    assert committee_read([empty, empty, b]) == "17"
    assert committee_read([empty, empty, empty]) == ""
    assert math.isclose(log_mean_exp([math.log(0.2), math.log(0.6)]), math.log(0.4))


def _model():
    torch.manual_seed(0)
    trunks = [Reader("L2-12", pretrained=False).eval() for _ in range(3)]
    dim = trunk_features(trunks[0], torch.zeros(1, 3, 48, 192)).shape[-1]
    return trunks, [Heads(dim).eval() for _ in range(3)]


def _crops(n=3):
    rng = np.random.default_rng(0)
    return [Image.fromarray(rng.integers(0, 255, (40, 120, 3), dtype=np.uint8)) for _ in range(n)]


def test_digital_tem_funcao_e_d1_usa_a_faixa_de_cima():
    trunks, heads = _model()
    crops = _crops()
    never = ReaderModel(trunks, heads, {**LIMITS, "d1_logit": math.inf})
    out = never.read(crops, "display_digital")
    assert [o.two_lines for o in out] == [False] * 3
    assert all(set(o.function) == set(FUNCTION_CLASSES)
               and math.isclose(sum(o.function.values()), 1, rel_tol=1e-5) for o in out)
    always = ReaderModel(trunks, heads, {**LIMITS, "d1_logit": -math.inf})
    fired = always.read(crops, "display_digital")
    band = always.read([upper_band(c) for c in crops], "numero_serie")
    assert [o.two_lines for o in fired] == [True] * 3
    assert [(o.text, o.log_conf) for o in fired] == [(o.text, o.log_conf) for o in band]


def test_outras_classes_sem_funcao():
    trunks, heads = _model()
    out = ReaderModel(trunks, heads, LIMITS).read(_crops(2), "numero_serie")
    assert all(o.function is None and o.two_lines is None for o in out)
    assert ReaderModel(trunks, heads, LIMITS).read([], "numero_serie") == []


def _artifact(tmp_path, **change):
    trunks, heads = _model()
    data = {"formato": READER_ARTIFACT_FORMAT, "candidato": "L2-12",
            "classes_funcao": list(FUNCTION_CLASSES),
            "troncos": [t.state_dict() for t in trunks],
            "cabecas": [h.state_dict() for h in heads], "limiares": dict(LIMITS), "info": {}}
    data.update(change)
    path = tmp_path / "leitor.pt"
    torch.save(data, path)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_carga_confere_sha_e_formato(tmp_path):
    path, sha = _artifact(tmp_path)
    model = load_reader(path, sha)
    assert model.function_threshold == LIMITS["limiar_funcao"]
    with pytest.raises(ReaderArtifactError):
        load_reader(path, "0" * 64)
    (tmp_path / "outro").mkdir()
    path2, sha2 = _artifact(tmp_path / "outro", formato="outro")
    with pytest.raises(ReaderArtifactError):
        load_reader(path2, sha2)


def test_formato_igual_ao_do_treino():
    from training.reader_final import ARTIFACT_FORMAT
    assert ARTIFACT_FORMAT == READER_ARTIFACT_FORMAT


def test_gpu_sem_tf32():
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    exact_gpu_math()
    assert not torch.backends.cudnn.allow_tf32
    assert not torch.backends.cuda.matmul.allow_tf32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="sem GPU")
def test_leitor_na_gpu_desliga_o_tf32():
    torch.backends.cudnn.allow_tf32 = True
    trunks, heads = _model()
    ReaderModel(trunks, heads, LIMITS, device="cuda")
    assert not torch.backends.cudnn.allow_tf32
