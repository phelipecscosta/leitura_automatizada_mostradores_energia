from meter_reader.contracts import Reason, Status
from training.reader_holdout import summarize


class _Row:
    def __init__(self, status, reason=None):
        self.status, self.reason = status, reason


def test_resumo_da_leitura():
    rows = [_Row(Status.APPROVED), _Row(Status.APPROVED),
            _Row(Status.NEEDS_REVIEW, Reason.LOW_CONFIDENCE)]
    out = summarize(rows)
    assert out["imagens"] == 3 and out["aprovadas"] == 2
    assert abs(out["fracao_aprovadas"] - 2 / 3) < 1e-12
    assert out["motivos"] == {Reason.LOW_CONFIDENCE.value: 1}
    assert summarize([])["fracao_aprovadas"] == 0.0
