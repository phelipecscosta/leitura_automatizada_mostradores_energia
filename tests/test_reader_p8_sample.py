from training.reader_p8_sample import proportional


def test_cotas_proporcionais():
    out = proportional({"a": 600, "b": 300, "c": 100}, 50)
    assert out == {"a": 30, "b": 15, "c": 5} and sum(out.values()) == 50
    out = proportional({"a": 1, "b": 1, "c": 1}, 50)
    assert sum(out.values()) == 50 and max(out.values()) - min(out.values()) <= 1
    