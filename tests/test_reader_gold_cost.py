from training.reader_gold_cost import label_source


def test_origem_do_rotulo_pela_mais_forte():
    assert label_source({"haiku_csv"}, judged=True) == "humano"
    assert label_source({"marca_professor", "humano_f2"}, judged=False) == "humano"
    assert label_source({"haiku_csv", "pseudo"}, judged=False) == "aceito_professor"
    assert label_source({"marca_professor"}, judged=False) == "marca_professor"
    assert label_source(set(), judged=False) == "sem_rotulo"
    assert label_source({"pseudo"}, judged=False) == "outro"
    