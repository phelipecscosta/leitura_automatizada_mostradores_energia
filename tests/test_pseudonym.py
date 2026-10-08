"""Testes da chave de pseudonimização (training/pseudonym.py)."""

import pytest
import re

from training import pseudonym as ps


def test_create_and_load(tmp_path):
    path = tmp_path / ps.KEY_NAME
    ps.create_key(path)
    assert len(ps.load_key(path)) == ps.KEY_BYTES


def test_never_overwrites(tmp_path):
    path = tmp_path / ps.KEY_NAME
    ps.create_key(path)
    original = path.read_bytes()
    with pytest.raises(ps.PseudonymError):
        ps.create_key(path)
    assert path.read_bytes() == original  # a chave existente ficou intacta


def test_keys_are_random(tmp_path):
    a, b = tmp_path / "a.key", tmp_path / "b.key"
    ps.create_key(a)
    ps.create_key(b)
    assert ps.load_key(a) != ps.load_key(b)


@pytest.mark.parametrize("content", [
    b"zz" * 32,           # não é hexadecimal
    b"ab" * 16,           # 16 bytes: curta demais
    "é".encode("utf-8"),  # não é ASCII
])
def test_invalid_key_fails(tmp_path, content):
    path = tmp_path / ps.KEY_NAME
    path.write_bytes(content)
    with pytest.raises(ps.PseudonymError):
        ps.load_key(path)


def test_missing_key_fails_without_path(tmp_path):
    with pytest.raises(ps.PseudonymError) as error:
        ps.load_key(tmp_path / ps.KEY_NAME)
    assert str(tmp_path) not in str(error.value)  # mensagem sem caminho (E15)

KEY = bytes(32)        # chave fictícia, só para os testes
OTHER_KEY = bytes([1]) * 32


def test_code_format():
    code = ps.pseudonym(KEY, "arquivo", "x.jpg")
    assert re.fullmatch(r"F-[0-9a-f]{12}", code)
    assert ps.pseudonym(KEY, "medidor", "123").startswith("M-")


def test_deterministic_and_key_dependent():
    code = ps.pseudonym(KEY, "arquivo", "x.jpg")
    assert code == ps.pseudonym(KEY, "arquivo", "x.jpg")
    assert code != ps.pseudonym(OTHER_KEY, "arquivo", "x.jpg")


def test_kind_and_case_change_the_code():
    code = ps.pseudonym(KEY, "arquivo", "x.jpg")
    assert code[2:] != ps.pseudonym(KEY, "medidor", "x.jpg")[2:]
    assert code != ps.pseudonym(KEY, "arquivo", "X.jpg")


@pytest.mark.parametrize("kind, value", [("lote", "x.jpg"), ("arquivo", "")])
def test_invalid_inputs(kind, value):
    with pytest.raises(ValueError):
        ps.pseudonym(KEY, kind, value)


def test_mapping_is_complete_and_unique():
    values = ["b.jpg", "a.jpg", "b.jpg", "c.jpg"]  # com repetição
    mapping = ps.build_mapping(KEY, "arquivo", values)
    assert set(mapping) == set(values)
    assert len(set(mapping.values())) == len(mapping)  # códigos distintos


def test_mapping_detects_collision(monkeypatch):
    # Com 1 caractere hexadecimal há só 16 códigos possíveis:
    # 17 valores colidem com certeza (princípio da casa dos pombos)
    monkeypatch.setattr(ps, "CODE_HEX_CHARS", 1)
    values = [f"foto_{i}.jpg" for i in range(17)]
    with pytest.raises(ps.PseudonymError) as error:
        ps.build_mapping(KEY, "arquivo", values)
    assert not any(v in str(error.value) for v in values)  # sem dados na mensagem

def test_write_table(tmp_path):
    manifest = tmp_path / "manifesto.csv"
    manifest.write_text(
        "lote,arquivo\n2000-01-01,a.jpg\n2000-01-01,b.jpg\n2000-01-02,a.jpg\n",
        encoding="utf-8",
    )
    output = tmp_path / "tabela.csv"
    rows, entries = ps.write_table(KEY, manifest, output)
    assert (rows, entries) == (3, 2)  # nome repetido aparece na diferença

    lines = output.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "arquivo,pseudonimo"
    assert lines[1] == f"a.jpg,{ps.pseudonym(KEY, 'arquivo', 'a.jpg')}"
    assert len(lines) == 3

def test_lookup(tmp_path):
    manifest = tmp_path / "manifesto.csv"
    manifest.write_text("lote,arquivo\n2000-01-01,a.jpg\n", encoding="utf-8")
    table = tmp_path / "tabela.csv"
    ps.write_table(KEY, manifest, table)
    code = ps.pseudonym(KEY, "arquivo", "a.jpg")
    assert ps.lookup(table, [code, "F-000000000000"]) == {
        code: "a.jpg",
        "F-000000000000": None,
    }
