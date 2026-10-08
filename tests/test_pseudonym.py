"""Testes da chave de pseudonimização (training/pseudonym.py)."""

import pytest

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
