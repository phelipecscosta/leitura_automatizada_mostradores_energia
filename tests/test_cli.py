"""Testes do comando de linha, com triagem falsa, imagens sintéticas e pastas temporárias."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from meter_reader import cli
from meter_reader.config import DATA_DIR_VAR
from meter_reader.contracts import TRIAGE_REJECTIONS
from meter_reader.control_sheet import RunKind, read_control
from meter_reader.extraction import ExtractionFileError
from meter_reader.pipeline import ProvisionalReading
from meter_reader.triage import TriageResult

BASE_NAME = "BaseExtracao_20052026_Dia.csv"
HEADER = "Numero do medidor;Posicao do medidor lida;Nota de Leitura Atual;Foto do medidor"


class FakeTriage:
    """Rejeita como "outros" as fotos cujo nome começa com "poste"."""

    def predict(self, path: Path) -> TriageResult:
        if path.name.startswith("poste"):
            return TriageResult((0, 0, 0, 0, 1), 1.0, TRIAGE_REJECTIONS["outros"], None)
        return TriageResult((1, 0, 0, 0, 0), 0.0, None, "digital")

class FakeDetector:
    """Passa todas as fotos, com um display digital (o tipo da FakeTriage)."""

    def predict(self, path: Path):
        import numpy as np
        from meter_reader.detector import DetectionResult
        from meter_reader.fields import FieldSelection
        box = np.array([[10, 10, 40, 20]], np.float32)
        return DetectionResult(box, np.array([0.9], np.float32), np.array([1]), 0.9, True,
                               "digital", FieldSelection("digital", box[0], None))

def _image(path: Path) -> None:
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)).save(path)


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Pasta de fotos com BaseExtracao, raiz de saída vazia e sem pasta de dados."""
    monkeypatch.delenv(DATA_DIR_VAR, raising=False)
    photos, out = tmp_path / "fotos", tmp_path / "saida"
    photos.mkdir()
    out.mkdir()
    for name in ("medidor1.jpg", "medidor2.JPG", "poste.jpg", "orfa.jpg"):
        _image(photos / name)
    (photos / BASE_NAME).write_text(
        "\n".join([HEADER, "A1;10;NA;medidor1.jpg", "A2;20;NA;medidor2.JPG",
                   "A3;30;NA;poste.jpg", "A4;40;NA;nova.jpg"]) + "\n", encoding="utf-8")
    return photos, out


def _run(photos, out, complementary=False, calls=None):
    def factory():
        if calls is not None:
            calls.append(1)
        return FakeTriage()

    return cli.analyze(photos, photos / BASE_NAME, complementary=complementary,
                       output_root=out, triage_factory=factory, triage_version="C4 teste",
                       detector_factory=FakeDetector, detector_version="teste",
                       reader=ProvisionalReading(), now=lambda: datetime(2026, 11, 10, 16))


def test_full_run_moves_saves_and_reports(env) -> None:
    photos, out = env
    result = _run(photos, out)
    lot = out / "20052026"
    assert result.control_path == lot / "Controle_20052026.xlsx"
    assert result.kind is RunKind.FULL
    assert (lot / "Verificar" / "medidor1.jpg").exists()
    assert (lot / "Verificar" / "medidor2.JPG").exists()
    assert (lot / "Rejeitadas" / "Outros" / "poste.jpg").exists()
    assert (lot / "Rejeitadas" / "Não listada" / "orfa.jpg").exists()
    # Só o BaseExtracao fica na origem: arquivos que não são imagens nunca são movidos
    assert [p.name for p in photos.iterdir()] == [BASE_NAME]
    rows, executions = read_control(result.control_path)
    assert len(rows) == 4 and len(executions) == 1
    assert executions[0].model_versions == "triagem C4 teste; detector teste; leitura provisorio"
    assert "gerada" in result.feedback and "Leitura: provisória" in result.feedback
    assert "medidor1" not in result.feedback


def test_second_run_without_complementary_is_refused(env) -> None:
    photos, out = env
    _run(photos, out)
    _image(photos / "nova.jpg")
    with pytest.raises(cli.AlreadyProcessedError, match="já foi processado"):
        _run(photos, out)
    assert (photos / "nova.jpg").exists()  # nada foi movido


def test_complementary_run_keeps_previous_rows(env) -> None:
    photos, out = env
    first = _run(photos, out)
    _image(photos / "nova.jpg")
    second = _run(photos, out, complementary=True)
    rows, executions = read_control(second.control_path)
    assert [r.photo_name for r in rows] == [r.photo_name for r in first.rows] + ["nova.jpg"]
    assert [e.kind for e in executions] == [RunKind.FULL, RunKind.COMPLEMENTARY]
    assert second.kind is RunKind.COMPLEMENTARY and "atualizada" in second.feedback
    assert len(second.rows) == 1  # o resumo conta só as imagens deste processamento


def test_complementary_without_previous_run_is_refused(env) -> None:
    photos, out = env
    with pytest.raises(cli.InputError, match="complementar"):
        _run(photos, out, complementary=True)


def test_invalid_base_touches_nothing(env) -> None:
    photos, out = env
    (photos / BASE_NAME).write_text(HEADER + "\nA1;dez;NA;medidor1.jpg\n", encoding="utf-8")
    calls = []
    with pytest.raises(ExtractionFileError, match="Linha 2"):
        _run(photos, out, calls=calls)
    assert calls == []  # a triagem nem é carregada
    assert list(out.iterdir()) == []
    assert len(list(photos.iterdir())) == 5


def test_source_inside_output_is_refused(env) -> None:
    photos, out = env
    inside = out / "20052026" / "Verificar"
    inside.mkdir(parents=True)
    (inside / BASE_NAME).write_bytes((photos / BASE_NAME).read_bytes())
    with pytest.raises(cli.InputError, match="pasta de saída"):
        cli.analyze(inside, inside / BASE_NAME, complementary=False, output_root=out,
                    triage_factory=FakeTriage, triage_version="v",
                    detector_factory=FakeDetector, detector_version="v",
                    reader=ProvisionalReading())


def test_source_inside_original_base_is_refused(env, monkeypatch) -> None:
    photos, out = env
    monkeypatch.setenv(DATA_DIR_VAR, str(photos.parent))
    with pytest.raises(cli.InputError, match="somente leitura"):
        _run(photos, out)


def test_folder_without_images_is_refused(env) -> None:
    photos, out = env
    for p in photos.iterdir():
        if p.name != BASE_NAME:
            p.unlink()
    with pytest.raises(cli.InputError, match="não contém imagens"):
        _run(photos, out)


def test_move_failure_keeps_sheet_and_explains(env, monkeypatch) -> None:
    photos, out = env

    def broken(*args, **kwargs):
        raise OSError("disco cheio")

    monkeypatch.setattr(cli.folders, "move_images", broken)
    with pytest.raises(cli.InputError, match="--complementar"):
        _run(photos, out)
    assert (out / "20052026" / "Controle_20052026.xlsx").exists()


# ---------------------------------------------------------------------
# Ponto de entrada: códigos de saída
# ---------------------------------------------------------------------


@pytest.fixture
def configured(env, monkeypatch):
    photos, out = env
    monkeypatch.setattr(cli, "get_output_dir", lambda: out)
    monkeypatch.setattr(cli, "get_triage_artifact", lambda: Path("triagem.pt"))
    monkeypatch.setattr(cli, "load_triage", lambda path, sha: FakeTriage())
    monkeypatch.setattr(cli, "get_detector_artifact", lambda: Path("detector.pt"))
    monkeypatch.setattr(cli, "load_detector", lambda path, sha: FakeDetector())
    return photos, out


def _argv(photos, *extra):
    return ["analisar", "--fotos", str(photos), "--base", str(photos / BASE_NAME), *extra]


def test_main_success_then_already_processed(configured, capsys) -> None:
    photos, _ = configured
    assert cli.main(_argv(photos)) == cli.EXIT_OK
    assert "Processamento concluído." in capsys.readouterr().out
    _image(photos / "nova.jpg")
    assert cli.main(_argv(photos)) == cli.EXIT_ALREADY_PROCESSED
    assert "--complementar" in capsys.readouterr().err
    assert cli.main(_argv(photos, "--complementar")) == cli.EXIT_OK


def test_main_input_error_exit_code(configured, capsys) -> None:
    photos, _ = configured
    (photos / BASE_NAME).write_text("cabecalho errado\n", encoding="utf-8")
    assert cli.main(_argv(photos)) == cli.EXIT_INPUT
    assert capsys.readouterr().err.startswith("Erro: Cabeçalho")


def test_main_config_error_exit_code(env, monkeypatch, capsys) -> None:
    photos, _ = env

    def missing():
        raise cli.ConfigError("Variável METER_READER_OUTPUT_DIR não definida.")

    monkeypatch.setattr(cli, "get_output_dir", missing)
    assert cli.main(_argv(photos)) == cli.EXIT_INPUT
    assert "não definida" in capsys.readouterr().err
