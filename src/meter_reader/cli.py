"""Comando de linha do esqueleto do veículo (E08; D4).

    meter-reader analisar --fotos <pasta> --base <BaseExtracao_DDMMAAAA_Dia.csv> [--complementar]

Sequência: lê e valida o BaseExtracao inteiro -> confere a pasta das fotos e
o lote -> carrega a triagem, o detector e o leitor -> decide a linha de cada imagem -> grava a
planilha -> move as imagens -> lista as cópias antigas -> exibe o resumo.
A planilha é gravada antes da movimentação: se mover falhar no meio, as fotos
que ficaram na origem são refeitas com --complementar.

Códigos de saída: 0 concluído; 2 entrada ou configuração inválida;
3 lote já processado sem --complementar. As mensagens de erro são as mesmas
que a interface (ET8) vai exibir e não mostram valores do cliente.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from meter_reader import folders
from meter_reader.config import (ConfigError, get_detector_artifact, get_output_dir,
                                 get_reader_artifact, get_triage_artifact, inside_data_dir)
from meter_reader.contracts import ControlRow
from meter_reader.control_sheet import (PARAMETERS_DISABLED, ControlFileError, ExecutionRecord,
                                        RunKind, control_file_name, save_control)
from meter_reader.extraction import ExtractionFileError, read_extraction
from meter_reader.feedback import Summary, format_feedback
from meter_reader.detector import PRODUCT_DETECTOR_SHA256, DetectorArtifactError, load_detector
from meter_reader.pipeline import (DetectorStage, ProvisionalReading, ReaderStage, TriageStage,
                                   process_images)
from meter_reader.reader_model import PRODUCT_READER_SHA256, ReaderArtifactError, load_reader
from meter_reader.reading import ArtifactReader
from meter_reader.triage import PRODUCT_TRIAGE_SHA256, TriageArtifactError, load_triage

EXIT_OK, EXIT_INPUT, EXIT_ALREADY_PROCESSED = 0, 2, 3

# Extensões tratadas como imagens; os demais arquivos da pasta (inclusive o
# próprio BaseExtracao) são ignorados e nunca movidos
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})

# O perfil GPU ainda não existe (E06): a triagem, o detector e o leitor rodam em CPU (V-a)
EXECUTION_PROFILE = "CPU"

# Texto da spec (2.4, ação 3.5), seguido da instrução própria do terminal
ALREADY_PROCESSED_MESSAGE = (
    "Este lote já foi processado anteriormente. Deseja processar as imagens presentes "
    "nesta pasta? Os resultados dessas imagens substituirão os anteriores, e os demais "
    "resultados serão mantidos.\n"
    "Para continuar, repita o comando com --complementar."
)


class InputError(ValueError):
    """Pasta de origem ou pedido inválido. A mensagem é exibível ao utilizador."""


class AlreadyProcessedError(Exception):
    """O lote já tem planilha de controle, e --complementar não foi pedido."""


@dataclass(frozen=True)
class RunResult:
    """Resultado de um processamento."""

    rows: list[ControlRow]
    control_path: Path
    kind: RunKind
    feedback: str


def list_images(folder: Path) -> list[Path]:
    """Imagens da pasta (sem subpastas), em ordem de nome."""
    return sorted(p for p in Path(folder).iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def check_source(source: Path, output_root: Path) -> None:
    """Recusa pastas de origem que o processamento não pode esvaziar."""
    if not source.is_dir():
        raise InputError("A pasta das fotos não existe.")
    if source == output_root or output_root in source.parents:
        raise InputError("A pasta das fotos não pode ficar dentro da pasta de saída.")
    if inside_data_dir(source):
        # E15: a base original é somente leitura; o processamento usa uma cópia
        raise InputError("A pasta das fotos está na base original, que é somente leitura. "
                         "Processe uma cópia.")


def analyze(photos_dir: Path | str, base_path: Path | str, *, complementary: bool,
            output_root: Path, triage_factory: Callable[[], TriageStage],
            triage_version: str, detector_factory: Callable[[], DetectorStage],
            detector_version: str, reader_factory: Callable[[], ReaderStage],
            now: Callable[[], datetime] = datetime.now) -> RunResult:
    """Executa um processamento completo ou complementar."""
    batch = read_extraction(base_path)  # falha cedo, antes de tocar em qualquer foto
    source = Path(photos_dir).expanduser().resolve()
    check_source(source, output_root)

    lot = folders.lot_dir(output_root, batch.batch_date)
    control_path = lot / control_file_name(batch.batch_date)
    exists = control_path.exists()
    if exists and not complementary:
        raise AlreadyProcessedError(ALREADY_PROCESSED_MESSAGE)
    if complementary and not exists:
        raise InputError("Não há processamento anterior deste lote para complementar.")
    images = list_images(source)
    if not images:
        raise InputError("A pasta indicada não contém imagens.")

    reader = reader_factory()
    rows = process_images(images, batch, triage_factory(), detector_factory(), reader)

    kind = RunKind.COMPLEMENTARY if exists else RunKind.FULL
    record = ExecutionRecord(now(), kind, EXECUTION_PROFILE,
                             f"triagem {triage_version}; detector {detector_version}; "
                             f"leitura {reader.version}",
                             PARAMETERS_DISABLED)
    lot.mkdir(parents=True, exist_ok=True)
    save_control(control_path, rows, record)  # primeiro grava a planilha
    try:
        folders.move_images(source, rows, lot)  # depois move
    except OSError as exc:
        raise InputError("A planilha foi gravada, mas a movimentação das imagens parou no "
                         "meio. As imagens que ficaram na pasta de origem podem ser "
                         "processadas de novo com --complementar.") from exc

    stale = folders.stale_copies(rows, lot)
    stale_file = folders.write_stale_list(lot, stale, batch.batch_date)
    feedback = format_feedback(
        Summary.from_rows(rows), control_path, updated=exists, parameters=PARAMETERS_DISABLED,
        stale_by_folder=folders.stale_by_folder(stale, lot),
        stale_list=None if stale_file is None else stale_file.name,
        provisional=isinstance(reader, ProvisionalReading))
    return RunResult(rows, control_path, kind, feedback)

def _load_reader(path: Path) -> ReaderStage:
    """Leitor do produto: o artefato conferido pelo SHA-256, em CPU (V-a)."""
    return ArtifactReader(load_reader(path, PRODUCT_READER_SHA256),
                          f"L2-12 {PRODUCT_READER_SHA256[:12]}")

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="meter-reader",
                                     description="Leitura automatizada de medidores de energia.")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("analisar", help="Analisa as fotos de um lote.")
    run.add_argument("--fotos", required=True, type=Path, help="Pasta com as fotos do lote.")
    run.add_argument("--base", required=True, type=Path,
                     help="Arquivo BaseExtracao_DDMMAAAA_Dia.csv do lote.")
    run.add_argument("--complementar", action="store_true",
                     help="Processa as fotos restantes de um lote já processado.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Ponto de entrada do comando meter-reader."""
    args = _parser().parse_args(argv)
    start = time.perf_counter()
    try:
        output_root = get_output_dir()
        triage_artifact = get_triage_artifact()
        detector_artifact = get_detector_artifact()
        reader_artifact = get_reader_artifact()
        result = analyze(
            args.fotos, args.base, complementary=args.complementar, output_root=output_root,
            triage_factory=lambda: load_triage(triage_artifact, PRODUCT_TRIAGE_SHA256),
            triage_version=f"C4 {PRODUCT_TRIAGE_SHA256[:12]}",
            detector_factory=lambda: load_detector(detector_artifact, PRODUCT_DETECTOR_SHA256),
            detector_version=f"S1_coco {PRODUCT_DETECTOR_SHA256[:12]}",
            reader_factory=lambda: _load_reader(reader_artifact))
    except AlreadyProcessedError as exc:
        print(exc, file=sys.stderr)
        return EXIT_ALREADY_PROCESSED
    except (ConfigError, ExtractionFileError, ControlFileError, TriageArtifactError,
            DetectorArtifactError, ReaderArtifactError, InputError) as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return EXIT_INPUT
    elapsed = time.perf_counter() - start
    print(result.feedback)
    per_image = 1000 * elapsed / max(len(result.rows), 1)
    print(f"\nTempo total: {elapsed:.1f} s ({per_image:.0f} ms por imagem)".replace(".", ","))
    return EXIT_OK
