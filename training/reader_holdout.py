"""Leitura única do teste 03/07 pelo sistema inteiro (protocolo do leitor, seção 1.16,
U-a; ET6c, T6c.8).

Comandos, a partir da raiz do repositório do produto:
    python -m training.reader_holdout preparar    # copia o lote para a pasta da leitura
    (rodar o meter-reader com os comandos mostrados: é a leitura única)
    python -m training.reader_holdout registrar   # grava os status e os motivos (uma vez)

Tudo fica na pasta de trabalho (`teste_0307/`), fora dos repositórios e da pasta de
dados, que nunca é alterada. O `preparar` recusa rodar de novo depois que a pasta ou o
registro existem: o 03/07 é lido uma vez (1.16, h). Só contagens são exibidas.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

FOLDER = "teste_0307"
RECORD_NAME = Path("experimentos") / "leitor" / "teste_0307_sistema.json"
TEST_VERSION = "r5"


def _lot_dir(photos) -> Path:
    dirs = {p.path.parent for p in photos}
    if len(dirs) != 1:
        raise SystemExit("O teste 03/07 não está numa pasta única.")
    return dirs.pop()


def summarize(rows) -> dict:
    """Status e motivos da planilha, com a fração de aprovadas sobre as imagens lidas."""
    from meter_reader.contracts import Status

    status = Counter(r.status.value for r in rows)
    approved = sum(r.status is Status.APPROVED for r in rows)
    return {"imagens": len(rows), "aprovadas": approved,
            "fracao_aprovadas": approved / len(rows) if rows else 0.0,
            "status": dict(status),
            "motivos": dict(Counter(r.reason.value for r in rows if r.reason is not None))}


def prepare() -> None:
    from meter_reader.config import get_work_dir
    from training.group_dataset import TEST_LABEL_VERSIONS, load_group_photos

    work_dir = get_work_dir()
    work = work_dir / FOLDER
    if work.exists() or (work_dir / RECORD_NAME).exists():
        raise SystemExit("O 03/07 já foi preparado ou lido: a leitura é única (1.16, h).")
    lot_dir = _lot_dir(load_group_photos(TEST_LABEL_VERSIONS[TEST_VERSION]))
    photos, out = work / "fotos", work / "saida"
    photos.mkdir(parents=True)
    out.mkdir()
    files = [f for f in lot_dir.iterdir() if f.is_file()
             and (f.suffix.lower() in (".jpg", ".jpeg", ".png") or f.name.startswith("BaseExtracao_"))]
    for f in files:
        shutil.copy2(f, photos / f.name)   # cópia: o original na pasta de dados não muda
    bases = sorted(photos.glob("BaseExtracao_*.csv"))
    if len(bases) != 1:
        raise SystemExit("O teste 03/07 não tem um BaseExtracao único.")
    print(f"Copiados {len(files) - 1} arquivos de imagem e o BaseExtracao para {FOLDER}/fotos.")
    print("\nRode, no mesmo terminal (é a leitura única do 03/07):")
    print(f'$env:METER_READER_OUTPUT_DIR = "{out}"')
    print(f'meter-reader analisar --fotos "{photos}" --base "{bases[0]}"')


def record() -> None:
    from meter_reader.config import get_work_dir
    from meter_reader.control_sheet import read_control

    work_dir = get_work_dir()
    out_path = work_dir / RECORD_NAME
    if out_path.exists():
        raise SystemExit("A leitura do 03/07 já foi registrada e não é refeita.")
    controls = sorted((work_dir / FOLDER / "saida").glob("*/Controle_*.xlsx"))
    if len(controls) != 1:
        raise SystemExit("A planilha de controle do 03/07 não foi encontrada.")
    rows, executions = read_control(controls[0])
    if len(executions) != 1:
        raise SystemExit("A planilha tem mais de uma execução: a leitura não foi única.")
    result = {**summarize(rows), "versoes": executions[0].model_versions,
              "perfil": executions[0].profile}
    out_path.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=1, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Leitura única do teste 03/07.")
    parser.add_argument("comando", choices=["preparar", "registrar"])
    {"preparar": prepare, "registrar": record}[parser.parse_args().comando]()


if __name__ == "__main__":
    main()
