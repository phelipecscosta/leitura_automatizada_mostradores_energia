"""Gera o manifesto das imagens: uma linha por imagem, de todos os lotes.

Aplica as regras de entrada do produto (leitor, pareamento e integridade) e
acrescenta a assinatura visual. O manifesto contém dados do cliente e é
gravado na pasta de trabalho local, fora dos repositórios (T1.7, decisão 1).

Uso, a partir da raiz do repositório do produto:
    python -m training.build_manifest
"""

from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path

from meter_reader.config import get_data_dir, get_work_dir
from meter_reader.contracts import IntegrityIssue
from meter_reader.extraction import read_extraction
from meter_reader.integrity import check_integrity, measure
from meter_reader.pairing import pair_images
from training.fingerprint import dhash

MANIFEST_NAME = "manifesto_imagens.csv"
COLUMNS = (
    "lote", "arquivo", "sufixo", "integridade", "brilho", "contraste",
    "pareada", "numero_esperado", "partes_numero", "leitura", "nota", "dhash",
)
_SUFFIX_RE = re.compile(r"_(\d{3})\.jpg$", re.IGNORECASE)


def manifest_rows(lot_dir: Path) -> list[dict]:
    """Uma linha por imagem do lote, com as regras de entrada aplicadas."""
    csv_files = sorted(lot_dir.glob("BaseExtracao_*.csv"))
    if len(csv_files) != 1:
        raise ValueError("Cada lote precisa ter exatamente um arquivo BaseExtracao.")
    batch = read_extraction(csv_files[0])

    images = sorted(
        p for p in lot_dir.iterdir() if p.is_file() and p.suffix.lower() == ".jpg"
    )
    paired = {r.photo_name: r.row for r in pair_images([p.name for p in images], batch)}

    rows = []
    for path in images:
        integrity = check_integrity(path)
        readable = integrity.issue is not IntegrityIssue.UNREADABLE
        stats = measure(path) if readable else None
        row = paired[path.name]
        suffix = _SUFFIX_RE.search(path.name)
        rows.append({
            "lote": batch.batch_date.isoformat(),
            "arquivo": path.name,
            "sufixo": suffix.group(1) if suffix else "",
            "integridade": integrity.issue.value if integrity.issue else "",
            "brilho": round(stats.mean, 2) if stats else "",
            "contraste": round(stats.std, 2) if stats else "",
            "pareada": row is not None,
            "numero_esperado": row.expected_meter_number if row else "",
            "partes_numero": len(row.meter_numbers) if row else "",
            "leitura": row.reading if row else "",
            "nota": (row.note or "") if row else "",
            "dhash": dhash(path) if readable else "",
        })
    return rows


def write_manifest(rows: list[dict], path: Path) -> None:
    """Grava o manifesto em CSV UTF-8, separado por vírgula."""
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: Path) -> str:
    """Impressão digital do arquivo: identifica o conteúdo sem revelá-lo."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    lot_dirs = sorted(p for p in get_data_dir().iterdir() if p.is_dir())
    rows = [row for lot in lot_dirs for row in manifest_rows(lot)]
    output = get_work_dir() / MANIFEST_NAME
    write_manifest(rows, output)
    # Só contagens e a impressão digital; nunca caminhos (decisão E15)
    print(f"Lotes: {len(lot_dirs)} | imagens: {len(rows)}")
    print(f"SHA-256 do manifesto: {file_sha256(output)}")


if __name__ == "__main__":
    main()
