"""Gera a folha de rotulagem do Lab02 (T2.5).

Uma linha por foto a rotular, identificada pelo pseudônimo. A folha fica na
pasta de trabalho, porque guarda o caminho local da foto e a leitura digitada
(mostrada só na passada 2). A ferramenta de rotulagem grava apenas o
pseudônimo, então o arquivo de rótulos já nasce publicável.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from meter_reader.config import get_data_dir, get_work_dir
from training.build_manifest import MANIFEST_NAME, file_sha256
from training.build_sample import ROLE_BLIND, ROLE_LABEL, ROLE_ZERO, SAMPLE_NAME
from training.pseudonym import KEY_NAME, build_mapping, load_key

SHEET_NAME = "folha_rotulagem.csv"
SHEET_COLUMNS = (
    "pseudonimo", "lote", "conjunto", "papel", "posicao", "posicao_estresse",
    "leitura", "caminho",
)
# Ordem de trabalho: às cegas primeiro (piloto), depois a rotulagem, depois o estresse
_ROLE_ORDER = {ROLE_BLIND: 0, ROLE_LABEL: 1, ROLE_ZERO: 2}


def build_sheet(sample: list[dict], readings: dict[str, str],
                mapping: dict[str, str], paths: dict[str, str]) -> list[dict]:
    """Linhas da folha: só as fotos com papel, já pseudonimizadas e ordenadas."""
    rows = []
    for s in sample:
        if not s["papel"]:
            continue
        name = s["arquivo"]
        rows.append({
            "pseudonimo": mapping[name], "lote": s["lote"],
            "conjunto": s["conjunto"], "papel": s["papel"],
            "posicao": s["posicao"], "posicao_estresse": s["posicao_estresse"],
            "leitura": readings[name], "caminho": paths[name],
        })
    # Dentro de cada papel, alterna os lotes pela posição sorteada, para que
    # o cansaço do rotulador não se concentre num lote só
    rows.sort(key=lambda r: (
        _ROLE_ORDER[r["papel"]],
        int(r["posicao_estresse"] if r["papel"] == ROLE_ZERO else r["posicao"]),
        r["lote"],
    ))
    return rows


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def main() -> None:
    work_dir, data_dir = get_work_dir(), get_data_dir()
    sample = _read_csv(work_dir / SAMPLE_NAME)
    manifest = _read_csv(work_dir / MANIFEST_NAME)

    readings = {r["arquivo"]: r["leitura"] for r in manifest}
    mapping = build_mapping(load_key(work_dir / KEY_NAME), "arquivo", readings)
    # Caminho relativo à pasta de dados: a folha continua válida se a base mudar de lugar
    paths = {
        p.name: p.relative_to(data_dir).as_posix()
        for p in data_dir.rglob("*") if p.suffix.lower() == ".jpg"
    }

    rows = build_sheet(sample, readings, mapping, paths)
    output = work_dir / SHEET_NAME
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SHEET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    # Só contagens e a impressão digital (E15)
    for role, n in sorted(Counter(r["papel"] for r in rows).items()):
        print(f"{role:14} {n:4}")
    print(f"SHA-256 da folha: {file_sha256(output)}")


if __name__ == "__main__":
    main()
