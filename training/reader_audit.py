"""Auditoria A1-b do teste 03/07 (protocolo do leitor, seção 1.16, itens c a f; ET6c,
T6c.9).

    python -m training.reader_audit montar      # blocos, ordem e armadilhas (uma vez)
    (auditoria na ferramenta local da pesquisa)
    python -m training.reader_audit resultado   # risco das aprovadas (só contagens)

Todas as aprovadas, em blocos de 50 com 3 armadilhas em posições sorteadas. As
armadilhas vêm das fotos do 03/07 que foram para "Verificar" só pela confiança da
função (número de série e leitura conferidos com o CSV, classe mais provável de
consumo), com um caractere trocado na leitura ou no número de série. Os arquivos ficam
na pasta de trabalho (`teste_0307/`).
"""

from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path

import pandas as pd

FOLDER = "teste_0307"
ITEMS_NAME, RESERVE_NAME, MARKS_NAME = ("auditoria_itens.csv", "auditoria_reserva.csv",
                                        "auditoria_marcas.csv")
BLOCK, TRAPS, MAX_MISSED = 50, 3, 2          # 1.16 (d)
RESERVE_BLOCKS = 2                          # armadilhas de reserva: 2 refações por bloco
CONFUSABLE = {"0": "8", "8": "0", "1": "7", "7": "1", "5": "6", "6": "5", "3": "8",
              "B": "8", "S": "5", "O": "0", "D": "0", "I": "1", "Z": "2"}
DIGITS = "0123456789"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def seeded(name: str) -> random.Random:
    return random.Random(int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big"))


def swap_char(value: str, rng: random.Random) -> str:
    """Troca um caractere, de preferência num par parecido; nunca devolve o original."""
    spots = [i for i, c in enumerate(value) if c in CONFUSABLE]
    if spots:
        i = rng.choice(spots)
        new = CONFUSABLE[value[i]]
    else:
        i = rng.randrange(len(value))
        pool = DIGITS if value[i].isdigit() else LETTERS
        new = rng.choice([c for c in pool if c != value[i]])
    return value[:i] + new + value[i + 1:]


def make_trap(row: dict, rng: random.Random) -> dict:
    """Armadilha a partir de uma linha conferida: troca um caractere da leitura ou do
    número de série (sorteado)."""
    trap = dict(row, tipo="armadilha")
    if rng.random() < 0.5:
        trap["leitura"] = str(int(swap_char(str(row["leitura"]), rng)))
        trap["campo_trocado"] = "leitura"
    else:
        trap["numero"] = swap_char(row["numero"], rng)
        trap["campo_trocado"] = "numero"
    return trap


def build_blocks(real: list[dict], pool: list[dict]) -> tuple[list[dict], list[dict]]:
    """Ordem das aprovadas, blocos de 50 com 3 armadilhas em posições sorteadas, e a
    reserva de armadilhas para os blocos refeitos."""
    order = seeded("a1b-ordem")
    trap_rng = seeded("a1b-armadilhas")
    real = real[:]
    order.shuffle(real)
    pool = pool[:]
    trap_rng.shuffle(pool)
    blocks = [real[i:i + BLOCK] for i in range(0, len(real), BLOCK)]
    need = len(blocks) * TRAPS * (1 + RESERVE_BLOCKS)
    if len(pool) < need:
        raise SystemExit(f"Fotos para armadilhas insuficientes: {len(pool)} de {need}.")
    traps = [make_trap(r, trap_rng) for r in pool[:need]]
    items = []
    for b, block in enumerate(blocks, 1):
        chosen = traps[(b - 1) * TRAPS:b * TRAPS]
        seq = [dict(r, tipo="real", campo_trocado="") for r in block]
        for t in chosen:
            seq.insert(order.randrange(len(seq) + 1), t)
        items += [dict(r, bloco=b, posicao=k) for k, r in enumerate(seq, 1)]
    return items, traps[len(blocks) * TRAPS:]


def build() -> None:
    from meter_reader.config import get_work_dir
    from meter_reader.contracts import Reason, Status
    from meter_reader.control_sheet import read_control
    from meter_reader.extraction import read_extraction
    from meter_reader.pairing import index_by_photo
    from training.group_dataset import TEST_LABEL_VERSIONS, load_group_photos
    from training.reader_holdout import TEST_VERSION

    work = get_work_dir() / FOLDER
    if (work / ITEMS_NAME).exists():
        raise SystemExit("A auditoria já foi montada e não é sorteada de novo.")
    rows, _ = read_control(next((work / "saida").glob("*/Controle_*.xlsx")))
    index = index_by_photo(read_extraction(next((work / "fotos").glob("BaseExtracao_*.csv"))))
    code = {p.path.name: p.pseudonym
            for p in load_group_photos(TEST_LABEL_VERSIONS[TEST_VERSION])}
    real, pool = [], []
    for r in rows:
        row = index.get(r.photo_name)
        if row is None or r.photo_name not in code:
            continue
        item = {"pseudonimo": code[r.photo_name], "leitura": r.reading, "numero": r.meter_number,
                "funcao": r.function or ""}
        if r.status is Status.APPROVED:
            real.append({**item, "leitura": row.reading, "numero": row.expected_meter_number})
        elif (r.status is Status.NEEDS_REVIEW and r.reason is Reason.LOW_CONFIDENCE
              and r.meter_number == row.expected_meter_number and r.reading == row.reading):
            # Só a confiança da função reteve: leitura e número conferidos com o CSV, e a
            # classe mais provável é de consumo; a armadilha erra só no caractere trocado
            pool.append({**item, "leitura": row.reading, "numero": row.expected_meter_number})
    real.sort(key=lambda d: d["pseudonimo"])
    pool.sort(key=lambda d: d["pseudonimo"])
    items, reserve = build_blocks(real, pool)
    pd.DataFrame(items).to_csv(work / ITEMS_NAME, index=False)
    pd.DataFrame(reserve).to_csv(work / RESERVE_NAME, index=False)
    n_blocks = max(i["bloco"] for i in items)
    print(f"Aprovadas: {len(real)}; blocos: {n_blocks}; armadilhas: {n_blocks * TRAPS}; "
          f"reserva: {len(reserve)}; fotos possíveis para armadilha: {len(pool)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Auditoria A1-b do 03/07.")
    parser.add_argument("comando", choices=["montar"])
    {"montar": build}[parser.parse_args().comando]()


if __name__ == "__main__":
    main()
