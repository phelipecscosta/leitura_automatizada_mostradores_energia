"""Ensaio de ponta a ponta do software no lote 22/05 (ET6c, T6c.7; decisão En-b).

O lote 22/05 entrou no treino do modelo final: as taxas daqui são otimistas e servem
só de alarme de defeito de código (recorte, orientação, colunas, pastas), nunca de
avaliação.

Comandos, a partir da raiz do repositório do produto:
    python -m training.reader_rehearsal preparar   # copia o lote para a pasta do ensaio
    (rodar o meter-reader com os comandos mostrados)
    python -m training.reader_rehearsal conferir   # Aprovadas x rótulos existentes
    python -m training.reader_rehearsal diagnosticar   # tipos de divergência (só contagens)
    python -m training.reader_rehearsal inspecao   # contagens da inspeção visual (local/)
    python -m training.reader_rehearsal limpar     # apaga as cópias e a saída do ensaio

Tudo fica na pasta de trabalho (`ensaio_2205/`), fora dos repositórios e da pasta de
dados, que nunca é alterada. Só contagens são exibidas.
"""

from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path

LOT = "2026-05-22"
FOLDER = "ensaio_2205"
CYCLOMETRIC_TOLERANCE = 1   # B10-b
VISUAL_NAME = "amostra_visual.csv"
VISUAL_PER_GROUP = 10


def _lot_dir(photos) -> Path:
    dirs = {p.path.parent for p in photos if p.lot == LOT}
    if len(dirs) != 1:
        raise SystemExit("O lote do ensaio não está numa pasta única.")
    return dirs.pop()


def reading_ok(label: str, value: int, field_class: str) -> bool:
    """Leitura aprovada x rótulo do display (número inteiro; tolerância no ciclométrico)."""
    if not label.isdigit():
        return False
    tol = CYCLOMETRIC_TOLERANCE if field_class == "display_ciclometrico" else 0
    return abs(int(label) - value) <= tol


def tally(approved: list[dict], text_labels: dict, serial_labels: dict, gold: dict) -> dict:
    """Contagens da conferência das Aprovadas, por campo e pela origem do rótulo.

    `approved`: pseudonimo, classe (do display), leitura e numero (da planilha);
    `text_labels` e `serial_labels`: pseudônimo -> lista de (texto, origem);
    `gold`: pseudônimo -> (status, classe da função) do rótulo-ouro.
    """
    from meter_reader.reader_text import CONSUMPTION_CLASSES

    out = {"aprovadas": len(approved), "leitura": Counter(), "numero": Counter(),
           "funcao": Counter()}
    for a in approved:
        labels = text_labels.get((a["pseudonimo"], a["classe"]), [])
        if not labels:
            out["leitura"]["sem_rotulo"] += 1
        else:
            ok = any(reading_ok(t, a["leitura"], a["classe"]) for t, _ in labels)
            out["leitura"][f"{'confere' if ok else 'diverge'}_{labels[0][1]}"] += 1
        serials = serial_labels.get(a["pseudonimo"], [])
        if not serials:
            out["numero"]["sem_rotulo"] += 1
        else:
            ok = any(t == a["numero"] for t, _ in serials)
            out["numero"][f"{'confere' if ok else 'diverge'}_{serials[0][1]}"] += 1
        if a["classe"] == "display_digital":
            status, cls = gold.get(a["pseudonimo"], ("sem_rotulo_ouro", None))
            if status in ("funcao", "vazio"):
                out["funcao"]["consumo" if cls in CONSUMPTION_CLASSES else "nao_consumo"] += 1
            else:
                out["funcao"][status] += 1
    return {k: dict(v) if isinstance(v, Counter) else v for k, v in out.items()}

def edit_distance(a: str, b: str) -> int:
    """Distância de Levenshtein (inserção, remoção e troca)."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def serial_kind(read: str, parts: tuple[str, ...]) -> str:
    """Tipo de divergência entre o número lido e o esperado (bloco mais à esquerda)."""
    expected = parts[0]
    if read.upper() == expected.upper():
        return "so_maiusculas_minusculas"
    if any(read.upper() == p.upper() for p in parts[1:]):
        return "outro_bloco_do_csv"
    if expected.upper() in read.upper():
        return "lido_contem_o_esperado"
    if read.upper() in expected.upper():
        return "esperado_contem_o_lido"
    d = edit_distance(read.upper(), expected.upper())
    return "um_caractere" if d == 1 else ("dois_caracteres" if d == 2 else "outro")


def reading_kind(read: int, expected: int) -> str:
    """Tipo de divergência entre a leitura lida e a do CSV."""
    a, b = str(read), str(expected)
    if abs(read - expected) <= 1:
        return "diferenca_de_1"
    if len(a) != len(b):
        return "numero_de_digitos"
    d = edit_distance(a, b)
    return "um_digito" if d == 1 else "outro"

def digit_detail(read: int, expected: int) -> str:
    """Leitura com número de dígitos diferente do CSV: onde está o dígito a mais ou a menos."""
    a, b = str(read), str(expected)
    if read // 10 == expected:
        return "digito_a_mais_no_fim"
    if expected // 10 == read:
        return "digito_a_menos_no_fim"
    if len(a) > len(b) and a.endswith(b):
        return "digitos_a_mais_no_inicio"
    if len(b) > len(a) and b.endswith(a):
        return "digitos_a_menos_no_inicio"
    return "outro"


def teacher_kind(read: str, teacher: str | None, status: str | None, expected: str) -> str:
    """Número lido pelo leitor x resposta do professor (rotulagem da ET6a) no mesmo recorte."""
    if teacher is None:
        return "sem_resposta_do_professor"
    if status in ("ilegivel", "parcial") or not teacher:
        return f"professor_{status or 'vazio'}"
    if teacher.upper() == read.upper():
        return "professor_igual_ao_lido"
    if teacher.upper() == expected.upper():
        return "professor_igual_ao_csv"
    return "professor_outro"


def diagnose() -> None:
    import pandas as pd

    from meter_reader.config import get_work_dir
    from meter_reader.contracts import Reason
    from meter_reader.control_sheet import read_control
    from meter_reader.extraction import read_extraction
    from meter_reader.pairing import index_by_photo
    from training.group_dataset import LABEL_VERSIONS, load_group_photos
    from training.reader_dataset import DIGITAL
    from training.reader_labels import CROPS_NAME, labels_name
    from training.teacher import MODEL

    work = get_work_dir() / FOLDER
    controls = sorted((work / "saida").glob("*/Controle_*.xlsx"))
    bases = sorted((work / "fotos").glob("BaseExtracao_*.csv"))
    if len(controls) != 1 or len(bases) != 1:
        raise SystemExit("A planilha ou o BaseExtracao do ensaio não foi encontrado.")
    rows, _ = read_control(controls[0])
    index = index_by_photo(read_extraction(bases[0]))
    pseudonym = {p.path.name: p.pseudonym
                 for p in load_group_photos(LABEL_VERSIONS["r4"]) if p.lot == LOT}
    raw = pd.read_csv(get_work_dir() / labels_name(MODEL), dtype=str, keep_default_na=False)
    raw = raw[raw["classe"] == "numero_serie"].drop_duplicates("pseudonimo").set_index("pseudonimo")
    lot_status = Counter(raw.loc[raw.index.isin(set(pseudonym.values())), "status"])
    crops = pd.read_csv(get_work_dir() / CROPS_NAME, usecols=["pseudonimo", "classe"])
    digital = set(crops.loc[crops["classe"] == DIGITAL, "pseudonimo"])
    serial, reading, case, teacher = Counter(), Counter(), Counter(), Counter()
    swap, digits, sample = Counter(), Counter(), []
    for r in rows:
        row = index.get(r.photo_name)
        if row is None:
            continue
        if r.reason is Reason.METER_NUMBER_MISMATCH and r.meter_number is not None:
            serial[serial_kind(r.meter_number, row.meter_numbers)] += 1
            code = pseudonym.get(r.photo_name)
            hit = raw.loc[code] if code in raw.index else None
            kind = teacher_kind(r.meter_number, None if hit is None else hit["texto"],
                                None if hit is None else hit["status"], row.expected_meter_number)
            teacher[kind] += 1
            family = "display_digital" if code in digital else "display_ciclometrico"
            same = r.reading is not None and reading_ok(str(r.reading), row.reading, family)
            swap[f"{kind}|leitura_{'confere' if same else 'nao_confere'}"] += 1
            sample.append({"pseudonimo": code, "categoria": f"numero_{kind}"})
            case["esperado_com_minuscula" if row.expected_meter_number != row.expected_meter_number.upper()
                 else "esperado_sem_minuscula"] += 1
        if r.reason is Reason.READING_MISMATCH and r.reading is not None:
            rk = reading_kind(r.reading, row.reading)
            reading[rk] += 1
            code = pseudonym.get(r.photo_name)
            family = "digital" if code in digital else "ciclometrico"
            if rk == "numero_de_digitos":
                digits[f"{family}|{digit_detail(r.reading, row.reading)}"] += 1
            sample.append({"pseudonimo": code, "categoria": f"leitura_{rk}"})
    print("Número divergente, por tipo:", dict(serial))
    print("Número divergente, esperado com letra minúscula:", dict(case))
    print("Número divergente x professor no mesmo recorte:", dict(teacher))
    print("Professor no número de série do lote inteiro, por status:", dict(lot_status))
    print("Leitura divergente, por tipo:", dict(reading))
    print("\nNúmero divergente x professor x leitura do display:")
    for k, v in sorted(swap.items()):
        print(f"  {k}: {v}")
    print("Leitura com número de dígitos diferente, por família:", dict(digits))
    # Amostra para inspeção visual, só na pasta de trabalho: até 10 por categoria
    df = pd.DataFrame(sample).dropna()
    picked = (df.sample(frac=1, random_state=0).groupby("categoria").head(VISUAL_PER_GROUP)
              .sort_values(["categoria", "pseudonimo"]))
    picked.to_csv(work / VISUAL_NAME, index=False)
    print(f"Amostra visual: {len(picked)} recortes em {FOLDER}/{VISUAL_NAME}")

def severity(read: str, expected: str) -> str:
    """Gravidade do erro do leitor quando a placa mostra o valor do CSV."""
    d = edit_distance(read.upper(), expected.upper())
    return "leve_1" if d == 1 else ("medio_2" if d == 2 else "grosseiro_3_ou_mais")


def inspection() -> None:
    import pandas as pd

    from meter_reader.config import get_work_dir
    from meter_reader.control_sheet import read_control
    from meter_reader.extraction import read_extraction
    from meter_reader.pairing import index_by_photo
    from training.group_dataset import LABEL_VERSIONS, load_group_photos

    work = get_work_dir() / FOLDER
    marks = pd.read_csv(work / "inspecao_visual.csv", dtype=str)
    rows, _ = read_control(next((work / "saida").glob("*/Controle_*.xlsx")))
    by_name = {r.photo_name: r for r in rows}
    index = index_by_photo(read_extraction(next((work / "fotos").glob("BaseExtracao_*.csv"))))
    name = {p.pseudonym: p.path.name
            for p in load_group_photos(LABEL_VERSIONS["r4"]) if p.lot == LOT}
    sample = pd.read_csv(work / VISUAL_NAME, dtype=str)
    print(f"Marcados: {len(marks)} de {len(sample)}")
    pd.set_option("display.width", 200)
    print(pd.crosstab(marks["categoria"], marks["marca"], margins=True,
                      margins_name="total").to_string())
    sev = Counter()
    for m in marks[marks["marca"] == "leitor_errou_professor_igual_csv"].itertuples(index=False):
        n = name[m.pseudonimo]
        ctrl, row = by_name[n], index[n]
        if m.categoria.startswith("numero_"):
            sev[f"numero|{severity(ctrl.meter_number or '', row.expected_meter_number)}"] += 1
        else:
            sev[f"leitura|{severity(str(ctrl.reading), str(row.reading))}"] += 1
    print("\nLeitor errou com a placa igual ao CSV, por gravidade:", dict(sev))

def prepare() -> None:
    from meter_reader.config import get_work_dir
    from training.group_dataset import LABEL_VERSIONS, load_group_photos

    work = get_work_dir() / FOLDER
    if work.exists():
        raise SystemExit("A pasta do ensaio já existe: rode 'limpar' antes de preparar de novo.")
    lot_dir = _lot_dir(load_group_photos(LABEL_VERSIONS["r4"]))
    photos, out = work / "fotos", work / "saida"
    photos.mkdir(parents=True)
    out.mkdir()
    files = [f for f in lot_dir.iterdir() if f.is_file()
             and (f.suffix.lower() in (".jpg", ".jpeg", ".png") or f.name.startswith("BaseExtracao_"))]
    for f in files:
        shutil.copy2(f, photos / f.name)   # cópia: o original na pasta de dados não muda
    bases = sorted(photos.glob("BaseExtracao_*.csv"))
    if len(bases) != 1:
        raise SystemExit("O lote do ensaio não tem um BaseExtracao único.")
    print(f"Copiados {len(files) - 1} arquivos de imagem e o BaseExtracao para {FOLDER}/fotos.")
    print("\nRode, no mesmo terminal:")
    print(f'$env:METER_READER_OUTPUT_DIR = "{out}"')
    print(f'meter-reader analisar --fotos "{photos}" --base "{bases[0]}"')


def check() -> None:
    import pandas as pd

    from meter_reader.config import get_work_dir
    from meter_reader.contracts import Status
    from meter_reader.control_sheet import read_control
    from training.group_dataset import LABEL_VERSIONS, load_group_photos
    from training.reader_dataset import CYCLOMETRIC, DIGITAL, SERIAL, load_items
    from training.reader_function_gold import GOLD_NAME
    from training.reader_heads_train import FUNCTION_CLASSES, function_class
    from training.reader_labels import CROPS_NAME

    work_dir = get_work_dir()
    controls = sorted((work_dir / FOLDER / "saida").glob("*/Controle_*.xlsx"))
    if len(controls) != 1:
        raise SystemExit("A planilha de controle do ensaio não foi encontrada.")
    rows, executions = read_control(controls[0])
    photos = [p for p in load_group_photos(LABEL_VERSIONS["r4"]) if p.lot == LOT]
    pseudonym = {p.path.name: p.pseudonym for p in photos}

    print("Status:", dict(Counter(r.status.value for r in rows)))
    print("Motivos:", dict(Counter(r.reason.value for r in rows if r.reason is not None)))
    print("Versões:", executions[-1].model_versions)

    items, _ = load_items()
    items = items[(items["lote"] == LOT) & items["usa_texto"]]
    text_labels, serial_labels = {}, {}
    for r in items.itertuples(index=False):
        if r.classe in (DIGITAL, CYCLOMETRIC):
            text_labels.setdefault((r.pseudonimo, r.classe), []).append((r.texto, r.origem))
        elif r.classe == SERIAL:
            serial_labels.setdefault(r.pseudonimo, []).append((r.texto, r.origem))
    gold_df = pd.read_csv(work_dir / GOLD_NAME, dtype=str, keep_default_na=False)
    gold = {g.pseudonimo: (g.status, FUNCTION_CLASSES[function_class(g.funcao)])
            for g in gold_df.itertuples(index=False)}

    crops = pd.read_csv(work_dir / CROPS_NAME, usecols=["pseudonimo", "classe"])
    digital = set(crops.loc[crops["classe"] == DIGITAL, "pseudonimo"])
    approved = []
    for r in rows:
        if r.status is not Status.APPROVED:
            continue
        code = pseudonym.get(r.photo_name)
        if code is None:
            continue
        cls = DIGITAL if code in digital else CYCLOMETRIC
        approved.append({"pseudonimo": code, "classe": cls, "leitura": r.reading,
                         "numero": r.meter_number})
    print("\nConferência das Aprovadas (lote de treino: só alarme):")
    for k, v in tally(approved, text_labels, serial_labels, gold).items():
        print(f"  {k}: {v}")


def clean() -> None:
    from meter_reader.config import get_work_dir

    work = get_work_dir() / FOLDER
    if not work.exists():
        print("Nada a apagar.")
        return
    shutil.rmtree(work)
    print(f"Pasta {FOLDER} apagada (cópias e saída do ensaio).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ensaio de ponta a ponta no lote 22/05.")
    parser.add_argument("comando", choices=["preparar", "conferir", "diagnosticar", "inspecao",
                                            "limpar"])
    {"preparar": prepare, "conferir": check, "diagnosticar": diagnose, "inspecao": inspection,
     "limpar": clean}[parser.parse_args().comando]()


if __name__ == "__main__":
    main()
