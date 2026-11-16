"""Conferência das caixas anotadas contra o plano amostral (ET5a, T5.4).

Confere que todas as fotos da amostra (amostra_caixas.csv, SHA-256 fixo) estão
anotadas, com as caixas dentro dos limites de cada foto, e resume as caixas por
conjunto, classe e grupo. Aponta, por regra fixa, as fotos cujas caixas
contradizem o grupo da rotulagem completa: indícios de rótulo de grupo errado ou
de caixa com a classe trocada, a revisar na ferramenta. Saídas: só contagens; a
lista das fotos apontadas (só pseudônimos) vai para a pasta de trabalho.
"""
from __future__ import annotations

import pandas as pd

from training.box_labels import BOX_CLASSES, Box, PhotoRecord

SAMPLE_NAME = "amostra_caixas.csv"
SAMPLE_SHA256 = "65e690f04ed1777490c89f9ba8b978e227bafdeca01833ae8f1b3f908d657256"
FLAGS_NAME = "caixas_divergencias.csv"

# Regras de divergência entre caixas e grupo, fixadas antes de rodar a conferência
RULES = {
    "digital_com_ciclometrico": "grupo digital com caixa de display ciclométrico ou kvarh",
    "ciclometrico_com_digital": "grupo ciclométrico com caixa de display digital",
    "kvarh_fora_duas_leituras": "caixa kvarh fora do grupo de duas leituras",
    "duas_leituras_sem_kvarh": "duas leituras sem caixa kvarh (pode estar encoberto)",
    "determinado_sem_caixas": "medidor de tipo determinado anotado sem caixas",
    # Acrescentadas depois da primeira execução (revisão declarada, ET5a, T5.4)
    "outros_com_caixas": "foto sem medidor (outros) com caixa de algum campo",
    "indeterminado_com_caixas": "indeterminado com caixa: o tipo pode ser distinguível",
}


def flags(group: str, determined: bool, classes: set[str]) -> list[str]:
    """Regras violadas por uma foto, dadas o grupo e as classes das suas caixas."""
    found = []
    if group.startswith("dig") and classes & {"display_ciclometrico", "display_kvarh"}:
        found.append("digital_com_ciclometrico")
    if group.startswith("cic") and "display_digital" in classes:
        found.append("ciclometrico_com_digital")
    if "display_kvarh" in classes and group not in ("cic_duas_leituras", "multiplos"):
        found.append("kvarh_fora_duas_leituras")
    if group == "cic_duas_leituras" and "display_kvarh" not in classes:
        found.append("duas_leituras_sem_kvarh")
    if determined and not classes:
        found.append("determinado_sem_caixas")
    if group.startswith("out") and classes:
        found.append("outros_com_caixas")
    if group == "indeterminado" and classes:
        found.append("indeterminado_com_caixas")
    return found


def per_photo(sample: pd.DataFrame, photos: list[PhotoRecord], boxes: list[Box]) -> pd.DataFrame:
    """Uma linha por foto da amostra: anotada?, número de caixas, uma coluna por classe."""
    frame = sample[["pseudonimo", "conjunto", "grupo", "condicao", "determinado"]].copy()
    annotated = {p.pseudonym for p in photos}
    frame["anotada"] = frame["pseudonimo"].isin(annotated)
    counts = pd.DataFrame([(b.pseudonym, b.cls) for b in boxes], columns=["pseudonimo", "classe"])
    table = (counts.groupby(["pseudonimo", "classe"]).size().unstack(fill_value=0)
             .reindex(columns=list(BOX_CLASSES), fill_value=0))
    frame = frame.merge(table, left_on="pseudonimo", right_index=True, how="left")
    frame[list(BOX_CLASSES)] = frame[list(BOX_CLASSES)].fillna(0).astype(int)
    frame["caixas"] = frame[list(BOX_CLASSES)].sum(axis=1)
    return frame


def class_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Por conjunto e classe: total de caixas e fotos com ao menos uma caixa da classe."""
    rows = []
    for subset, part in frame.groupby("conjunto"):
        for cls in BOX_CLASSES:
            rows.append({"conjunto": subset, "classe": cls, "caixas": int(part[cls].sum()),
                         "fotos": int((part[cls] > 0).sum())})
    return pd.DataFrame(rows).set_index(["conjunto", "classe"])


def empty_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Fotos sem caixas por conjunto e grupo (os rejeitáveis devem concentrá-las)."""
    return (frame.assign(sem_caixas=frame["caixas"] == 0)
            .groupby(["conjunto", "grupo"]).agg(fotos=("pseudonimo", "size"),
                                                sem_caixas=("sem_caixas", "sum")))


def flagged(frame: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por foto anotada com ao menos uma regra violada, para a revisão:
    pseudônimo, conjunto, grupo, condição e regras (separadas por "|")."""
    rows = []
    for row in frame[frame["anotada"]].itertuples(index=False):
        classes = {c for c in BOX_CLASSES if getattr(row, c) > 0}
        found = flags(row.grupo, bool(row.determinado), classes)
        if found:
            rows.append({"pseudonimo": row.pseudonimo, "conjunto": row.conjunto,
                         "grupo": row.grupo, "condicao": row.condicao,
                         "regras": "|".join(found)})
    return pd.DataFrame(rows, columns=["pseudonimo", "conjunto", "grupo", "condicao", "regras"])

def main() -> None:
    # Importações pesadas só aqui: os testes usam as funções acima sem dados reais
    from PIL import Image

    from meter_reader.config import get_work_dir
    from training.box_labels import BOXES_NAME, PHOTOS_NAME, read_labels
    from training.build_manifest import file_sha256
    from training.group_dataset import LABEL_VERSIONS, TEST_LABELS, load_group_photos
    from training.run_final import VERSION

    work_dir = get_work_dir()
    if file_sha256(work_dir / SAMPLE_NAME) != SAMPLE_SHA256:
        raise SystemExit("A amostra difere da gravada na T5.2.")
    sample = pd.read_csv(work_dir / SAMPLE_NAME, dtype={"determinado": bool})
    codes = set(sample["pseudonimo"])
    paths = {p.pseudonym: p.path for labels in (LABEL_VERSIONS[VERSION], TEST_LABELS)
             for p in load_group_photos(labels) if p.pseudonym in codes}
    sizes = {}
    for code, path in paths.items():
        with Image.open(path) as image:  # só o cabeçalho: o tamanho, sem decodificar
            sizes[code] = image.size
    photos, boxes = read_labels(work_dir, sample=codes, sizes=sizes)  # valida os limites

    frame = per_photo(sample, photos, boxes)
    pd.set_option("display.width", 200, "display.max_rows", 200)
    print(f"Fotos da amostra: {len(frame)}; anotadas: {int(frame['anotada'].sum())}; "
          f"caixas: {len(boxes)}; tempo total: {sum(p.seconds for p in photos) / 3600:.1f} h\n")
    print(class_summary(frame), end="\n\n")
    print(empty_summary(frame), end="\n\n")
    marks = flagged(frame)
    per_rule = marks.assign(regra=marks["regras"].str.split("|")).explode("regra")
    print(per_rule.groupby(["conjunto", "regra"]).size().rename("fotos"), end="\n\n")
    print(f"Fotos a revisar: {len(marks)} ({(marks['conjunto'] == 'teste').sum()} no teste)\n")
    marks.to_csv(work_dir / FLAGS_NAME, index=False, encoding="utf-8", lineterminator="\n")
    print(f"SHA-256 de {BOXES_NAME}: {file_sha256(work_dir / BOXES_NAME)}")
    print(f"SHA-256 de {PHOTOS_NAME}: {file_sha256(work_dir / PHOTOS_NAME)}")


if __name__ == "__main__":
    main()

