"""Piloto do professor da rotulagem do leitor (T6a.3).

Sorteia recortes do desenvolvimento a partir das caixas que o leitor recebe
(select_fields no corte adotado, predições fora da dobra da semente 0 do S1_coco),
envia ao professor, aplica as regras de aceitação da T6a.1 e monta a fila de
julgamento. Composição fixa por classe (D5), sorteio simples com sementes em texto,
e uma subamostra repetida para medir a estabilidade. Tudo fica na pasta de
trabalho, só com pseudônimos e coordenadas; as saídas impressas são contagens.

Uso, a partir da raiz do repositório do produto:
    python -m training.reader_pilot sortear
    python -m training.reader_pilot enviar [--modelo M] [--limite N]
    python -m training.reader_pilot relatorio [--modelo M]
    python -m training.reader_pilot diagnostico [--modelo M]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from meter_reader.fields import select_fields
from training.box_sample import shuffled
from training.build_manifest import file_sha256

PILOT_COUNTS = {"display_digital": 80, "display_ciclometrico": 70, "numero_serie": 50}
STABILITY_N = 30  # recortes enviados duas vezes (D5)
AUDIT_N = 20      # aceitos sorteados para a auditoria (D5)
PILOT_NAME = "amostra_piloto_professor.csv"
PILOT_SHA256 = "5668a3edad4c906662141e319a369386f319b70cb78f03a0f9a6fcb778622a91"
PILOT_COLUMNS = ("ordem", "pseudonimo", "classe", "x0", "y0", "x1", "y1", "estabilidade")
RESPONSES_DIR = Path("experimentos") / "leitor" / "piloto"
AUDIT_QUEUE_NAME = "fila_auditoria_piloto.csv"
AUDIT_JUDGMENT_NAME = "julgamento_auditoria_piloto.csv"
QUEUE_COLUMNS = ("item", "pseudonimo", "classe", "x0", "y0", "x1", "y1", "rotacao", "opcoes")

# ---------------------------------------------------------------------
# Amostra do piloto
# ---------------------------------------------------------------------

def candidates(photos, preds: list[dict], cut: float) -> pd.DataFrame:
    """Uma linha por campo que o leitor receberia: consumo (digital ou ciclométrico)
    e número de série, com a caixa escolhida, a 0,1 px."""
    rows = []
    for photo, pred in zip(photos, preds):
        sel = select_fields(pred["boxes"], pred["scores"], pred["labels"], cut)
        chosen = []
        if sel.consumption is not None:
            cls = "display_digital" if sel.family == "digital" else "display_ciclometrico"
            chosen.append((cls, sel.consumption))
        if sel.serial is not None:
            chosen.append(("numero_serie", sel.serial))
        for cls, box in chosen:
            x0, y0, x1, y1 = (round(float(v), 1) for v in box)
            rows.append({"pseudonimo": photo.pseudonym, "classe": cls,
                         "x0": x0, "y0": y0, "x1": x1, "y1": y1})
    return pd.DataFrame(rows, columns=["pseudonimo", "classe", "x0", "y0", "x1", "y1"])


def draw(cands: pd.DataFrame, counts: dict = PILOT_COUNTS,
         stability_n: int = STABILITY_N) -> pd.DataFrame:
    """Sorteio por classe, marca de estabilidade e ordem de envio, todos fixos."""
    parts = []
    for cls, n in counts.items():
        pool = cands[cands["classe"] == cls].set_index("pseudonimo")
        if len(pool) < n:
            raise ValueError(f"{cls}: só {len(pool)} candidatos para {n}.")
        codes = shuffled(pool.index, f"piloto:{cls}")[:n]
        parts.append(pool.loc[codes].reset_index())
    sample = pd.concat(parts, ignore_index=True)
    keys = sample["pseudonimo"] + "|" + sample["classe"]
    repeated = set(shuffled(keys, "piloto:estabilidade")[:stability_n])
    sample["estabilidade"] = keys.isin(repeated)
    order = {k: i + 1 for i, k in enumerate(shuffled(keys, "piloto:ordem"))}
    sample["ordem"] = keys.map(order)
    return sample.sort_values("ordem")[list(PILOT_COLUMNS)].reset_index(drop=True)


def save_or_check(sample: pd.DataFrame, work_dir) -> str:
    """Grava a amostra; se já existir, confere que é idêntica ao sorteio."""
    path = work_dir / PILOT_NAME
    if path.exists():
        saved = pd.read_csv(path, float_precision="round_trip")
        if not saved.equals(sample.astype(saved.dtypes.to_dict())):
            raise ValueError("A amostra gravada difere do sorteio; nada foi alterado.")
        print("Amostra já gravada; conferida: idêntica ao sorteio.")
    else:
        sample.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
        print("Amostra gravada.")
    return file_sha256(path)


def load_sample(work_dir) -> pd.DataFrame:
    """Amostra gravada, conferida pelo SHA-256 do sorteio."""
    if file_sha256(work_dir / PILOT_NAME) != PILOT_SHA256:
        raise SystemExit("A amostra do piloto não é a sorteada (SHA-256 diferente).")
    return pd.read_csv(work_dir / PILOT_NAME, float_precision="round_trip")


# ---------------------------------------------------------------------
# Pedidos e respostas gravadas
# ---------------------------------------------------------------------

def responses_path(model: str) -> Path:
    """Respostas do piloto de um modelo, relativo à pasta de trabalho."""
    return RESPONSES_DIR / f"respostas_{model}.jsonl"


def queue_name(model: str) -> str:
    """Fila de julgamento de um modelo, na pasta de trabalho."""
    return f"fila_julgamento_piloto_{model}.csv"


def pilot_jobs(sample: pd.DataFrame) -> list[tuple[dict, int]]:
    """Pedidos do piloto: todos os recortes uma vez, na ordem de envio, e depois os da
    estabilidade uma segunda vez (repetição 2)."""
    rows = sample.sort_values("ordem").to_dict("records")
    return [(r, 1) for r in rows] + [(r, 2) for r in rows if r["estabilidade"]]


def done_keys(path: Path) -> set[tuple[str, str, int]]:
    """Pedidos já gravados (pseudônimo, classe, repetição): o envio retoma de onde parou."""
    if not path.exists():
        return set()
    keys = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            keys.add((rec["pseudonimo"], rec["classe"], rec["repeticao"]))
    return keys


def load_records(path: Path) -> dict:
    """Respostas gravadas, revalidadas pelas regras atuais, por (pseudônimo, classe,
    repetição)."""
    from training.teacher import revalidate

    recs = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            recs[(r["pseudonimo"], r["classe"], r["repeticao"])] = revalidate(r["classe"], r)
    return recs


def csv_rows(photos) -> dict:
    """Linha do BaseExtracao de cada foto, pelo nome exato, lida do lote da própria foto."""
    from meter_reader.extraction import read_extraction
    from meter_reader.pairing import index_by_photo

    rows, cache = {}, {}
    for p in photos:
        lot_dir = p.path.parent
        if lot_dir not in cache:
            files = sorted(lot_dir.glob("BaseExtracao_*.csv"))
            if len(files) != 1:
                raise SystemExit("Lote sem um BaseExtracao único.")
            cache[lot_dir] = index_by_photo(read_extraction(files[0]))
        row = cache[lot_dir].get(p.path.name)
        if row is not None:
            rows[p.pseudonym] = row
    return rows


def judgment_queue(df: pd.DataFrame, audit_n: int = AUDIT_N) -> pd.DataFrame:
    """Recortes a julgar: todos os não aceitos e `audit_n` aceitos sorteados, misturados
    numa ordem aleatória fixa (o julgamento não sabe quais são da auditoria)."""
    from training.teacher_eval import ACCEPTED

    keys = df["pseudonimo"] + "|" + df["classe"]
    audit = set(shuffled(keys[df["status"] == ACCEPTED], "piloto:auditoria")[:audit_n])
    queue = df[(df["status"] != ACCEPTED) | keys.isin(audit)].copy()
    qkeys = keys[queue.index]
    queue["auditoria"] = qkeys.isin(audit)
    order = {k: i + 1 for i, k in enumerate(shuffled(qkeys, "piloto:julgamento"))}
    queue["ordem_julgamento"] = qkeys.map(order)
    cols = ["ordem_julgamento", "pseudonimo", "classe", "x0", "y0", "x1", "y1",
            "status", "auditoria"]
    return queue.sort_values("ordem_julgamento")[cols].reset_index(drop=True)


def _context(work_dir, sample: pd.DataFrame) -> tuple[dict, dict]:
    """Fotos da amostra (por pseudônimo) e as linhas do BaseExtracao delas."""
    from training.detector_dataset import load_detector_photos

    codes = set(sample["pseudonimo"])
    photos = {p.pseudonym: p for p in load_detector_photos() if p.pseudonym in codes}
    csv = csv_rows(list(photos.values()))
    if set(csv) != codes:
        raise SystemExit("Há fotos do piloto sem linha no BaseExtracao.")
    return photos, csv


# ---------------------------------------------------------------------
# Comandos
# ---------------------------------------------------------------------

def sortear() -> None:
    from meter_reader.config import get_work_dir
    from training.crop_margin import dev_predictions

    photos, preds, cut = dev_predictions()
    cands = candidates(photos, preds, cut)
    print("Candidatos por classe:", cands["classe"].value_counts().to_dict())
    sample = draw(cands)
    print("Sorteados por classe:", sample["classe"].value_counts().to_dict(),
          "; estabilidade:", int(sample["estabilidade"].sum()))
    print("SHA-256 da amostra:", save_or_check(sample, get_work_dir()))


def enviar(limit: int | None, model: str) -> None:
    """Envia ao professor os pedidos ainda não gravados, até `limit` novos."""
    from anthropic import Anthropic
    from dotenv import load_dotenv
    from PIL import Image

    from meter_reader.config import get_work_dir
    from training.detector_dataset import load_detector_photos
    from training.teacher import ask, prepare_crop

    load_dotenv()  # ANTHROPIC_API_KEY, do .env do produto
    work_dir = get_work_dir()
    sample = load_sample(work_dir)
    paths = {p.pseudonym: p.path for p in load_detector_photos()}
    out = work_dir / responses_path(model)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = done_keys(out)
    client = Anthropic()
    sent = errors = tokens_in = tokens_out = 0
    for row, rep in pilot_jobs(sample):
        if (row["pseudonimo"], row["classe"], rep) in done:
            continue
        if limit is not None and sent >= limit:
            break
        box = (row["x0"], row["y0"], row["x1"], row["y1"])
        with Image.open(paths[row["pseudonimo"]]) as image:
            png = prepare_crop(image, box, row["classe"])
        rec = {"pseudonimo": row["pseudonimo"], "classe": row["classe"], "repeticao": rep,
               "modelo_pedido": model,
               "data": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               **ask(client, png, row["classe"], model)}
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        sent += 1
        errors += rec["erro"] is not None
        tokens_in += rec["tokens_entrada"]
        tokens_out += rec["tokens_saida"]
    print(f"Enviados agora: {sent}; com erro de formato: {errors}; tokens de entrada "
          f"{tokens_in}, de saída {tokens_out}; gravados no total: {len(done_keys(out))} de "
          f"{len(pilot_jobs(sample))}")


def relatorio(model: str) -> None:
    """Status de cada recorte (regras da T6a.1), medidas do piloto e fila de julgamento.
    Imprime só contagens."""
    from meter_reader.config import get_work_dir
    from training.teacher_eval import (ACCEPTED, FORMAT_ERROR, STATUSES,
                                       decimal_drum_suspect, label_status, stable)

    work_dir = get_work_dir()
    sample = load_sample(work_dir)
    recs = load_records(work_dir / responses_path(model))
    if len(recs) != len(pilot_jobs(sample)):
        raise SystemExit("Piloto incompleto: rode o envio de novo.")
    _, csv = _context(work_dir, sample)

    rows = []
    for s in sample.to_dict("records"):
        rec = recs[(s["pseudonimo"], s["classe"], 1)]
        row = csv[s["pseudonimo"]]
        resp = rec["resposta"] or {}
        rows.append({**s,
                     "status": label_status(s["classe"], rec, row.expected_meter_number,
                                            row.reading),
                     "tambor_decimal": s["classe"] == "display_ciclometrico"
                                       and decimal_drum_suspect(rec, row.reading),
                     "erro": rec["erro"], "funcao": resp.get("funcao"),
                     "duas_leituras": resp.get("duas_leituras")})
    df = pd.DataFrame(rows)

    table = pd.crosstab(df["classe"], df["status"]).reindex(columns=list(STATUSES), fill_value=0)
    table["aceitacao"] = (table[ACCEPTED] / table[list(STATUSES)].sum(axis=1)).round(3)
    print(table.to_string(), "\n")
    errs = df[df["status"] == FORMAT_ERROR].groupby(["classe", "erro"]).size()
    print("Erros de formato por tipo:\n" + (errs.to_string() if len(errs) else "nenhum"), "\n")
    dig = df[(df["classe"] == "display_digital") & df["funcao"].notna()]
    print("Funções lidas nos digitais:",
          dig["funcao"].replace("", "(vazia)").value_counts().to_dict())
    print("Digitais com duas leituras:", int((dig["duas_leituras"] == True).sum()))  # noqa: E712
    print("Ciclométricos com suspeita de tambor decimal:", int(df["tambor_decimal"].sum()))
    pairs = [(recs[(s["pseudonimo"], s["classe"], 1)], recs[(s["pseudonimo"], s["classe"], 2)])
             for s in sample.to_dict("records") if s["estabilidade"]]
    print(f"Estabilidade: {sum(stable(a, b) for a, b in pairs)} de {len(pairs)} idênticas")

    queue = judgment_queue(df)
    queue.to_csv(work_dir / queue_name(model), index=False, encoding="utf-8",
                 lineterminator="\n")
    print(f"Fila de julgamento: {len(queue)} recortes "
          f"({int(queue['auditoria'].sum())} da auditoria), gravada na pasta de trabalho")


def diagnostico(model: str) -> None:
    """Diagnóstico do piloto, sem custo: aceitação simples e ponderada (pesos da amostra
    das caixas), aceitação entre os que o professor julgou legíveis, anatomia das
    discordâncias e forma mascarada dos erros de formato. Só contagens."""
    from collections import Counter

    from meter_reader.config import get_work_dir
    from training.teacher_eval import (ACCEPTED, DISAGREE, FORMAT_ERROR, ILLEGIBLE,
                                       KIND_ORDER, QUEUE_F2, discord_kind, label_status,
                                       mask, reading_values)

    work_dir = get_work_dir()
    sample = load_sample(work_dir)
    recs = {(k[0], k[1]): v for k, v in load_records(work_dir / responses_path(model)).items()
            if k[2] == 1}
    photos, csv = _context(work_dir, sample)

    rows, formats = [], Counter()
    for s in sample.to_dict("records"):
        rec, row = recs[(s["pseudonimo"], s["classe"])], csv[s["pseudonimo"]]
        status = label_status(s["classe"], rec, row.expected_meter_number, row.reading)
        kind = None
        if status in (DISAGREE, QUEUE_F2):
            resp = rec["resposta"]
            if s["classe"] == "numero_serie":
                kind = discord_kind(resp["numero"], row.expected_meter_number.upper())
            elif s["classe"] == "display_digital" and resp["funcao"] not in ("03", ""):
                kind = "outra_funcao"
            else:
                exp = str(row.reading)
                kind = min((discord_kind(str(v), exp) for v in reading_values(resp["leitura"])),
                           key=KIND_ORDER.index)
        if status == FORMAT_ERROR:
            raw = rec["resposta_bruta"] or {}
            field = "numero" if s["classe"] == "numero_serie" else "leitura"
            formats[(s["classe"], mask(raw.get(field, "(sem resposta)")))] += 1
        rows.append({"classe": s["classe"], "status": status, "tipo": kind,
                     "peso": photos[s["pseudonimo"]].weight})
    df = pd.DataFrame(rows)

    print("Aceitação por classe:")
    for cls, g in df.groupby("classe"):
        acc = g["status"] == ACCEPTED
        legible = ~g["status"].isin([ILLEGIBLE, FORMAT_ERROR])
        ill = g["status"] == ILLEGIBLE
        print(f"  {cls}: simples {acc.mean():.1%}; ponderada "
              f"{np.average(acc, weights=g['peso']):.1%}; entre legíveis pelo professor "
              f"{acc[legible].mean():.1%} ({int(acc.sum())} de {int(legible.sum())}); "
              f"ilegíveis simples {ill.mean():.1%}, ponderada "
              f"{np.average(ill, weights=g['peso']):.1%}")
    print("\nAnatomia das discordâncias:")
    print(pd.crosstab(df["classe"], df["tipo"]).to_string())
    print("\nForma dos erros de formato (9 = dígito, A = letra):")
    if not formats:
        print("  nenhum")
    for (cls, shape), n in sorted(formats.items()):
        print(f"  {cls}: '{shape}' x{n}")

def candidate_text(field_class: str, resp: dict) -> dict:
    """Campos de texto de uma resposta validada: o que o julgamento confere."""
    from training.teacher import TEXT_FIELDS

    return {name: resp[name] for name in TEXT_FIELDS[field_class]}


def auditoria(model: str) -> None:
    """Fila de auditoria: os recortes aceitos (R1 a R3) do modelo no piloto, com a leitura
    do professor como única opção, em ordem aleatória fixa. Fica na pasta de trabalho."""
    from meter_reader.config import get_work_dir
    from training.teacher_eval import ACCEPTED, label_status

    work_dir = get_work_dir()
    sample = load_sample(work_dir)
    recs = load_records(work_dir / responses_path(model))
    _, csv = _context(work_dir, sample)
    rows = []
    for s in sample.to_dict("records"):
        rec, row = recs[(s["pseudonimo"], s["classe"], 1)], csv[s["pseudonimo"]]
        if label_status(s["classe"], rec, row.expected_meter_number, row.reading) != ACCEPTED:
            continue
        resp = rec["resposta"]
        rows.append({"pseudonimo": s["pseudonimo"], "classe": s["classe"],
                     "x0": s["x0"], "y0": s["y0"], "x1": s["x1"], "y1": s["y1"],
                     "rotacao": resp["rotacao"],
                     "opcoes": json.dumps([candidate_text(s["classe"], resp)],
                                          ensure_ascii=False)})
    queue = pd.DataFrame(rows)
    keys = queue["pseudonimo"] + "|" + queue["classe"]
    order = {k: i + 1 for i, k in enumerate(shuffled(keys, "piloto:auditoria_aceitos"))}
    queue["item"] = keys.map(order)
    queue = queue.sort_values("item")[list(QUEUE_COLUMNS)]
    queue.to_csv(work_dir / AUDIT_QUEUE_NAME, index=False, encoding="utf-8", lineterminator="\n")
    print(f"Fila de auditoria: {len(queue)} recortes aceitos; por classe "
          f"{queue['classe'].value_counts().to_dict()}")

def _number(text: str) -> str:
    """Leitura sem os zeros à esquerda (comparação numérica, spec 13.2)."""
    return text.lstrip("0") or "0"


def audit_result(option: dict, decision: str, value: str) -> str:
    """Resultado de um item da auditoria: correta, só zeros à esquerda errados, valor
    errado, parcial ou ilegível."""
    if decision == "opcao_1":
        return "correta"
    if decision != "outra":
        return decision
    judged = json.loads(value)
    same = all(_number(option[k]) == _number(judged[k]) if k == "leitura"
               else option[k] == judged[k] for k in option)
    return "so_zeros" if same else "valor_errado"


def resumo_auditoria() -> None:
    """Resumo da auditoria dos rótulos aceitos do piloto. Só contagens."""
    from meter_reader.config import get_work_dir

    work_dir = get_work_dir()
    queue = pd.read_csv(work_dir / AUDIT_QUEUE_NAME, dtype=str, keep_default_na=False)
    judged = pd.read_csv(work_dir / AUDIT_JUDGMENT_NAME, dtype=str, keep_default_na=False)
    if len(judged) != len(queue):
        raise SystemExit("Auditoria incompleta.")
    m = queue.merge(judged, on=["item", "pseudonimo", "classe"], validate="one_to_one")
    m["resultado"] = [audit_result(json.loads(o)[0], d, v)
                      for o, d, v in zip(m["opcoes"], m["decisao"], m["valor"])]
    print(pd.crosstab(m["classe"], m["resultado"]).to_string(), "\n")
    n = len(m)
    print(f"Erro de valor: {int((m['resultado'] == 'valor_errado').sum())} de {n}; "
          f"erro só de zeros à esquerda: {int((m['resultado'] == 'so_zeros').sum())} de {n}; "
          f"rotação errada: {int((m['rotacao_errada'] == 'True').sum())} de {n}")
    from training.teacher_eval import mask

    for r in m[m["resultado"] == "valor_errado"].itertuples():
        opt, val = json.loads(r.opcoes)[0], json.loads(r.valor)
        diff = [k for k in opt if (_number(opt[k]) != _number(val[k]) if k == "leitura"
                                   else opt[k] != val[k])]
        shape = lambda d: {k: (d[k] if k == "funcao" else mask(d[k])) for k in diff}
        print(f"  item {r.item} ({r.classe}): campos diferentes {diff}; "
              f"professor {shape(opt)}; julgado {shape(val)}")


def cruzado(model_a: str, model_b: str) -> None:
    """Status de cada recorte do piloto pelos dois modelos, cruzados por classe. Mostra,
    sem custo, o que o modelo B rende onde o A não teve rótulo aceito."""
    from meter_reader.config import get_work_dir
    from training.teacher_eval import label_status

    work_dir = get_work_dir()
    sample = load_sample(work_dir)
    ra = load_records(work_dir / responses_path(model_a))
    rb = load_records(work_dir / responses_path(model_b))
    _, csv = _context(work_dir, sample)
    rows = []
    for s in sample.to_dict("records"):
        key, row = (s["pseudonimo"], s["classe"], 1), csv[s["pseudonimo"]]
        rows.append({"classe": s["classe"],
                     "a": label_status(s["classe"], ra[key], row.expected_meter_number, row.reading),
                     "b": label_status(s["classe"], rb[key], row.expected_meter_number, row.reading)})
    df = pd.DataFrame(rows)
    for cls, g in df.groupby("classe"):
        print(f"\n{cls} (linhas: {model_a}; colunas: {model_b})")
        print(pd.crosstab(g["a"], g["b"]).to_string())

def main() -> None:
    from training.teacher import MODEL

    parser = argparse.ArgumentParser(description="Piloto do professor do leitor (T6a.3).")
    sub = parser.add_subparsers(dest="comando", required=True)
    sub.add_parser("sortear", help="sorteia e grava a amostra do piloto")
    sub.add_parser("resumo_auditoria", help="resume a auditoria dos rótulos aceitos")
    cross = sub.add_parser("cruzado", help="cruza os status de dois modelos no piloto")
    cross.add_argument("--modelo-a", default=MODEL)
    cross.add_argument("--modelo-b", default="claude-sonnet-5-5")
    for name, text in (("enviar", "envia ao professor os pedidos pendentes"),
                       ("relatorio", "aplica as regras e grava a fila de julgamento"),
                       ("diagnostico", "diagnóstico do piloto, sem custo"),
                       ("auditoria", "grava a fila de auditoria dos recortes aceitos")):
        p = sub.add_parser(name, help=text)
        p.add_argument("--modelo", default=MODEL, help="identificador do modelo do professor")
        if name == "enviar":
            p.add_argument("--limite", type=int, default=None, help="no máximo N pedidos novos")
    args = parser.parse_args()
    if args.comando == "sortear":
        sortear()
    elif args.comando == "resumo_auditoria":
        resumo_auditoria()
    elif args.comando == "cruzado":
        cruzado(args.modelo_a, args.modelo_b)
    elif args.comando == "enviar":
        enviar(args.limite, args.modelo)
    elif args.comando == "relatorio":
        relatorio(args.modelo)
    elif args.comando == "diagnostico":
        diagnostico(args.modelo)
    else:
        auditoria(args.modelo)

if __name__ == "__main__":
    main()
