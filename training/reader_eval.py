"""Avaliação do leitor sobre as predições fora da dobra (protocolo do leitor, seções
1.6 a 1.8 e R2).

Para cada item com texto: a leitura livre (gulosa) e a sua confiança, a
probabilidade do rótulo e a de dois impostores (seção 1.7), pela CTC (soma sobre os
alinhamentos). Só a 0 grau (revisão R7: a escolha da rotação pela confiança trocava
leituras certas por leituras confiantes de recortes invertidos). As notas por item são gravadas na pasta de trabalho. Só números.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from meter_reader.reader_text import TEXT_ALPHABET, encode, greedy_decode, reading_variants

SERIAL = "numero_serie"
# Pares confundíveis da seção 1.7: 0-8, 3-8, 5-6, 6-8, 9-8, 1-7
CONFUSABLE = {"0": "8", "8": "0369", "3": "8", "5": "6", "6": "58", "9": "8", "1": "7",
              "7": "1"}
EVAL_DIR = Path("experimentos") / "leitor" / "avaliacao"
DIGITS = "0123456789"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def sequence_logprobs(log_probs: np.ndarray, sequences: list[str],
                      alphabet: str = TEXT_ALPHABET) -> np.ndarray:
    """log P(sequência) pela CTC, para cada sequência, numa mesma saída (T, C)."""
    if not sequences:
        return np.zeros(0)
    lp = torch.from_numpy(np.asarray(log_probs, dtype=np.float32))
    n = len(sequences)
    targets = [torch.tensor(encode(s, alphabet), dtype=torch.long) for s in sequences]
    lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    loss = F.ctc_loss(lp.unsqueeze(1).expand(-1, n, -1), torch.cat(targets),
                      torch.full((n,), lp.shape[0], dtype=torch.long), lengths,
                      blank=0, reduction="none", zero_infinity=False)
    return -loss.numpy()


def value_logprob(log_probs: np.ndarray, value: str, field_class: str) -> float:
    """log P do valor: na leitura, o máximo entre as formas com zeros (Z-a); no número
    de série, o próprio texto."""
    if field_class == SERIAL:
        return float(sequence_logprobs(log_probs, [value])[0])
    return float(sequence_logprobs(log_probs, reading_variants(int(value))).max())


def same_value(a: str, b: str, field_class: str) -> bool:
    """Igualdade da comparação com o CSV: número inteiro na leitura; texto na série."""
    if field_class == SERIAL:
        return a == b
    return a.isdigit() and b.isdigit() and int(a) == int(b)


def impostors(label: str, field_class: str, key: str) -> tuple[str, str | None]:
    """Dois impostores com um caractere trocado (seção 1.7), com semente pela chave:
    um uniforme e um por par confundível (None se o rótulo não tem dígito confundível).
    Na leitura, a troca é feita no valor sem zeros à esquerda."""
    rng = random.Random(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big"))
    base = label if field_class == SERIAL else str(int(label))
    pos = rng.randrange(len(base))
    pool = DIGITS if base[pos].isdigit() else LETTERS
    uniform = base[:pos] + rng.choice([c for c in pool if c != base[pos]]) + base[pos + 1:]
    if field_class != SERIAL and uniform[0] == "0" and len(uniform) > 1:
        uniform = str(int(uniform))          # zero à esquerda criado pela troca
    spots = [i for i, c in enumerate(base) if c in CONFUSABLE]
    confusable = None
    if spots:
        i = rng.choice(spots)
        confusable = base[:i] + rng.choice(CONFUSABLE[base[i]]) + base[i + 1:]
        if field_class != SERIAL and confusable[0] == "0" and len(confusable) > 1:
            confusable = str(int(confusable))
        if same_value(confusable, base, field_class):
            confusable = None
    return uniform, confusable


def score_item(lp: np.ndarray, label: str, field_class: str, key: str) -> dict:
    """Notas de um item, na leitura a 0 grau (R7)."""
    text = greedy_decode(np.asarray(lp).argmax(-1).tolist(), TEXT_ALPHABET)
    conf = float(sequence_logprobs(lp, [text])[0]) if text else -np.inf
    uniform, confusable = impostors(label, field_class, key)
    return {"livre": text, "conf_livre": conf,
            "acerto": same_value(text, label, field_class),
            "logp_rotulo": value_logprob(lp, label, field_class),
            "imp_uniforme_lido": same_value(text, uniform, field_class),
            "logp_imp_uniforme": value_logprob(lp, uniform, field_class),
            "tem_imp_confundivel": confusable is not None,
            "imp_confundivel_lido": confusable is not None
            and same_value(text, confusable, field_class),
            "logp_imp_confundivel": value_logprob(lp, confusable, field_class)
            if confusable else np.nan}

def score_run(pred_path: Path, items: pd.DataFrame) -> pd.DataFrame:
    """Notas de todos os itens com texto de um treino (um lote avaliado)."""
    pred = np.load(pred_path)
    by_key = items.set_index(items["pseudonimo"] + "|" + items["classe"] + "|"
                             + items["recorte"])
    rows = []
    for k, key in enumerate(pred["chave_texto"]):
        r = by_key.loc[key]
        row = score_item(pred["texto_0"][k], r["texto"], r["classe"], key)
        rows.append({"chave": key, "classe": r["classe"], "origem": r["origem"], **row})
    return pd.DataFrame(rows)


def score_config(work_dir: Path, config: str, items: pd.DataFrame) -> pd.DataFrame:
    """Notas fora da dobra de uma configuração: os 3 lotes de cada semente. Gravadas em
    `avaliacao/<configuração>.csv`; se o arquivo existe, é lido."""
    from training.reader_train import META_NAME, PREDICTIONS_NAME, RUNS_DIR

    out_path = work_dir / EVAL_DIR / f"{config}.csv"
    if out_path.exists():
        return pd.read_csv(out_path, keep_default_na=False, na_values=[""],
                           dtype={"livre": str})
    frames = []
    for run in sorted((work_dir / RUNS_DIR / config).iterdir()):
        if not (run / META_NAME).exists():
            raise SystemExit(f"Treino incompleto em {config}.")
        lot, seed = run.name.rsplit("_s", 1)
        frame = score_run(run / PREDICTIONS_NAME, items)
        frame.insert(0, "semente", int(seed))
        frame.insert(1, "lote", lot)
        frames.append(frame)
    scores = pd.concat(frames, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out_path, index=False)
    return scores


def main() -> None:
    from meter_reader.config import get_work_dir
    from meter_reader.reader import CANDIDATES
    from training.reader_dataset import load_items

    work_dir = get_work_dir()
    items, _ = load_items()
    for config in CANDIDATES:
        scores = score_config(work_dir, config, items)
        table = scores.groupby(["classe", "semente"])["acerto"].mean().unstack().round(3)
        print(f"\n== {config}: acerto exato fora da dobra (leitura livre), por semente ==")
        print(table.to_string())
        print(f"acerto exato geral: {scores['acerto'].mean():.3f}")


if __name__ == "__main__":
    main()
