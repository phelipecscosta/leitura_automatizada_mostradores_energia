"""Modelo final do leitor e o seu artefato (ET6c, T6c.4; decisões P-a2, E-b e H-a).

Um único comando, que retoma do ponto em que parou:
1. Pseudo-rótulos finais (P-a2): a interseção das dobras do autotreinamento. Cada
   recorte foi candidato em duas dobras (as que não o avaliavam); entra só o recorte
   aceito nas duas, com o mesmo texto.
2. Troncos finais (E-b): a receita da validação cruzada (L2-12+auto1), agora com os 3
   lotes. 240 épocas com cosseno; a divisão interna de 10% (as mesmas fotos de antes)
   escolhe a época de cada semente.
3. Cabeças finais (H-a): a receita da R12, sobre o tronco congelado de cada semente.
4. Artefato: os 3 troncos, as 3 cabeças, os limiares da T6c.3 e a procedência, num
   arquivo carregável com weights_only=True. Nunca sobrescreve; mostra o SHA-256.
Tudo fica na pasta de trabalho. Só números.

Uso, a partir da raiz do repositório do produto:
    python -m training.reader_final
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

FINAL_LOT = "final"                    # nenhum item pertence a este lote: nada é avaliado
FINAL_CONFIG = "L2-12+auto1-final"
FINAL_HEADS = "L2-12+auto1-r12-final"
CANDIDATE = "L2-12"
SOURCE_PSEUDO = "L2-12"                # pseudo-rótulos da rodada 1 do autotreinamento
FINAL_PSEUDO_NAME = Path("experimentos") / "leitor" / "pseudo" / "final_interseccao.csv"
ARTIFACT_DIR = Path("modelos") / "leitor"
ARTIFACT_NAME = "leitor_l2-12.pt"
ARTIFACT_FORMAT = "leitor-1"
KEY = ["pseudonimo", "classe"]


def intersect_folds(folds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Pseudo-rótulos aceitos nas duas dobras em que o recorte foi candidato, com o
    mesmo texto (P-a2). `folds`: lote avaliado -> pseudo-rótulos da dobra."""
    parts = [df.assign(dobra=lot) for lot, df in folds.items() if len(df)]
    if not parts:
        return pd.DataFrame(columns=["pseudonimo", "classe", "lote", "texto"])
    df = pd.concat(parts, ignore_index=True)
    if (df["lote"] == df["dobra"]).any():
        raise ValueError("Uma dobra tem pseudo-rótulo do próprio lote avaliado.")
    g = df.groupby(KEY).agg(dobras=("dobra", "nunique"), textos=("texto", "nunique"),
                            lote=("lote", "first"), texto=("texto", "first")).reset_index()
    keep = (g["dobras"] == len(folds) - 1) & (g["textos"] == 1)
    return g.loc[keep, ["pseudonimo", "classe", "lote", "texto"]].reset_index(drop=True)


def final_pseudo(work_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Interseção das dobras, gravada uma vez (lida se já existir), e o SHA-256 de cada
    arquivo de origem."""
    from training.build_manifest import file_sha256
    from training.reader_pseudo import PSEUDO_DIR

    paths = sorted((work_dir / PSEUDO_DIR).glob(f"{SOURCE_PSEUDO}_*.csv"))
    if len(paths) != 3:
        raise SystemExit("São esperados os pseudo-rótulos das 3 dobras.")
    shas = {p.stem.split("_", 1)[1]: file_sha256(p) for p in paths}
    out = work_dir / FINAL_PSEUDO_NAME
    if out.exists():
        return pd.read_csv(out, dtype={"texto": str}, keep_default_na=False), shas
    folds = {p.stem.split("_", 1)[1]: pd.read_csv(p, dtype={"texto": str},
                                                   keep_default_na=False) for p in paths}
    inter = intersect_folds(folds)
    inter.to_csv(out, index=False)
    union = pd.concat(folds.values()).drop_duplicates(KEY)
    print(f"Pseudo-rótulos (P-a2): {len(inter)} na interseção, de {len(union)} na união; "
          "por classe:", inter["classe"].value_counts().to_dict(), flush=True)
    return inter, shas


def build_artifact(trunks: list[dict], heads: list[dict], thresholds: dict,
                   info: dict) -> dict:
    """Conteúdo do artefato; só tipos simples fora dos tensores (weights_only=True)."""
    from training.reader_heads_train import FUNCTION_CLASSES

    if not (len(trunks) == len(heads) == 3):
        raise ValueError("São esperados 3 troncos e 3 cabeças.")
    needed = ("d1_logit", "limiar_inferior_log", "temperatura_texto", "limiar_funcao")
    if any(k not in thresholds for k in needed):
        raise ValueError("Faltam limiares da T6c.3.")
    return {"formato": ARTIFACT_FORMAT, "candidato": CANDIDATE,
            "classes_funcao": list(FUNCTION_CLASSES),
            "troncos": trunks, "cabecas": heads,
            "limiares": {k: float(v) for k, v in thresholds.items()
                         if isinstance(v, (int, float))},
            "info": info}


def save_artifact(path: Path, data: dict) -> str:
    """Gravação atômica que nunca sobrescreve. Devolve o SHA-256 do arquivo."""
    import torch

    from training.build_manifest import file_sha256

    path = Path(path)
    if path.exists():
        raise FileExistsError("Artefato do leitor já gravado.")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(data, tmp)
    os.replace(tmp, path)
    return file_sha256(path)


def main() -> None:
    import torch

    from meter_reader.config import get_work_dir
    from training.build_manifest import file_sha256
    from training.reader_cache import CACHE_DIR, CACHE_NAME
    from training.reader_dataset import load_items
    from training.reader_function_r12 import extended_items
    from training.reader_heads_train import HEADS_DIR, train_heads
    from training.reader_labels import CROPS_SHA256
    from training.reader_pseudo import as_items
    from training.reader_thresholds import OUT_NAME as THRESHOLDS_NAME
    from training.reader_train import META_NAME, SEEDS, WEIGHTS_NAME, run_dir, run_grid

    work_dir = get_work_dir()
    artifact = work_dir / ARTIFACT_DIR / ARTIFACT_NAME
    if artifact.exists():
        raise SystemExit("O artefato do leitor já foi gravado e não é sobrescrito.")
    thresholds_path = work_dir / THRESHOLDS_NAME
    thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))

    # 1. Pseudo-rótulos finais
    items, _ = load_items()
    pseudo, pseudo_shas = final_pseudo(work_dir)
    train_items = pd.concat([items, as_items(pseudo)], ignore_index=True)

    # 2. Troncos finais (pula os já concluídos)
    run_grid({FINAL_CONFIG: {"candidato": CANDIDATE}}, train_items, lots=[FINAL_LOT],
             items_for=lambda lot: train_items)

    # 3. Cabeças finais
    head_items = extended_items()
    cache = work_dir / CACHE_DIR / CACHE_NAME
    for seed in SEEDS:
        trunk_dir = run_dir(work_dir, FINAL_CONFIG, FINAL_LOT, seed)
        out_dir = work_dir / HEADS_DIR / FINAL_HEADS / f"{FINAL_LOT}_s{seed}"
        if (out_dir / "meta.json").exists():
            print(f"Cabeças s{seed}: já concluído")
            continue
        m = train_heads(trunk_dir / WEIGHTS_NAME, CANDIDATE, False, head_items, FINAL_LOT,
                        seed, cache, CROPS_SHA256, out_dir)
        print(f"Cabeças s{seed}: função época {m['epoca_funcao']} (revocação média "
              f"{m['revocacao_media_interna']:.3f}); duas leituras época {m['epoca_duas']} "
              f"(AP {m['ap_interno']:.3f}); {m['segundos'] / 60:.1f} min", flush=True)

    # 4. Artefato
    trunks, heads, epochs, commits = [], [], [], set()
    for seed in SEEDS:
        trunk_dir = run_dir(work_dir, FINAL_CONFIG, FINAL_LOT, seed)
        meta = json.loads((trunk_dir / META_NAME).read_text(encoding="utf-8"))
        heads_dir = work_dir / HEADS_DIR / FINAL_HEADS / f"{FINAL_LOT}_s{seed}"
        hmeta = json.loads((heads_dir / "meta.json").read_text(encoding="utf-8"))
        trunks.append(torch.load(trunk_dir / WEIGHTS_NAME, map_location="cpu",
                                 weights_only=True))
        heads.append(torch.load(heads_dir / "pesos_cabecas.pt", map_location="cpu",
                                weights_only=True))
        epochs.append({"semente": seed, "epoca_tronco": int(meta["epoca_escolhida"]),
                       "ma_interna": float(meta["ma_interna"]),
                       "epoca_funcao": int(hmeta["epoca_funcao"]),
                       "epoca_duas": int(hmeta["epoca_duas"])})
        commits.add(str(meta["commit_produto"]))
    info = {"configuracao": FINAL_CONFIG, "cabecas": FINAL_HEADS, "sementes": list(SEEDS),
            "epocas": epochs, "commit_treino": sorted(commits),
            "lista_recortes_sha256": CROPS_SHA256, "pseudo_rotulos": int(len(pseudo)),
            "pseudo_origem_sha256": pseudo_shas,
            "limiares_sha256": file_sha256(thresholds_path),
            "torch": torch.__version__}
    digest = save_artifact(artifact, build_artifact(trunks, heads, thresholds, info))
    print("Épocas:", epochs)
    print(f"Artefato gravado em modelos/leitor/{ARTIFACT_NAME}, na pasta de trabalho.")
    print(f"SHA-256: {digest}")


if __name__ == "__main__":
    main()
