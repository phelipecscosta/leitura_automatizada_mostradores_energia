"""Métricas do D3, bootstrap pareado e registro das execuções.

Protocolo do Lab02, seções 2.6 e 2.7: métrica otimizadora = média da AP de
"outros" e da AP de "ilegível", só na amostra aleatória (o estresse fica
fora); intervalos e comparações por bootstrap sobre as fotos.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from meter_reader.triage import SCENE_CLASSES
from training.build_sample import ROLE_ZERO
from training.labels import MASKED, SCENE_METER_ONLY, LabeledPhoto

OUTROS = SCENE_CLASSES.index("outros")
METER_SLOTS = [SCENE_CLASSES.index("digital"), SCENE_CLASSES.index("ciclometrico")]
EPS = 1e-7


def targets(photos: list[LabeledPhoto]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(cena, ilegível, máscara da amostra aleatória) na ordem de `photos`."""
    scene = np.array([p.scene for p in photos])
    illegible = np.array([p.illegible for p in photos])
    random_sample = np.array([p.role != ROLE_ZERO for p in photos])
    return scene, illegible, random_sample


def _defined(scene, illegible) -> bool:
    """A AP só existe com positivos e negativos nas duas tarefas."""
    is_outros = scene == OUTROS
    meters = illegible != MASKED
    ill = illegible[meters] == 1
    return 0 < is_outros.sum() < len(scene) and 0 < ill.sum() < len(ill)


def optimizing_metric(p_scene, p_illegible, scene, illegible) -> tuple[float, float, float]:
    """(métrica otimizadora, AP de outros, AP de ilegível)."""
    ap_outros = average_precision_score(scene == OUTROS, p_scene[:, OUTROS])
    meters = illegible != MASKED
    ap_ill = average_precision_score(illegible[meters] == 1, p_illegible[meters])
    return (ap_outros + ap_ill) / 2, float(ap_outros), float(ap_ill)


def log_losses(p_scene, p_illegible, scene, illegible) -> tuple[float, float]:
    """Perda logarítmica sem pesos: (cena com regra parcial, legibilidade em medidores)."""
    exact = scene != SCENE_METER_ONLY
    p_true = np.where(exact, p_scene[np.arange(len(scene)), np.clip(scene, 0, None)],
                      p_scene[:, METER_SLOTS].sum(axis=1))
    scene_ll = float(-np.log(np.clip(p_true, EPS, 1)).mean())
    meters = illegible != MASKED
    y, p = illegible[meters], np.clip(p_illegible[meters], EPS, 1 - EPS)
    leg_ll = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    return scene_ll, leg_ll


def paired_bootstrap(model_a, model_b, scene, illegible, n_boot=2000, seed=0):
    """Diferença (A - B) da métrica otimizadora: (pontual, IC 2,5%, IC 97,5%, válidas).

    `model_a` e `model_b` são pares (p_cena, p_ilegível) nas mesmas fotos.
    """
    point = optimizing_metric(*model_a, scene, illegible)[0] - optimizing_metric(*model_b, scene, illegible)[0]
    rng = np.random.default_rng(seed)
    diffs = []
    for idx in rng.integers(0, len(scene), size=(n_boot, len(scene))):
        if not _defined(scene[idx], illegible[idx]):
            continue
        a = optimizing_metric(model_a[0][idx], model_a[1][idx], scene[idx], illegible[idx])[0]
        b = optimizing_metric(model_b[0][idx], model_b[1][idx], scene[idx], illegible[idx])[0]
        diffs.append(a - b)
    low, high = np.percentile(diffs, [2.5, 97.5])
    return float(point), float(low), float(high), len(diffs)


def bootstrap_ci(p_scene, p_illegible, scene, illegible, n_boot=2000, seed=0):
    """IC de 95% da métrica otimizadora de um modelo: (IC 2,5%, IC 97,5%, válidas)."""
    rng = np.random.default_rng(seed)
    values = [
        optimizing_metric(p_scene[idx], p_illegible[idx], scene[idx], illegible[idx])[0]
        for idx in rng.integers(0, len(scene), size=(n_boot, len(scene)))
        if _defined(scene[idx], illegible[idx])
    ]
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high), len(values)


def _product_commit() -> str:
    """Commit do produto que gerou a execução (rastreabilidade peso -> código)."""
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() or "desconhecido"


def record_run(work_dir: Path, name: str, config: dict, metrics: dict,
               photos: list[LabeledPhoto], p_scene, p_illegible) -> Path:
    """Grava o JSON da execução e o CSV das predições em <pasta de trabalho>/experimentos/."""
    folder = work_dir / "experimentos"
    folder.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = folder / f"{stamp}_{name}"

    record = {"nome": name, "data": stamp, "commit_produto": _product_commit(),
              "configuracao": config, "metricas": metrics}
    base.with_suffix(".json").write_text(json.dumps(record, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    header = "pseudonimo,lote,papel," + ",".join(f"p_{c}" for c in SCENE_CLASSES) + ",p_ilegivel"
    lines = [header] + [
        f"{p.pseudonym},{p.lot},{p.role}," + ",".join(f"{v:.6f}" for v in ps) + f",{pi:.6f}"
        for p, ps, pi in zip(photos, p_scene, p_illegible)
    ]
    base.with_suffix(".csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return base.with_suffix(".json")
