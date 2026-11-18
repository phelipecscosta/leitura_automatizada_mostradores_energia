"""Ablação COCO do S1 (protocolo, 1.4; revisão R4; ET5c, T5c.2).

Compara, nas predições fora da dobra do desenvolvimento, o S1 com o prefixo COCO
(S1_coco) e o S1 de base. Regra da seção 1.4: a ablação é adotada se o IC 95% da
diferença da métrica de escolha (média das 3 sementes, bootstrap pareado por foto,
revisão R2) ficar inteiro acima de zero e o candidato continuar viável. O limiar da
porta do vencedor é o menor dos limiares por semente (R2). Lê também os registros
dos dois modelos finais. Saídas: só contagens e medidas; resumo em JSON na pasta de
trabalho. Nenhuma foto do teste é lida.
"""
from __future__ import annotations

import json

import numpy as np

from meter_reader.detector import DETECTOR_CLASSES
from training.detector_dataset import BOXES_SHA256
from training.detector_eval import bootstrap_seed_mean_diff
from training.detector_report import LEGACY_SCORE_THRESH, SEEDS, align, gate_at, seed_result
from training.detector_train import unflatten_predictions

BASE, ABLATION = "S1", "S1_coco"
SUMMARY_NAME = "relatorio_ablacao_coco.json"


def decide(diffs: np.ndarray, viable: bool) -> dict:
    """Regra da seção 1.4 sobre as diferenças (ablação − base) do bootstrap."""
    low, high = (float(np.quantile(diffs, q)) for q in (0.025, 0.975))
    return {"diferenca_media": float(diffs.mean()), "ic95": [low, high], "viavel": viable,
            "adotada": bool(viable and low > 0)}


def load_seeds(work_dir, folder: str, photos, lots: list[str]) -> tuple[list[list[dict]], set]:
    """Predições fora da dobra das 3 sementes, na ordem de `photos`, e os cortes gravados."""
    from training.run_detector import part_path

    seeds, cuts = [], set()
    for seed in SEEDS:
        codes, preds = [], []
        for lot in lots:
            with np.load(part_path(work_dir, folder, lot, seed)) as data:
                meta = json.loads(str(data["meta"]))
                c, p = unflatten_predictions(data)
            if meta["sha_caixas"] != BOXES_SHA256:
                raise SystemExit(f"{folder}: parte treinada com outro gabarito.")
            cuts.add(meta.get("corte_confianca", LEGACY_SCORE_THRESH.get(folder)))
            codes += c
            preds += p
        seeds.append(align(codes, preds, photos))
    return seeds, cuts


def main() -> None:
    from meter_reader.config import get_work_dir
    from training.detector_dataset import load_detector_photos
    from training.run_detector import OUTPUT_FOLDER, final_path

    work_dir = get_work_dir()
    photos = [p for p in load_detector_photos() if p.subset == "desenvolvimento"]
    lots = sorted({p.lot for p in photos})
    summary, results = {}, {}
    for folder in (BASE, ABLATION):
        seeds, cuts = load_seeds(work_dir, folder, photos, lots)
        if len(cuts) != 1:
            raise SystemExit(f"{folder}: partes com cortes de confiança diferentes.")
        res = [seed_result(photos, s) for s in seeds]
        thresholds = [r.threshold for r in res]
        out = {"corte_confianca": cuts.pop(), "viavel": all(t is not None for t in thresholds),
               "selecao_por_semente": [r.selection for r in res],
               "selecao_media": float(np.mean([r.selection for r in res])),
               "ap_media": {c: float(np.mean([r.ap[c] for r in res])) for c in DETECTOR_CLASSES},
               "limiares_por_semente": thresholds}
        if out["viavel"]:
            out["limiar"] = min(thresholds)  # o lado seguro (R2)
            out["porta_por_semente"] = [gate_at(photos, r.preds, out["limiar"]) for r in res]
        summary[folder], results[folder] = out, res

    diffs = bootstrap_seed_mean_diff(photos, [r.matches for r in results[ABLATION]],
                                     [r.matches for r in results[BASE]])
    decision = decide(diffs, summary[ABLATION]["viavel"])
    decision["vencedor"] = ABLATION if decision["adotada"] else BASE
    decision["limiar_porta"] = summary[decision["vencedor"]].get("limiar")

    finals = {}
    for folder in (BASE, ABLATION):
        record = json.loads(final_path(work_dir, folder).with_suffix(".json")
                            .read_text(encoding="utf-8"))
        finals[folder] = {k: record[k] for k in ("epoca_parada", "epocas_escolhidas",
                                                 "fotos", "sha256_pesos", "commit_produto")}

    for folder, s in summary.items():
        print(f"{folder} (corte {s['corte_confianca']}): escolha média "
              f"{100 * s['selecao_media']:.2f}; por semente "
              f"{[round(100 * v, 2) for v in s['selecao_por_semente']]}; viável {s['viavel']}")
        print("  AP média: " + ", ".join(f"{c} {100 * v:.1f}" for c, v in s["ap_media"].items()))
        print(f"  limiares por semente {s['limiares_por_semente']}; limiar {s.get('limiar')}")
        for i, g in enumerate(s.get("porta_por_semente", [])):
            print(f"  s{i}: perda {100 * g['perda']:.3f}% (lim. sup. {100 * g['perda_lim_sup']:.3f}%,"
                  f" {g['fotos_com_campo_perdidas']} foto(s)); cobertura {100 * g['cobertura']:.1f}%")
    print(f"\nDiferença {ABLATION} − {BASE}: {100 * decision['diferenca_media']:+.2f} pontos, "
          f"IC 95% [{100 * decision['ic95'][0]:+.2f}; {100 * decision['ic95'][1]:+.2f}]")
    print(f"Ablação adotada: {decision['adotada']}; vencedor {decision['vencedor']}, "
          f"limiar da porta {decision['limiar_porta']}")
    for folder, f in finals.items():
        print(f"Modelo final {folder}: épocas escolhidas {sorted(f['epocas_escolhidas'])}, "
              f"parada {f['epoca_parada']}, {f['fotos']} fotos, SHA-256 {f['sha256_pesos'][:12]}…, "
              f"commit {f['commit_produto'][:7]}")
    out = {"resumos": summary, "decisao": decision, "modelos_finais": finals}
    (work_dir / OUTPUT_FOLDER / SUMMARY_NAME).write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Resumo gravado em {SUMMARY_NAME}, na pasta de trabalho.")


if __name__ == "__main__":
    main()
