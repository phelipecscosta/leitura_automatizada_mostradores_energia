"""Pesos COCO do SSDlite para a ablação do S1 (protocolo, 1.4; ET5c, T5c.1).

A ablação carrega no S1 só as camadas do SSDlite do torchvision treinado no COCO
com o mesmo nome e a mesma forma, nos grupos escolhidos na revisão R4. Este módulo
mede essa compatibilidade, por grupo de camadas, e faz a carga parcial.
Nenhum dado do cliente é usado: só as formas das âncoras, para montar o S1.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping

import torch
from torch import nn
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights

COCO_WEIGHTS = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1

# Grupos de camadas, pelo início do nome no state_dict do SSD do torchvision
GROUPS = (("espinha", "backbone.features."),
          ("extras", "backbone.extra."),
          ("cabeca_classificacao", "head.classification_head."),
          ("cabeca_regressao", "head.regression_head."))
GROUP_NAMES = tuple(name for name, _ in GROUPS)


def group_of(key: str) -> str:
    """Grupo de uma camada pelo nome; 'outros' se não pertencer a nenhum."""
    for name, prefix in GROUPS:
        if key.startswith(prefix):
            return name
    return "outros"


def compare_state(target: Mapping[str, torch.Tensor],
                  source: Mapping[str, torch.Tensor]) -> dict[str, Counter]:
    """Compara os tensores do modelo (target) com os do COCO (source), por grupo.

    Contagens: 'iguais' (mesmo nome e forma), 'forma' (mesmo nome, forma diferente),
    'ausentes' (nome inexistente no COCO); e os números de valores
    ('valores_iguais' e 'valores_total'), para medir a fração carregável.
    """
    out: dict[str, Counter] = {}
    for key, tensor in target.items():
        count = out.setdefault(group_of(key), Counter())
        count["valores_total"] += tensor.numel()
        if key not in source:
            count["ausentes"] += 1
        elif source[key].shape != tensor.shape:
            count["forma"] += 1
        else:
            count["iguais"] += 1
            count["valores_iguais"] += tensor.numel()
    return out


ABLATION_GROUP = "espinha"  # opção (a) da revisão R4: só o prefixo contíguo da espinha


def prefix_keys(target: Mapping[str, torch.Tensor], source: Mapping[str, torch.Tensor],
                group: str = ABLATION_GROUP) -> list[str]:
    """Nomes do grupo, na ordem da rede, até a primeira camada ausente no COCO ou de
    forma diferente (opção a da R4). Depois dela, nada é carregado, mesmo com forma
    igual, porque a camada passaria a receber entradas que não vêm do COCO."""
    keys = []
    for key, tensor in target.items():
        if group_of(key) != group:
            continue
        if key not in source or source[key].shape != tensor.shape:
            break
        keys.append(key)
    return keys


def load_prefix(model: nn.Module, source: Mapping[str, torch.Tensor]) -> list[str]:
    """Copia para o modelo o prefixo contíguo da espinha do COCO e devolve os nomes, na
    ordem da rede. O resto fica como está: no S1 da ablação, o ImageNet no restante da
    espinha e a inicialização do S1 nos extras e nas cabeças."""
    state = model.state_dict()
    keys = prefix_keys(state, source)
    if not keys:
        raise ValueError("Nenhuma camada do COCO encaixa no início da espinha.")
    model.load_state_dict({k: source[k] for k in keys}, strict=False)
    return keys


def coco_state() -> dict[str, torch.Tensor]:
    """Pesos do SSDlite COCO do torchvision (baixados para o cache do PyTorch na 1ª vez)."""
    return COCO_WEIGHTS.get_state_dict(progress=False)


def main() -> None:
    """Mede a compatibilidade com o S1 montado com as âncoras gravadas: só contagens e
    nomes de camadas da rede (nenhum dado do cliente)."""
    from meter_reader.config import get_work_dir
    from training.run_detector import candidate, load_shapes

    shapes, _ = load_shapes(get_work_dir())
    target = candidate("S1").build(shapes["S1"]["formas"]).state_dict()
    source = coco_state()
    for name, count in compare_state(target, source).items():
        frac = count["valores_iguais"] / count["valores_total"]
        print(f"{name}: iguais {count['iguais']}, forma diferente {count['forma']}, "
              f"ausentes {count['ausentes']}; valores carregáveis {frac:.1%}")
    # Primeiras camadas de forma diferente por grupo, para localizar a causa
    shown = Counter()
    for key, tensor in target.items():
        group = group_of(key)
        if key in source and source[key].shape != tensor.shape and shown[group] < 3:
            shown[group] += 1
            print(f"  [{group}] {key}: S1 {tuple(tensor.shape)} × COCO {tuple(source[key].shape)}")
    keys = prefix_keys(target, source)
    print(f"Prefixo carregado na ablação (opção a): {len(keys)} tensores; "
          f"primeira camada fora dele: {next(k for k in target if group_of(k) == ABLATION_GROUP and k not in keys)}")


if __name__ == "__main__":
    main()
