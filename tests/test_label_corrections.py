"""Testes da aplicação das correções da revisão R3 (T4f.8)."""

import pytest

from training.group_dataset import GroupLabelFileError
from training.label_corrections import apply_corrections

ROWS = [
    {"pseudonimo": "F-00000000000a", "lote": "2026-05-20", "conjunto": "treino",
     "pareada": "True", "grupo": "dig1_quadrado_branco", "condicao": "raiz"},
    {"pseudonimo": "F-00000000000b", "lote": "2026-05-21", "conjunto": "treino",
     "pareada": "True", "grupo": "indeterminado", "condicao": "raiz"},
]


def item(code, current, final):
    return {"pseudonimo": code, "grupo_atual": current[0], "condicao_atual": current[1],
            "grupo_final": final[0], "condicao_final": final[1]}


def test_applies_changes_and_keeps_confirmed_rows():
    review = [item("F-00000000000a", ("dig1_quadrado_branco", "raiz"), ("indeterminado", "raiz")),
              item("F-00000000000b", ("indeterminado", "raiz"), ("indeterminado", "raiz"))]
    rows, changed = apply_corrections(ROWS, review)
    assert changed == 1
    assert rows[0]["grupo"] == "indeterminado" and rows[1]["grupo"] == "indeterminado"
    assert ROWS[0]["grupo"] == "dig1_quadrado_branco"  # a lista original não é alterada


@pytest.mark.parametrize("review_item, message", [
    (item("F-00000000000z", ("indeterminado", "raiz"), ("multiplos", "raiz")), "fora dos rótulos"),
    (item("F-00000000000a", ("multiplos", "raiz"), ("indeterminado", "raiz")), "diferente do arquivo"),
    (item("F-00000000000a", ("dig1_quadrado_branco", "raiz"), ("multiplos", "visor")), "inexistente"),
])
def test_refuses_inconsistent_review(review_item, message):
    with pytest.raises(GroupLabelFileError, match=message):
        apply_corrections(ROWS, [review_item])
