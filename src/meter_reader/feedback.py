"""Resumo em texto do processamento (spec v14, 2.4, ação 6).

Conta as linhas do processamento atual (as imagens lidas nesta execução) e
monta o texto exibido ao utilizador. Nenhum nome de foto aparece no resumo:
ele pode ser copiado e enviado por outros canais, e os nomes identificam o
cliente (E18). As cópias antigas (opção c) aparecem só como contagem por pasta;
os caminhos ficam num arquivo dentro da pasta do lote (folders.write_stale_list).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from meter_reader.contracts import ControlRow, Status

# Linhas do resumo, na ordem da spec (2.4, ação 6.2, itens 2 a 9)
SUMMARY_LINES = (
    (Status.APPROVED, "Imagens de medidores aprovadas com sucesso"),
    (Status.NEEDS_REVIEW, "Imagens de medidores que precisam de verificação"),
    (Status.REJECTED_LOW_QUALITY, "Imagens de medidores rejeitadas por baixa qualidade"),
    (Status.REJECTED_MULTIPLE, "Imagens rejeitadas por múltiplos medidores"),
    (Status.REJECTED_OTHER, "Imagens rejeitadas por contexto diverso (não medidores)"),
    (Status.REJECTED_CORRUPTED, "Imagens rejeitadas por arquivo corrompido ou vazio"),
    (Status.REJECTED_NOT_LISTED, "Imagens rejeitadas por não estarem listadas no BaseExtracao"),
    (Status.NOT_PROCESSED, "Imagens não processadas por não estarem listadas na tabela "
                           "fornecida com os parâmetros"),
)


@dataclass(frozen=True)
class Summary:
    """Contagens de um processamento."""

    total: int
    by_status: dict[Status, int]

    @classmethod
    def from_rows(cls, rows: Iterable[ControlRow]) -> Summary:
        counts = Counter(r.status for r in rows)
        return cls(sum(counts.values()), {s: counts.get(s, 0) for s in Status})


def _number(n: int) -> str:
    return f"{n:,}".replace(",", ".")  # separador de milhar brasileiro


def _percent(part: int, total: int) -> str:
    value = 100 * part / total if total else 0.0
    return f"{value:.1f}%".replace(".", ",")


def format_feedback(summary: Summary, control_path: Path, updated: bool, parameters: str,
                    stale_by_folder: dict[str, int] | None = None,
                    stale_list: str | None = None, provisional: bool = False) -> str:
    """Texto do feedback: mensagem final, resumo estatístico e estado dos Parâmetros."""
    control_path = Path(control_path)
    action = "atualizada" if updated else "gerada"
    lines = [
        "Processamento concluído.",
        f'Planilha "{control_path.name}" {action} no endereço {control_path.parent}.',
        "",
        "Resumo (quantidade e percentual sobre as imagens lidas):",
    ]
    rows = [("Imagens lidas", summary.total)]
    rows += [(label, summary.by_status[status]) for status, label in SUMMARY_LINES]
    width = max(len(label) for label, _ in rows)
    for label, count in rows:
        lines.append(f"  {label.ljust(width)}  {_number(count):>7}  "
                     f"{_percent(count, summary.total):>6}")
    lines += ["", f"Validação por Parâmetros: {parameters}"]
    if provisional:
        lines.append("Leitura: provisória. Toda imagem que chega ao leitor vai para "
                     "verificação; nenhuma é aprovada automaticamente.")
    if stale_by_folder:
        total = sum(stale_by_folder.values())
        subject = ("1 imagem deste processamento tem" if total == 1
                   else f"{_number(total)} imagens deste processamento têm")
        lines += ["", f"Atenção: {subject} uma cópia antiga em outra pasta do lote. "
                      "Nenhuma foi apagada:"]
        lines += [f"  {folder}: {_number(n)}" for folder, n in sorted(stale_by_folder.items())]
        if stale_list:
            lines.append(f"Lista com as pastas e os nomes: {stale_list}, na pasta do lote.")
    return "\n".join(lines)
