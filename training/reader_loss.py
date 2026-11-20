"""Perda do leitor: CTC do texto + CTC da função + BCE de duas leituras, somadas com
pesos iguais, cada uma só nos itens em que vale (máscaras; seções 1.3 e 1.12)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _ctc(logits: torch.Tensor, targets: torch.Tensor, lengths: torch.Tensor,
         mask: torch.Tensor) -> torch.Tensor:
    """CTC média nos itens da máscara. Alvos concatenados de todo o lote."""
    offsets = torch.cumsum(lengths, 0) - lengths
    keep = [targets[o:o + n] for o, n, m in zip(offsets.tolist(), lengths.tolist(),
                                                 mask.tolist()) if m]
    log_probs = logits[mask].float().log_softmax(-1).permute(1, 0, 2)   # (T, B, C)
    t = torch.full((log_probs.shape[1],), log_probs.shape[0], dtype=torch.long)
    flat = torch.cat(keep) if keep else targets.new_zeros(0)
    return F.ctc_loss(log_probs, flat, t, lengths[mask], blank=0, reduction="mean",
                      zero_infinity=True)


def reader_loss(out: dict, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
    """Perda total e as parcelas (para o registro)."""
    device = out["texto"].device
    total, parts = torch.zeros((), device=device), {}
    for name in ("texto", "funcao"):
        mask = batch[f"usa_{name}"].to(device)
        if mask.any():
            loss = _ctc(out[name], batch[name].to(device),
                        batch[f"{name}_comprimento"].to(device), mask)
            total, parts[name] = total + loss, float(loss.detach())
    mask = batch["usa_duas"].to(device)
    if mask.any():
        loss = F.binary_cross_entropy_with_logits(out["duas"][mask].float(),
                                                  batch["duas"].to(device)[mask])
        total, parts["duas"] = total + loss, float(loss.detach())
    return total, parts
