"""Leitor do produto: carga do artefato, comitê das 3 sementes e cascata do digital
(spec 2.5, itens 11 a 13; ET6c, T6c.5).

- Comitê: cada semente faz a sua leitura livre (melhor caminho da CTC); entre as não
  vazias, vale a de maior probabilidade média nas 3 sementes (empate: a menor em
  ordem alfabética). É a regra usada na medida dos limiares (T6c.3) e da função
  (R16 a R18).
- Confiança: o log da probabilidade média com T = 1 (`log_conf`), que é a que os
  limiares usam. A temperatura da T6c.3 só calibra a confiança informada
  (`confidence`).
- Digital (D1 e R12): a média dos logits de duas leituras no recorte inteiro decide;
  se dispara, o texto e a função saem da faixa de cima. A função é a média das
  probabilidades das 3 cabeças.
Roda em CPU por padrão, em precisão simples.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from meter_reader.reader import Reader
from meter_reader.reader_heads import Heads, trunk_features
from meter_reader.reader_input import READER_HEIGHT, READER_WIDTH, reader_tensor, upper_band
from meter_reader.reader_text import FUNCTION_CLASSES, TEXT_ALPHABET, encode, greedy_decode
from meter_reader.validation import FieldReading

# SHA-256 do único artefato do leitor aceito por esta versão do produto (T6c.4).
PRODUCT_READER_SHA256 = "3676bb91aba3c59fbc0e866da2ea7e07512d0ea50a2f778c81335d98ad8b3771"
READER_ARTIFACT_FORMAT = "leitor-1"
READER_CANDIDATE = "L2-12"
DIGITAL = "display_digital"
_THRESHOLD_KEYS = ("d1_logit", "limiar_inferior_log", "temperatura_texto", "limiar_funcao")
BATCH = 64


class ReaderArtifactError(ValueError):
    """Artefato do leitor ausente, adulterado ou de outro formato."""


def sequence_logprob(log_probs: torch.Tensor, text: str, temperature: float = 1.0) -> float:
    """log P(texto) pela CTC numa saída (passos, classes) de log-probabilidades. Com
    T != 1, as saídas são divididas por T antes da normalização."""
    if not text:
        return -math.inf
    lp = (log_probs.float() / temperature).log_softmax(-1).unsqueeze(1)   # (passos, 1, C)
    target = torch.tensor([encode(text, TEXT_ALPHABET)], dtype=torch.long)
    loss = F.ctc_loss(lp, target, torch.tensor([lp.shape[0]]), torch.tensor([len(text)]),
                      blank=0, reduction="none", zero_infinity=False)
    return float(-loss[0])


def log_mean_exp(values) -> float:
    """log da média das probabilidades, a partir dos logs."""
    v = np.asarray(values, dtype=float)
    if np.all(np.isneginf(v)):
        return -math.inf
    m = v[np.isfinite(v)].max()
    return float(m + np.log(np.mean(np.exp(v - m))))


def committee_read(lps: list[torch.Tensor]) -> str:
    """Leitura do comitê: a leitura livre de maior probabilidade média (sem as vazias)."""
    reads = {greedy_decode(lp.argmax(-1).tolist(), TEXT_ALPHABET) for lp in lps} - {""}
    if not reads:
        return ""
    conf = {r: log_mean_exp([sequence_logprob(lp, r) for lp in lps]) for r in reads}
    return max(sorted(conf), key=conf.get)


@dataclass(frozen=True)
class ReaderOutput:
    """Saída do leitor num recorte. `function` e `two_lines` só existem no digital."""

    text: str
    log_conf: float             # T = 1: comparada aos limiares
    confidence: float           # calibrada pela temperatura: informada na saída
    function: dict[str, float] | None = None
    two_lines: bool | None = None

    def field(self) -> FieldReading:
        return FieldReading(self.text, self.log_conf)


class ReaderModel:
    """Os 3 troncos e as 3 cabeças do artefato, com os limiares da T6c.3."""

    def __init__(self, trunks: list[Reader], heads: list[Heads], thresholds: dict,
                 device: str = "cpu"):
        if not (len(trunks) == len(heads) == 3):
            raise ReaderArtifactError("O leitor precisa de 3 troncos e 3 cabeças.")
        self.trunks = [t.to(device).eval() for t in trunks]
        self.heads = [h.to(device).eval() for h in heads]
        self.thresholds = {k: float(thresholds[k]) for k in _THRESHOLD_KEYS}
        self.device = device

    @property
    def log_low(self) -> float:
        return self.thresholds["limiar_inferior_log"]

    @property
    def function_threshold(self) -> float:
        return self.thresholds["limiar_funcao"]

    def _forward(self, crops: list[Image.Image]) -> list[dict[str, torch.Tensor]]:
        """Por semente: log-probabilidades do texto, probabilidades da função e logit
        de duas leituras, em CPU."""
        out = [{"texto": [], "funcao": [], "duas": []} for _ in self.trunks]
        with torch.no_grad():
            for start in range(0, len(crops), BATCH):
                x = torch.stack([reader_tensor(c) for c in crops[start:start + BATCH]])
                x = x.to(self.device)
                for k, (trunk, heads) in enumerate(zip(self.trunks, self.heads)):
                    out[k]["texto"].append(trunk(x)["texto"].float().log_softmax(-1).cpu())
                    func, two = heads(trunk_features(trunk, x).float())
                    out[k]["funcao"].append(func.softmax(-1).cpu())
                    out[k]["duas"].append(two.cpu())
        return [{key: torch.cat(v) for key, v in o.items()} for o in out]

    def _text(self, lps: list[torch.Tensor]) -> tuple[str, float, float]:
        text = committee_read(lps)
        if not text:
            return "", -math.inf, 0.0
        t = self.thresholds["temperatura_texto"]
        log_conf = log_mean_exp([sequence_logprob(lp, text) for lp in lps])
        confidence = float(np.mean([math.exp(sequence_logprob(lp, text, t)) for lp in lps]))
        return text, log_conf, confidence

    def read(self, crops: list[Image.Image], field_class: str) -> list[ReaderOutput]:
        """Leitura do comitê de cada recorte de uma mesma classe de campo."""
        if not crops:
            return []
        whole = self._forward(crops)
        if field_class != DIGITAL:
            return [ReaderOutput(*self._text([o["texto"][i] for o in whole]))
                    for i in range(len(crops))]
        logit = torch.stack([o["duas"] for o in whole]).mean(0)
        fired = (logit >= self.thresholds["d1_logit"]).tolist()
        idx = [i for i, f in enumerate(fired) if f]
        upper = self._forward([upper_band(crops[i]) for i in idx]) if idx else None
        position = {i: n for n, i in enumerate(idx)}
        outputs = []
        for i in range(len(crops)):
            src, j = (upper, position[i]) if fired[i] else (whole, i)
            probs = torch.stack([o["funcao"][j] for o in src]).mean(0).tolist()
            outputs.append(ReaderOutput(*self._text([o["texto"][j] for o in src]),
                                        function=dict(zip(FUNCTION_CLASSES, probs)),
                                        two_lines=bool(fired[i])))
        return outputs


def load_reader(path: Path, expected_sha256: str | None = PRODUCT_READER_SHA256,
                device: str = "cpu") -> ReaderModel:
    """Carrega o artefato do leitor, conferindo o SHA-256 (se informado) e o formato."""
    path = Path(path)
    if not path.is_file():
        raise ReaderArtifactError("Artefato do leitor não encontrado.")
    if expected_sha256 is not None:
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
            raise ReaderArtifactError("O artefato do leitor não corresponde ao SHA-256 esperado.")
    data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(data, dict) or data.get("formato") != READER_ARTIFACT_FORMAT:
        raise ReaderArtifactError("Formato do artefato do leitor desconhecido.")
    if data.get("candidato") != READER_CANDIDATE \
            or tuple(data.get("classes_funcao", ())) != FUNCTION_CLASSES:
        raise ReaderArtifactError("O artefato do leitor é de outro modelo.")
    if any(k not in data.get("limiares", {}) for k in _THRESHOLD_KEYS):
        raise ReaderArtifactError("Faltam limiares no artefato do leitor.")
    trunks, heads = [], []
    for trunk_state, heads_state in zip(data["troncos"], data["cabecas"]):
        trunk = Reader(READER_CANDIDATE, pretrained=False)
        trunk.load_state_dict(trunk_state)
        trunk.eval()
        with torch.no_grad():
            dim = trunk_features(trunk, torch.zeros(1, 3, READER_HEIGHT, READER_WIDTH)).shape[-1]
        head = Heads(dim)
        head.load_state_dict(heads_state)
        trunks.append(trunk)
        heads.append(head)
    return ReaderModel(trunks, heads, data["limiares"], device)
