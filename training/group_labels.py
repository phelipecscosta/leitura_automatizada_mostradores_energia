"""Rótulos da rotulagem completa (ET4c) a partir das pastas de cluster_Rotulado.

Cada foto está numa pasta de grupo e, opcionalmente, numa subpasta de
condição (02_dataset.md, seção 4). Este módulo traduz o nome das pastas no
disco para códigos canônicos curtos (T4e.1, opção N-a) e liga cada grupo à
sua classe do produto (E49). A classe não é gravada no arquivo de rótulos:
ela sai sempre daqui, que é a fonte única.
"""

from __future__ import annotations

import unicodedata

from meter_reader.contracts import PRODUCT_CLASSES

# Grupo canônico -> (nome por extenso, classe do produto)
GROUPS = {
    "cic_disco": ("Ciclométrico - Disco", "ciclometrico"),
    "cic_duas_leituras": ("Ciclométrico - Duas leituras", "ciclometrico"),
    "cic_uma_leitura": ("Ciclométrico - Uma leitura", "ciclometrico"),
    "dig1_quadrado_branco": ("Digital 1 - Quadrado - Branco", "digital"),
    "dig2_redondo_branco": ("Digital 2 - Redondo - Branco", "digital"),
    "dig3_vidro_redondo": ("Digital 3 - Vidro redondo", "digital"),
    "dig4_verde_arduino": ("Digital 4 - Verde arduino", "digital"),
    "dig5_pequenino": ("Digital 5 - Pequenino", "digital"),
    "indeterminado": ("Medidor - Indeterminado", "indeterminado"),
    "multiplos": ("Múltiplos", "multiplos"),
    "out_acesso": ("Outros - Acesso, portas, portões, bilhetes, rua", "outros"),
    "out_sem_acesso": ("Outros - Sem acesso - grade, porteiras", "outros"),
    "out_sem_contexto": ("Outros - Sem contexto", "outros"),
    "out_poste": ("Outros - Poste somente", "outros"),
}

# Condição canônica -> nome por extenso (tabela 4.3)
CONDITIONS = {
    "raiz": "Raiz",
    "caixa": "Caixa",
    "parede": "Parede",
    "visor": "Visor",
    "caixa_visor": "Caixa e Visor",
    "distante_parede": "Distante_Parede",
    "distante_poste": "Distante_Poste",
    "circulo": "Círculo",
    "grade": "Grade - Obstruído",
}

# Pasta no disco (relativa a cluster_Rotulado, com "/") -> (grupo, condição)
FOLDERS = {
    "Ciclométrico - disco": ("cic_disco", "raiz"),
    "Ciclométrico - disco/Caixa e Visor - Ciclométrico - disco": ("cic_disco", "caixa_visor"),
    "Ciclométrico - disco/Distante_Parede - Ciclométrico": ("cic_disco", "distante_parede"),
    "Ciclométrico - Duas leituras": ("cic_duas_leituras", "raiz"),
    "Ciclométrico - Uma leitura": ("cic_uma_leitura", "raiz"),
    "Ciclométrico - Uma leitura/Caixa - Ciclométrico - quadrado - branco": ("cic_uma_leitura", "caixa"),
    "Ciclométrico - Uma leitura/Parede - Ciclométrico - quadrado - branco": ("cic_uma_leitura", "parede"),
    "Ciclométrico - Uma leitura/Visor - ciclométrico": ("cic_uma_leitura", "visor"),
    "Digital 1 - Quadrado - Branco": ("dig1_quadrado_branco", "raiz"),
    "Digital 1 - Quadrado - Branco/Caixa - Digital 1": ("dig1_quadrado_branco", "caixa"),
    "Digital 1 - Quadrado - Branco/Circulo": ("dig1_quadrado_branco", "circulo"),
    "Digital 1 - Quadrado - Branco/Distante_Parede - Digital": ("dig1_quadrado_branco", "distante_parede"),
    "Digital 1 - Quadrado - Branco/Distante_poste_digital": ("dig1_quadrado_branco", "distante_poste"),
    "Digital 1 - Quadrado - Branco/Grade - Obstruido": ("dig1_quadrado_branco", "grade"),
    "Digital 1 - Quadrado - Branco/Visor + caixa": ("dig1_quadrado_branco", "caixa_visor"),
    "Digital 1 - Quadrado - Branco/Visor - Digital 1": ("dig1_quadrado_branco", "visor"),
    "Digital 2 - Redondo - Branco": ("dig2_redondo_branco", "raiz"),
    "Digital 2 - Redondo - Branco/Caixa - Digital 2": ("dig2_redondo_branco", "caixa"),
    "Digital 3 - Vidro redondo": ("dig3_vidro_redondo", "raiz"),
    "Digital 4 - Verde arduino": ("dig4_verde_arduino", "raiz"),
    "Digital 5 - Pequenino": ("dig5_pequenino", "raiz"),
    "Medidor - Indeterminado": ("indeterminado", "raiz"),
    "Medidor - Indeterminado/Distante_Parede - indeterminado": ("indeterminado", "distante_parede"),
    "Medidor - Indeterminado/Distante_poste_ideterminado": ("indeterminado", "distante_poste"),
    "Múltiplos": ("multiplos", "raiz"),
    "outros - Acesso, portas, portões, bilhetes, rua": ("out_acesso", "raiz"),
    "outros - sem acesso - grade, porteiras": ("out_sem_acesso", "raiz"),
    "outros - Sem contexto": ("out_sem_contexto", "raiz"),
    "outros_poste_somente": ("out_poste", "raiz"),
}


class LabelFolderError(ValueError):
    """Pasta de cluster_Rotulado sem mapeamento conhecido."""


def _nfc(text: str) -> str:
    """Forma única dos acentos, para comparar nomes vindos do disco."""
    return unicodedata.normalize("NFC", text)


_FOLDERS_NFC = {_nfc(folder): label for folder, label in FOLDERS.items()}


def folder_label(relative_folder: str) -> tuple[str, str]:
    """(grupo, condição) de uma pasta relativa a cluster_Rotulado."""
    try:
        return _FOLDERS_NFC[_nfc(relative_folder)]
    except KeyError:
        # O nome da pasta é um rótulo, e não um dado do cliente: pode aparecer
        raise LabelFolderError(f"Pasta sem mapeamento: {relative_folder}") from None


def product_class(group: str) -> str:
    """Classe do produto de um grupo canônico."""
    return GROUPS[group][1]

# ---------------------------------------------------------------------
# Grupos de treino (protocolo da triagem, seção 1.1, opção U-b)
# ---------------------------------------------------------------------

# Grupos com menos de 30 fotos, fundidos pela descrição. O arquivo de rótulos
# guarda o grupo original; a fusão só acontece no carregamento.
MERGES = {
    "dig5_pequenino": "dig1_quadrado_branco",
    "out_poste": "out_sem_contexto",
}

# Ordem fixa das saídas do modelo: pela classe do produto e, dentro dela, pelo
# código. Mudar esta ordem invalida qualquer modelo treinado.
TRAINING_GROUPS = tuple(sorted(
    (g for g in GROUPS if g not in MERGES),
    key=lambda g: (PRODUCT_CLASSES.index(product_class(g)), g),
))

# Classe do produto (índice em PRODUCT_CLASSES) de cada saída do modelo:
# é o mapa usado para somar as probabilidades dos grupos por classe
TRAINING_GROUP_CLASS = tuple(PRODUCT_CLASSES.index(product_class(g)) for g in TRAINING_GROUPS)


def training_group(group: str) -> str:
    """Grupo de treino de um grupo do arquivo de rótulos (aplica a fusão)."""
    if group not in GROUPS:
        raise ValueError(f"Grupo desconhecido: {group}")
    return MERGES.get(group, group)
