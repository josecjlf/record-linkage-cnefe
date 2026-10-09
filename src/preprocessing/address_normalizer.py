"""Contrato semantico da normalizacao, independente de DataFrame."""

from __future__ import annotations

import re
from types import MappingProxyType


_ACCENTED_CHARACTERS = "áàâãäéèêëíìîïóòôõöúùûüçñýÿ"
_ASCII_EQUIVALENTS = "aaaaaeeeeiiiiooooouuuucnyy"
_ACCENT_TRANSLATION = str.maketrans(
    _ACCENTED_CHARACTERS, _ASCII_EQUIVALENTS
)
_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")
_DIGITS_ONLY = re.compile(r"[^0-9]+")

NO_NUMBER_MARKERS = frozenset({"sn", "s n", "sem numero"})

# Aplicado somente quando o valor representa um tipo de logradouro isolado.
STREET_TYPE_ALIASES = MappingProxyType(
    {
        "r": "rua",
        "rua": "rua",
        "av": "avenida",
        "avd": "avenida",
        "avenida": "avenida",
        "tv": "travessa",
        "trav": "travessa",
        "travessa": "travessa",
        "rod": "rodovia",
        "rodovia": "rodovia",
        "estr": "estrada",
        "estrada": "estrada",
        "faz": "fazenda",
        "fazenda": "fazenda",
        "sit": "sitio",
        "sitio": "sitio",
        "pc": "praca",
        "pca": "praca",
        "praca": "praca",
    }
)

# Os termos adicionais foram observados como primeiro token da base de busca.
KNOWN_STREET_TYPES = frozenset(
    {
        *STREET_TYPE_ALIASES.values(),
        "acesso",
        "alameda",
        "assentamento",
        "beco",
        "caminho",
        "comunidade",
        "condominio",
        "conjunto",
        "distrito",
        "eixo",
        "entrada",
        "ladeira",
        "largo",
        "loteamento",
        "ponta",
        "povoado",
        "quadra",
        "ramal",
        "regiao",
        "via",
        "vila",
    }
)

# Aplicado apenas a complementos. Nao usamos estas trocas no nome da rua.
COMPLEMENT_TOKEN_ALIASES = MappingProxyType(
    {
        "qd": "quadra",
        "lt": "lote",
        "prox": "proximo",
    }
)


def normalize_text(value: str | None) -> str | None:
    """Normaliza texto em ASCII minusculo sem corrigir palavras."""
    if value is None:
        return None
    normalized = str(value).lower().translate(_ACCENT_TRANSLATION)
    normalized = _NON_ALPHANUMERIC.sub(" ", normalized)
    normalized = " ".join(normalized.split())
    return normalized or None


def normalize_street_name(value: str | None) -> str | None:
    """Normaliza o nome sem expandir abreviacoes potencialmente ambiguas."""
    return normalize_text(value)


def normalize_street_type(value: str | None) -> str | None:
    """Canoniza abreviacoes somente em um campo conhecido como tipo."""
    normalized = normalize_text(value)
    if normalized is None:
        return None
    return STREET_TYPE_ALIASES.get(normalized, normalized)


def split_street_type(value: str | None) -> tuple[str | None, str | None]:
    """Separa um tipo inicial conhecido sem adivinhar tokens desconhecidos."""
    normalized = normalize_street_name(value)
    if normalized is None:
        return None, None
    first_token, separator, remainder = normalized.partition(" ")
    canonical_type = STREET_TYPE_ALIASES.get(first_token, first_token)
    if canonical_type not in KNOWN_STREET_TYPES:
        return None, normalized
    return canonical_type, remainder or None


def normalize_number(value: str | None) -> str | None:
    """Canoniza marcador sem numero e zeros iniciais de numeros puros."""
    normalized = normalize_text(value)
    if normalized is None:
        return None
    if normalized in NO_NUMBER_MARKERS:
        return "sn"
    if normalized.isdigit():
        return normalized.lstrip("0") or "0"
    return normalized


def has_without_number_marker(*values: str | None) -> bool:
    """Detecta somente marcadores explicitos de ausencia de numero."""
    return any(normalize_text(value) in NO_NUMBER_MARKERS for value in values)


def comparable_number(
    number: str | None, *number_context: str | None
) -> str | None:
    """Cria numero comparavel sem apagar a versao literal normalizada."""
    if has_without_number_marker(number, *number_context):
        return "sn"
    return normalize_number(number)


def normalize_postal_code(value: str | None) -> str | None:
    """Remove formatacao e aceita somente CEP com oito digitos."""
    if value is None:
        return None
    digits = _DIGITS_ONLY.sub("", str(value))
    return digits if len(digits) == 8 else None


def normalize_municipality_code(value: str | None) -> str | None:
    """Produz os seis digitos comparaveis, removendo o DV do codigo IBGE."""
    if value is None:
        return None
    digits = _DIGITS_ONLY.sub("", str(value))
    if len(digits) == 7:
        return digits[:6]
    if len(digits) == 6:
        return digits
    return None


def normalize_neighborhood(value: str | None) -> str | None:
    return normalize_text(value)


def normalize_complement(value: str | None) -> str | None:
    """Expande apenas tokens inequivocos observados em complementos."""
    normalized = normalize_text(value)
    if normalized is None:
        return None
    return " ".join(
        COMPLEMENT_TOKEN_ALIASES.get(token, token)
        for token in normalized.split()
    )


def build_normalized_address(*components: str | None) -> str | None:
    """Compoe uma chave auditavel sem substituir os componentes estruturados."""
    normalized_parts = [normalize_text(value) for value in components]
    joined = " ".join(part for part in normalized_parts if part)
    return joined or None
