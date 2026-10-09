"""Estrategias exatas ordenadas da baseline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class NumberRequirement(str, Enum):
    ANY = "ANY"
    PRESENT = "PRESENT"
    MISSING_OR_SN = "MISSING_OR_SN"


@dataclass(frozen=True)
class ExactMatchStrategy:
    name: str
    key_pairs: tuple[tuple[str, str], ...]
    number_requirement: NumberRequirement = NumberRequirement.ANY


_MUNICIPALITY = (
    "municipio_codigo_normalizado",
    "municipio_codigo_normalizado",
)
_POSTAL_CODE = ("cep_normalizado", "cep_normalizado")
_STREET = ("logradouro_normalizado", "logradouro_normalizado")
_NUMBER = (
    "numero_comparavel_normalizado",
    "numero_comparavel_normalizado",
)


BASELINE_STRATEGIES = (
    ExactMatchStrategy(
        name="EXACT_MUNICIPIO_CEP_LOGRADOURO_NUMERO",
        key_pairs=(_MUNICIPALITY, _POSTAL_CODE, _STREET, _NUMBER),
        number_requirement=NumberRequirement.PRESENT,
    ),
    ExactMatchStrategy(
        name="EXACT_MUNICIPIO_LOGRADOURO_NUMERO",
        key_pairs=(_MUNICIPALITY, _STREET, _NUMBER),
        number_requirement=NumberRequirement.PRESENT,
    ),
    ExactMatchStrategy(
        name="EXACT_MUNICIPIO_CEP_LOGRADOURO_SEM_NUMERO",
        key_pairs=(_MUNICIPALITY, _POSTAL_CODE, _STREET),
        number_requirement=NumberRequirement.MISSING_OR_SN,
    ),
    ExactMatchStrategy(
        name="EXACT_MUNICIPIO_LOGRADOURO_SEM_NUMERO",
        key_pairs=(_MUNICIPALITY, _STREET),
        number_requirement=NumberRequirement.MISSING_OR_SN,
    ),
)
