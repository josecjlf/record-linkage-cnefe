"""Aplicacao do contrato de normalizacao a base de busca Pandas."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from src.ingestion.schemas import SEARCH_COLUMNS
from src.ingestion.validation import (
    DatasetValidationError,
    validate_required_columns,
)
from src.preprocessing.address_normalizer import (
    build_normalized_address,
    comparable_number,
    has_without_number_marker,
    normalize_complement,
    normalize_municipality_code,
    normalize_neighborhood,
    normalize_number,
    normalize_postal_code,
    normalize_street_name,
    split_street_type,
)


SEARCH_NORMALIZED_COLUMNS = (
    "municipio_codigo_normalizado",
    "cep_normalizado",
    "bairro_normalizado",
    "tipo_logradouro_normalizado",
    "nome_logradouro_normalizado",
    "logradouro_texto_normalizado",
    "logradouro_normalizado",
    "numero_normalizado",
    "numero_ausente",
    "sem_numero",
    "numero_comparavel_normalizado",
    "complemento_normalizado",
    "endereco_normalizado",
)


def _nullable_string(value: object) -> str | None:
    return None if pd.isna(value) else str(value)


def _build_nullable_address(*values: object) -> str | None:
    return build_normalized_address(*(_nullable_string(value) for value in values))


def _normalize_series(
    series: pd.Series, normalizer: Callable[[str | None], str | None]
) -> pd.Series:
    def normalize_scalar(value: object) -> str | None:
        return normalizer(_nullable_string(value))

    return series.map(normalize_scalar).astype("string")


def normalize_search_addresses(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Adiciona campos normalizados sem alterar as colunas de entrada."""
    validate_required_columns(
        dataframe.columns, SEARCH_COLUMNS, "base de busca"
    )
    reserved_columns = sorted(set(dataframe.columns) & set(SEARCH_NORMALIZED_COLUMNS))
    if reserved_columns:
        raise DatasetValidationError(
            "Base de busca possui colunas reservadas para normalizacao: "
            f"{', '.join(reserved_columns)}. Renomeie essas colunas na entrada "
            "para preservar seus valores originais."
        )
    result = dataframe.copy(deep=True)

    result["municipio_codigo_normalizado"] = _normalize_series(
        result["consulta_municipio"], normalize_municipality_code
    )
    result["cep_normalizado"] = _normalize_series(
        result["consulta_cep"], normalize_postal_code
    )
    result["bairro_normalizado"] = _normalize_series(
        result["consulta_bairro"], normalize_neighborhood
    )
    result["logradouro_texto_normalizado"] = _normalize_series(
        result["consulta_logradouro"], normalize_street_name
    )

    street_parts = result["consulta_logradouro"].map(
        lambda value: split_street_type(_nullable_string(value))
    )
    result["tipo_logradouro_normalizado"] = pd.Series(
        [street_type for street_type, _ in street_parts],
        index=result.index,
        dtype="string",
    )
    result["nome_logradouro_normalizado"] = pd.Series(
        [street_name for _, street_name in street_parts],
        index=result.index,
        dtype="string",
    )
    result["logradouro_normalizado"] = pd.Series(
        [
            _build_nullable_address(street_type, street_name)
            if not pd.isna(street_type)
            else street_text
            for street_type, street_name, street_text in zip(
                result["tipo_logradouro_normalizado"],
                result["nome_logradouro_normalizado"],
                result["logradouro_texto_normalizado"],
                strict=True,
            )
        ],
        index=result.index,
        dtype="string",
    )

    result["numero_normalizado"] = _normalize_series(
        result["consulta_numero"], normalize_number
    )
    result["numero_ausente"] = result["numero_normalizado"].isna()
    result["sem_numero"] = pd.Series(
        [
            has_without_number_marker(
                _nullable_string(number), _nullable_string(complement)
            )
            for number, complement in zip(
                result["consulta_numero"],
                result["consulta_complemento"],
                strict=True,
            )
        ],
        index=result.index,
        dtype="bool",
    )
    result["numero_comparavel_normalizado"] = pd.Series(
        [
            comparable_number(
                _nullable_string(number), _nullable_string(complement)
            )
            for number, complement in zip(
                result["consulta_numero"],
                result["consulta_complemento"],
                strict=True,
            )
        ],
        index=result.index,
        dtype="string",
    )
    result["complemento_normalizado"] = _normalize_series(
        result["consulta_complemento"], normalize_complement
    )
    result["endereco_normalizado"] = pd.Series(
        [
            _build_nullable_address(
                municipality,
                postal_code,
                street,
                number,
                neighborhood,
                complement,
            )
            for municipality, postal_code, street, number, neighborhood, complement
            in zip(
                result["municipio_codigo_normalizado"],
                result["cep_normalizado"],
                result["logradouro_normalizado"],
                result["numero_comparavel_normalizado"],
                result["bairro_normalizado"],
                result["complemento_normalizado"],
                strict=True,
            )
        ],
        index=result.index,
        dtype="string",
    )
    return result
