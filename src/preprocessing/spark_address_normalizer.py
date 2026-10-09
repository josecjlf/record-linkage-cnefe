"""Expressoes Spark nativas semanticamente equivalentes ao contrato Python."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pyspark.sql import Column, DataFrame, functions as F

from src.preprocessing.address_normalizer import (
    COMPLEMENT_TOKEN_ALIASES,
    NO_NUMBER_MARKERS,
    STREET_TYPE_ALIASES,
)


_ACCENTED_CHARACTERS = "áàâãäéèêëíìîïóòôõöúùûüçñýÿ"
_ASCII_EQUIVALENTS = "aaaaaeeeeiiiiooooouuuucnyy"

CNEFE_COMPONENT_NORMALIZED_COLUMNS = (
    "municipio_codigo_normalizado",
    "cep_normalizado",
    "localidade_normalizada",
    "tipo_logradouro_normalizado",
    "titulo_logradouro_normalizado",
    "nome_logradouro_normalizado",
    "logradouro_normalizado",
    "numero_normalizado",
    "numero_ausente",
    "sem_numero",
    "numero_comparavel_normalizado",
    "modificador_normalizado",
    *(f"nome_comp_elem{position}_normalizado" for position in range(1, 6)),
    *(f"valor_comp_elem{position}_normalizado" for position in range(1, 6)),
    "complemento_normalizado",
    "endereco_normalizado",
)


def normalize_text_column(column: Column) -> Column:
    normalized = F.lower(F.trim(column.cast("string")))
    normalized = F.translate(
        normalized, _ACCENTED_CHARACTERS, _ASCII_EQUIVALENTS
    )
    normalized = F.regexp_replace(normalized, r"[^a-z0-9]+", " ")
    normalized = F.trim(F.regexp_replace(normalized, r"\s+", " "))
    return F.when(column.isNull() | (normalized == ""), F.lit(None)).otherwise(
        normalized
    )


def _replace_exact_aliases(
    normalized: Column, aliases: Mapping[str, str]
) -> Column:
    entries = [
        literal
        for alias, canonical in aliases.items()
        for literal in (F.lit(alias), F.lit(canonical))
    ]
    alias_map = F.create_map(*entries)
    return F.coalesce(F.element_at(alias_map, normalized), normalized)


def normalize_street_type_column(column: Column) -> Column:
    return _replace_exact_aliases(
        normalize_text_column(column), STREET_TYPE_ALIASES
    )


def normalize_number_column(column: Column) -> Column:
    normalized = normalize_text_column(column)
    without_leading_zeroes = F.regexp_replace(normalized, r"^0+(?=[0-9])", "")
    return (
        F.when(normalized.isin(*NO_NUMBER_MARKERS), F.lit("sn"))
        .when(normalized.rlike(r"^[0-9]+$"), without_leading_zeroes)
        .otherwise(normalized)
    )


def has_without_number_marker_column(*columns: Column) -> Column:
    result = F.lit(False)
    for column in columns:
        marker_present = F.coalesce(
            normalize_text_column(column).isin(*NO_NUMBER_MARKERS),
            F.lit(False),
        )
        result = result | marker_present
    return result


def normalize_postal_code_column(column: Column) -> Column:
    digits = F.regexp_replace(column.cast("string"), r"[^0-9]+", "")
    return F.when(F.length(digits) == 8, digits).otherwise(F.lit(None))


def normalize_municipality_code_column(column: Column) -> Column:
    digits = F.regexp_replace(column.cast("string"), r"[^0-9]+", "")
    return (
        F.when(F.length(digits) == 7, F.substring(digits, 1, 6))
        .when(F.length(digits) == 6, digits)
        .otherwise(F.lit(None))
    )


def normalize_complement_column(column: Column) -> Column:
    normalized = normalize_text_column(column)
    result = normalized
    for alias, canonical in COMPLEMENT_TOKEN_ALIASES.items():
        # Os limites nao consomem o espaco entre tokens repetidos.
        result = F.regexp_replace(
            result,
            rf"(?<!\S){alias}(?!\S)",
            canonical,
        )
    return result


def concat_normalized_columns(columns: Sequence[Column]) -> Column:
    joined = F.concat_ws(" ", *columns)
    joined = F.trim(F.regexp_replace(joined, r"\s+", " "))
    return F.when(joined == "", F.lit(None)).otherwise(joined)


def normalize_cnefe_addresses(dataframe: DataFrame) -> DataFrame:
    """Acrescenta campos normalizados e mantem todas as colunas originais."""
    result = (
        dataframe.withColumn(
            "municipio_codigo_normalizado",
            normalize_municipality_code_column(F.col("COD_MUNICIPIO")),
        )
        .withColumn("cep_normalizado", normalize_postal_code_column(F.col("CEP")))
        .withColumn(
            "localidade_normalizada", normalize_text_column(F.col("DSC_LOCALIDADE"))
        )
        .withColumn(
            "tipo_logradouro_normalizado",
            normalize_street_type_column(F.col("NOM_TIPO_SEGLOGR")),
        )
        .withColumn(
            "titulo_logradouro_normalizado",
            normalize_text_column(F.col("NOM_TITULO_SEGLOGR")),
        )
        .withColumn(
            "nome_logradouro_normalizado",
            normalize_text_column(F.col("NOM_SEGLOGR")),
        )
        .withColumn(
            "numero_normalizado", normalize_number_column(F.col("NUM_ENDERECO"))
        )
        .withColumn(
            "numero_ausente", F.col("numero_normalizado").isNull()
        )
        .withColumn(
            "modificador_normalizado",
            normalize_complement_column(F.col("DSC_MODIFICADOR")),
        )
        .withColumn(
            "sem_numero",
            has_without_number_marker_column(
                F.col("NUM_ENDERECO"), F.col("DSC_MODIFICADOR")
            ),
        )
        .withColumn(
            "numero_comparavel_normalizado",
            F.when(F.col("sem_numero"), F.lit("sn")).otherwise(
                F.col("numero_normalizado")
            ),
        )
    )

    complement_columns: list[Column] = []
    for position in range(1, 6):
        name_column = f"nome_comp_elem{position}_normalizado"
        value_column = f"valor_comp_elem{position}_normalizado"
        result = result.withColumn(
            name_column,
            normalize_complement_column(F.col(f"NOM_COMP_ELEM{position}")),
        ).withColumn(
            value_column,
            normalize_complement_column(F.col(f"VAL_COMP_ELEM{position}")),
        )
        normalized_name = F.col(name_column)
        normalized_value = F.col(value_column)
        complement_columns.append(
            F.when(normalized_name == normalized_value, normalized_name).otherwise(
                concat_normalized_columns([normalized_name, normalized_value])
            )
        )

    result = result.withColumn(
        "logradouro_normalizado",
        concat_normalized_columns(
            [
                F.col("tipo_logradouro_normalizado"),
                F.col("titulo_logradouro_normalizado"),
                F.col("nome_logradouro_normalizado"),
            ]
        ),
    ).withColumn(
        "complemento_normalizado",
        concat_normalized_columns(complement_columns),
    )

    return result.withColumn(
        "endereco_normalizado",
        concat_normalized_columns(
            [
                F.col("municipio_codigo_normalizado"),
                F.col("cep_normalizado"),
                F.col("logradouro_normalizado"),
                F.col("numero_comparavel_normalizado"),
                F.col("localidade_normalizada"),
                F.col("modificador_normalizado"),
                F.col("complemento_normalizado"),
            ]
        ),
    )
