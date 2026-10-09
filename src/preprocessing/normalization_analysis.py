"""Estatisticas reproduziveis de impacto e colisoes da normalizacao."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path

import pandas as pd
from pyspark.sql import DataFrame, functions as F

from src.ingestion.schemas import SEARCH_COLUMNS
from src.preprocessing.address_normalizer import normalize_text


SEARCH_FIELD_PAIRS = {
    "logradouro": ("consulta_logradouro", "logradouro_normalizado"),
    "numero": ("consulta_numero", "numero_normalizado"),
    "bairro": ("consulta_bairro", "bairro_normalizado"),
    "cep": ("consulta_cep", "cep_normalizado"),
    "complemento": ("consulta_complemento", "complemento_normalizado"),
    "municipio": (
        "consulta_municipio",
        "municipio_codigo_normalizado",
    ),
}

CNEFE_FIELD_PAIRS = {
    "municipio": ("COD_MUNICIPIO", "municipio_codigo_normalizado"),
    "cep": ("CEP", "cep_normalizado"),
    "localidade": ("DSC_LOCALIDADE", "localidade_normalizada"),
    "tipo_logradouro": (
        "NOM_TIPO_SEGLOGR",
        "tipo_logradouro_normalizado",
    ),
    "titulo_logradouro": (
        "NOM_TITULO_SEGLOGR",
        "titulo_logradouro_normalizado",
    ),
    "nome_logradouro": ("NOM_SEGLOGR", "nome_logradouro_normalizado"),
    "numero": ("NUM_ENDERECO", "numero_normalizado"),
    "modificador": ("DSC_MODIFICADOR", "modificador_normalizado"),
    "complemento": ("_complemento_original", "complemento_normalizado"),
}

CNEFE_ORIGINAL_ADDRESS_COLUMNS = (
    "COD_MUNICIPIO",
    "CEP",
    "DSC_LOCALIDADE",
    "NOM_TIPO_SEGLOGR",
    "NOM_TITULO_SEGLOGR",
    "NOM_SEGLOGR",
    "NUM_ENDERECO",
    "DSC_MODIFICADOR",
    *(f"NOM_COMP_ELEM{position}" for position in range(1, 6)),
    *(f"VAL_COMP_ELEM{position}" for position in range(1, 6)),
)


def _pandas_missing(series: pd.Series) -> pd.Series:
    return series.isna() | series.astype("string").str.strip().eq("")


def _pandas_boolean_sum(series: pd.Series) -> int:
    if pd.api.types.is_bool_dtype(series.dtype):
        return int(series.sum())
    return int(series.astype("string").str.lower().eq("true").sum())


def analyze_search_normalization(dataframe: pd.DataFrame) -> dict[str, object]:
    """Calcula metricas exatas para a pequena base de busca."""
    fields: dict[str, object] = {}
    for component, (original_column, normalized_column) in SEARCH_FIELD_PAIRS.items():
        original = dataframe[original_column]
        normalized = dataframe[normalized_column]
        original_missing = _pandas_missing(original)
        normalized_missing = _pandas_missing(normalized)
        normalized_original = original.map(
            lambda value: normalize_text(None if pd.isna(value) else str(value))
        ).astype("string")
        equal_values = normalized_original.eq(normalized).fillna(False) | (
            normalized_original.isna() & normalized.isna()
        )
        fields[component] = {
            "original_missing": int(original_missing.sum()),
            "normalized_missing": int(normalized_missing.sum()),
            "original_distinct": int(original[~original_missing].nunique()),
            "normalized_distinct": int(normalized[~normalized_missing].nunique()),
            "changed_beyond_basic_text": int((~equal_values).sum()),
        }

    original_key = dataframe.loc[:, list(SEARCH_COLUMNS)].fillna("").astype(str).agg(
        "\x1f".join, axis=1
    )
    collision_frame = pd.DataFrame(
        {
            "normalized": dataframe["endereco_normalizado"],
            "original": original_key,
        }
    ).dropna(subset=["normalized"])
    grouped = collision_frame.groupby("normalized", dropna=False).agg(
        rows=("original", "size"),
        original_variants=("original", "nunique"),
    )
    collisions = grouped[grouped["original_variants"] > 1]

    return {
        "records": len(dataframe),
        "fields": fields,
        "without_number": _pandas_boolean_sum(dataframe["sem_numero"]),
        "missing_number": _pandas_boolean_sum(dataframe["numero_ausente"]),
        "normalized_address_distinct": int(
            dataframe["endereco_normalizado"].nunique(dropna=True)
        ),
        "normalization_collision_groups": len(collisions),
        "rows_in_normalization_collision_groups": int(collisions["rows"].sum()),
    }


def _spark_missing(column_name: str):
    column = F.col(column_name)
    return column.isNull() | (F.trim(column) == "")


def analyze_cnefe_normalization(dataframe: DataFrame) -> dict[str, object]:
    """Agrega metricas no Spark sem coletar registros individuais."""
    original_complement = F.concat_ws(
        " ",
        *[
            F.col(column)
            for position in range(1, 6)
            for column in (
                f"NOM_COMP_ELEM{position}",
                f"VAL_COMP_ELEM{position}",
            )
        ],
    )
    dataframe = dataframe.withColumn(
        "_complemento_original",
        F.when(F.trim(original_complement) == "", F.lit(None)).otherwise(
            original_complement
        ),
    )
    summary = dataframe.agg(
        F.count(F.lit(1)).alias("records"),
        *[
            F.sum(F.when(F.col("sem_numero"), 1).otherwise(0)).alias(
                "without_number"
            ),
            F.sum(F.when(F.col("numero_ausente"), 1).otherwise(0)).alias(
                "missing_number"
            ),
        ],
    ).first().asDict()

    fields: dict[str, object] = {}
    field_items = list(CNEFE_FIELD_PAIRS.items())
    for start in range(0, len(field_items), 3):
        batch = field_items[start : start + 3]
        aggregations = []
        for component, (original_column, normalized_column) in batch:
            original_missing = _spark_missing(original_column)
            normalized_missing = _spark_missing(normalized_column)
            basic_original = F.lower(F.trim(F.col(original_column)))
            aggregations.extend(
                [
                    F.sum(F.when(original_missing, 1).otherwise(0)).alias(
                        f"{component}__original_missing"
                    ),
                    F.sum(F.when(normalized_missing, 1).otherwise(0)).alias(
                        f"{component}__normalized_missing"
                    ),
                    F.approx_count_distinct(
                        F.when(~original_missing, F.col(original_column)),
                        rsd=0.01,
                    ).alias(f"{component}__original_distinct_approx"),
                    F.approx_count_distinct(
                        F.when(~normalized_missing, F.col(normalized_column)),
                        rsd=0.01,
                    ).alias(f"{component}__normalized_distinct_approx"),
                    F.sum(
                        F.when(
                            ~basic_original.eqNullSafe(F.col(normalized_column)),
                            1,
                        ).otherwise(0)
                    ).alias(f"{component}__changed_beyond_lower_trim"),
                ]
            )
        batch_metrics = dataframe.agg(*aggregations).first().asDict()
        for component, _ in batch:
            fields[component] = {
                "original_missing": int(
                    batch_metrics[f"{component}__original_missing"]
                ),
                "normalized_missing": int(
                    batch_metrics[f"{component}__normalized_missing"]
                ),
                "original_distinct_approx": int(
                    batch_metrics[f"{component}__original_distinct_approx"]
                ),
                "normalized_distinct_approx": int(
                    batch_metrics[f"{component}__normalized_distinct_approx"]
                ),
                "changed_beyond_lower_trim": int(
                    batch_metrics[f"{component}__changed_beyond_lower_trim"]
                ),
            }

    original_key = F.concat_ws(
        "\u001f",
        *[
            F.coalesce(F.col(column), F.lit("<NULL>"))
            for column in CNEFE_ORIGINAL_ADDRESS_COLUMNS
        ],
    )
    grouped = (
        dataframe.select(
            "endereco_normalizado",
            "COD_SETOR",
            original_key.alias("endereco_original_key"),
        )
        .where(F.col("endereco_normalizado").isNotNull())
        .groupBy("endereco_normalizado")
        .agg(
            F.count(F.lit(1)).alias("rows"),
            F.countDistinct("endereco_original_key").alias("original_variants"),
            F.countDistinct("COD_SETOR").alias("sectors"),
        )
    )
    collision_metrics = grouped.agg(
        F.count(F.lit(1)).alias("normalized_address_distinct"),
        F.sum(F.col("original_variants")).alias("original_address_distinct"),
        F.sum(F.when(F.col("original_variants") > 1, 1).otherwise(0)).alias(
            "normalization_collision_groups"
        ),
        F.sum(
            F.when(F.col("original_variants") > 1, F.col("rows")).otherwise(0)
        ).alias("rows_in_normalization_collision_groups"),
        F.sum(F.when(F.col("sectors") > 1, 1).otherwise(0)).alias(
            "normalized_groups_with_multiple_sectors"
        ),
    ).first()

    collision_values = {
        key: int(value or 0) for key, value in collision_metrics.asDict().items()
    }
    collision_values["collapsed_distinct_addresses"] = (
        collision_values["original_address_distinct"]
        - collision_values["normalized_address_distinct"]
    )

    return {
        "records": int(summary["records"]),
        "fields": fields,
        "without_number": int(summary["without_number"] or 0),
        "missing_number": int(summary["missing_number"] or 0),
        **collision_values,
    }


def write_normalization_report(
    output_path: Path,
    *,
    search_metrics: dict[str, object],
    cnefe_metrics: dict[str, object],
) -> None:
    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "methodology": {
            "search_cardinalities": "exact",
            "cnefe_field_cardinalities": "approximate, Spark rsd=0.01",
            "collision_groups": "exact grouping by normalized address",
        },
        "search": search_metrics,
        "cnefe": cnefe_metrics,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
