"""I/O da normalizacao, separado das regras de endereco."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import pandas as pd
from pyspark.sql import DataFrame, SparkSession

from src.ingestion.schemas import CNEFE_SELECTED_COLUMNS
from src.ingestion.search_addresses import ingest_search_addresses
from src.ingestion.validation import (
    DatasetValidationError,
    validate_required_columns,
)
from src.preprocessing.pandas_address_normalizer import (
    SEARCH_NORMALIZED_COLUMNS,
    normalize_search_addresses,
)
from src.preprocessing.spark_address_normalizer import (
    CNEFE_COMPONENT_NORMALIZED_COLUMNS,
    normalize_cnefe_addresses,
)


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class NormalizedDatasetResult:
    output_path: Path
    record_count: int
    original_column_count: int
    normalized_column_count: int


def prepare_normalized_search(
    *, input_path: Path, output_path: Path, expected_records: int
) -> tuple[pd.DataFrame, NormalizedDatasetResult]:
    """Normaliza a busca e persiste CSV sem remover suas colunas originais."""
    source = ingest_search_addresses(
        input_path, expected_records=expected_records
    )
    normalized = normalize_search_addresses(source)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(output_path, sep=";", encoding="utf-8", index=False)

    stored = pd.read_csv(
        output_path,
        sep=";",
        encoding="utf-8",
        dtype="string",
        keep_default_na=False,
    )
    validate_required_columns(
        stored.columns,
        (*source.columns, *SEARCH_NORMALIZED_COLUMNS),
        "busca normalizada",
    )
    if len(stored) != len(source):
        raise DatasetValidationError(
            "CSV normalizado da busca perdeu registros: "
            f"esperado={len(source)}, encontrado={len(stored)}."
        )

    LOGGER.info(
        "Busca normalizada: registros=%s, colunas=%s, destino=%s",
        len(stored),
        len(stored.columns),
        output_path,
    )
    return normalized, NormalizedDatasetResult(
        output_path=output_path,
        record_count=len(stored),
        original_column_count=len(source.columns),
        normalized_column_count=len(SEARCH_NORMALIZED_COLUMNS),
    )


def prepare_normalized_cnefe(
    spark: SparkSession, *, input_path: Path, output_path: Path
) -> tuple[DataFrame, NormalizedDatasetResult]:
    """Normaliza o Parquet CNEFE e valida preservacao de linhas e colunas."""
    if not input_path.is_dir():
        raise FileNotFoundError(
            f"Parquet processado do CNEFE nao encontrado: {input_path}"
        )

    source = spark.read.parquet(str(input_path))
    validate_required_columns(
        source.columns, CNEFE_SELECTED_COLUMNS, "CNEFE processado"
    )
    source_count = source.count()
    if source_count == 0:
        raise DatasetValidationError("CNEFE processado nao possui registros.")

    normalized = normalize_cnefe_addresses(source)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    (
        normalized.write.mode("overwrite")
        .option("compression", "snappy")
        .parquet(str(output_path))
    )

    stored = spark.read.parquet(str(output_path))
    validate_required_columns(
        stored.columns,
        (*CNEFE_SELECTED_COLUMNS, *CNEFE_COMPONENT_NORMALIZED_COLUMNS),
        "CNEFE normalizado",
    )
    stored_count = stored.count()
    if stored_count != source_count:
        raise DatasetValidationError(
            "Parquet normalizado do CNEFE perdeu registros: "
            f"esperado={source_count}, encontrado={stored_count}."
        )

    LOGGER.info(
        "CNEFE normalizado: registros=%s, colunas=%s, destino=%s",
        stored_count,
        len(stored.columns),
        output_path,
    )
    return stored, NormalizedDatasetResult(
        output_path=output_path,
        record_count=stored_count,
        original_column_count=len(source.columns),
        normalized_column_count=len(CNEFE_COMPONENT_NORMALIZED_COLUMNS),
    )

