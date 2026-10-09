"""Ingestao do CNEFE com PySpark e persistencia em Parquet."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession, functions as F

from src.ingestion.schemas import (
    CNEFE_COLUMNS,
    CNEFE_SELECTED_COLUMNS,
    CNEFE_SPARK_SCHEMA,
)
from src.ingestion.validation import (
    DatasetValidationError,
    read_csv_header,
    validate_exact_schema_order,
    validate_input_file,
    validate_required_columns,
)


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CnefeValidationMetrics:
    total_records: int
    bahia_records: int
    filtered_other_uf_records: int
    missing_sector_records: int
    invalid_sector_records: int
    missing_identifier_records: int


@dataclass(frozen=True)
class PreparedCnefeResult:
    output_path: Path
    record_count: int
    column_count: int


def create_spark_session(*, app_name: str, master: str) -> SparkSession:
    """Cria uma sessao Spark local/configuravel, sem iniciar um cluster separado."""
    spark = (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def read_cnefe(spark: SparkSession, path: Path) -> DataFrame:
    """Le o CSV bruto com schema explicito e sem inferencia de tipos."""
    validate_input_file(path, "CNEFE")
    header = read_csv_header(path)
    validate_exact_schema_order(header, CNEFE_COLUMNS, "CNEFE")
    return (
        spark.read.option("header", True)
        .option("sep", ";")
        .option("encoding", "UTF-8")
        .option("mode", "FAILFAST")
        .schema(CNEFE_SPARK_SCHEMA)
        .csv(str(path))
    )


def validate_cnefe(
    dataframe: DataFrame, *, expected_uf_code: str
) -> CnefeValidationMetrics:
    """Executa uma única agregação para validar volume e chaves do CNEFE."""
    validate_required_columns(
        dataframe.columns, CNEFE_SELECTED_COLUMNS, "CNEFE"
    )
    is_bahia = F.col("COD_UF") == F.lit(expected_uf_code)
    missing_sector = F.col("COD_SETOR").isNull() | (
        F.length(F.trim(F.col("COD_SETOR"))) == 0
    )
    invalid_sector = (~missing_sector) & (
        ~F.col("COD_SETOR").rlike(r"^[0-9]{15}[A-Z]?$")
    )
    missing_identifier = F.col("COD_UNICO_ENDERECO").isNull() | (
        F.length(F.trim(F.col("COD_UNICO_ENDERECO"))) == 0
    )
    metrics = dataframe.agg(
        F.count(F.lit(1)).alias("total_records"),
        F.sum(F.when(is_bahia, 1).otherwise(0)).alias("bahia_records"),
        F.sum(F.when(is_bahia & missing_sector, 1).otherwise(0)).alias(
            "missing_sector_records"
        ),
        F.sum(F.when(is_bahia & invalid_sector, 1).otherwise(0)).alias(
            "invalid_sector_records"
        ),
        F.sum(F.when(is_bahia & missing_identifier, 1).otherwise(0)).alias(
            "missing_identifier_records"
        ),
    ).first()

    total_records = int(metrics["total_records"])
    bahia_records = int(metrics["bahia_records"] or 0)
    missing_sector_records = int(metrics["missing_sector_records"] or 0)
    invalid_sector_records = int(metrics["invalid_sector_records"] or 0)
    missing_identifier_records = int(
        metrics["missing_identifier_records"] or 0
    )
    if total_records == 0:
        raise DatasetValidationError("CNEFE nao possui registros.")
    if bahia_records == 0:
        raise DatasetValidationError(
            f"CNEFE nao possui registros para COD_UF={expected_uf_code}."
        )
    if missing_sector_records:
        raise DatasetValidationError(
            "CNEFE possui registros da Bahia sem COD_SETOR: "
            f"{missing_sector_records}."
        )
    if invalid_sector_records:
        raise DatasetValidationError(
            "CNEFE possui COD_SETOR com formato invalido: "
            f"{invalid_sector_records}."
        )
    if missing_identifier_records:
        raise DatasetValidationError(
            "CNEFE possui registros da Bahia sem COD_UNICO_ENDERECO: "
            f"{missing_identifier_records}."
        )

    return CnefeValidationMetrics(
        total_records=total_records,
        bahia_records=bahia_records,
        filtered_other_uf_records=total_records - bahia_records,
        missing_sector_records=missing_sector_records,
        invalid_sector_records=invalid_sector_records,
        missing_identifier_records=missing_identifier_records,
    )


def transform_cnefe(
    dataframe: DataFrame, *, expected_uf_code: str
) -> DataFrame:
    """Filtra Bahia e seleciona os campos necessarios ao processamento."""
    return dataframe.filter(F.col("COD_UF") == expected_uf_code).select(
        *CNEFE_SELECTED_COLUMNS
    )


def write_cnefe_parquet(dataframe: DataFrame, output_path: Path) -> None:
    """Grava Parquet com schema textual preservado e compressao Snappy."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    (
        dataframe.write.mode("overwrite")
        .option("compression", "snappy")
        .parquet(str(output_path))
    )


def prepare_cnefe_parquet(
    spark: SparkSession,
    *,
    input_path: Path,
    output_path: Path,
    expected_uf_code: str,
) -> PreparedCnefeResult:
    """Executa a ingestao oficial e valida o Parquet materializado."""
    raw = read_cnefe(spark, input_path)
    metrics = validate_cnefe(raw, expected_uf_code=expected_uf_code)
    LOGGER.info(
        "CNEFE validado: total=%s, bahia=%s, outras_ufs=%s",
        metrics.total_records,
        metrics.bahia_records,
        metrics.filtered_other_uf_records,
    )

    prepared = transform_cnefe(raw, expected_uf_code=expected_uf_code)
    write_cnefe_parquet(prepared, output_path)

    stored = spark.read.parquet(str(output_path))
    validate_exact_schema_order(
        stored.columns, CNEFE_SELECTED_COLUMNS, "CNEFE Parquet"
    )
    stored_count = stored.count()
    if stored_count != metrics.bahia_records:
        raise DatasetValidationError(
            "Parquet do CNEFE perdeu registros: "
            f"esperado={metrics.bahia_records}, encontrado={stored_count}."
        )

    LOGGER.info(
        "CNEFE Parquet validado: registros=%s, colunas=%s, destino=%s",
        stored_count,
        len(stored.columns),
        output_path,
    )
    return PreparedCnefeResult(
        output_path=output_path,
        record_count=stored_count,
        column_count=len(stored.columns),
    )
