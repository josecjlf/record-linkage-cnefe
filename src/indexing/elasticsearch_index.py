"""Conexao, ciclo de vida seguro e carga bulk do indice CNEFE."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
import logging
from pathlib import Path
from time import monotonic
from typing import Any, Iterator

from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk
from pyspark.sql import DataFrame, SparkSession

from src.indexing.index_mapping import (
    CNEFE_INDEX_MAPPINGS,
    CNEFE_INDEX_SETTINGS,
    CNEFE_INDEX_SOURCE_COLUMNS,
    MAPPING_VERSION,
    build_bulk_action,
)
from src.ingestion.validation import validate_required_columns


LOGGER = logging.getLogger(__name__)


class ElasticsearchUnavailableError(RuntimeError):
    """O servico nao respondeu a uma verificacao simples."""


class IndexAlreadyExistsError(RuntimeError):
    """Protege um indice existente de recriacao implicita."""


class BulkIndexingError(RuntimeError):
    """Uma ou mais operacoes bulk falharam."""


@dataclass(frozen=True)
class PartitionIndexResult:
    successful_operations: int
    failed_operations: int
    error_samples: tuple[str, ...]


@dataclass(frozen=True)
class CnefeIndexReport:
    generated_at_utc: str
    index_name: str
    mapping_version: int
    source_rows: int
    successful_bulk_operations: int
    failed_bulk_operations: int
    indexed_documents: int
    overwritten_duplicate_operations: int
    duration_seconds: float
    error_samples: tuple[str, ...]


def create_elasticsearch_client(
    url: str, *, request_timeout_seconds: int
) -> Elasticsearch:
    return Elasticsearch(url, request_timeout=request_timeout_seconds)


def check_elasticsearch(client: Elasticsearch) -> dict[str, object]:
    """Falha com mensagem clara se o no local nao estiver acessivel."""
    try:
        info = client.info()
        health = client.cluster.health()
    except Exception as error:
        raise ElasticsearchUnavailableError(
            "Elasticsearch indisponivel. Suba o servico com "
            "'docker compose up -d --wait elasticsearch' e confira "
            "ELASTICSEARCH_URL. No host use http://localhost:9200; "
            "no container app use http://elasticsearch:9200."
        ) from error
    return {
        "cluster_name": info["cluster_name"],
        "version": info["version"]["number"],
        "health_status": health["status"],
        "number_of_nodes": health["number_of_nodes"],
    }


def create_cnefe_index(
    client: Elasticsearch,
    *,
    index_name: str,
    recreate: bool = False,
) -> None:
    """Cria o indice e so o remove mediante confirmacao explicita do CLI."""
    exists = bool(client.indices.exists(index=index_name))
    if exists and not recreate:
        raise IndexAlreadyExistsError(
            f"Indice '{index_name}' ja existe. Use --recreate-index somente "
            "se deseja apagar e reconstruir seu conteudo."
        )
    if exists:
        LOGGER.warning("Removendo indice por solicitacao explicita: %s", index_name)
        client.indices.delete(index=index_name)

    client.indices.create(
        index=index_name,
        settings=CNEFE_INDEX_SETTINGS,
        mappings=CNEFE_INDEX_MAPPINGS,
    )
    LOGGER.info("Indice criado: %s", index_name)


def _serialize_error(item: dict[str, Any]) -> str:
    return json.dumps(item, ensure_ascii=False, default=str)[:2000]


def _index_partition(
    rows: Iterator[Any],
    *,
    elasticsearch_url: str,
    index_name: str,
    chunk_size: int,
    request_timeout_seconds: int,
    max_retries: int,
) -> Iterator[PartitionIndexResult]:
    client = create_elasticsearch_client(
        elasticsearch_url,
        request_timeout_seconds=request_timeout_seconds,
    )
    successful = 0
    failed = 0
    error_samples: list[str] = []

    actions = (
        build_bulk_action(row.asDict(recursive=False), index_name=index_name)
        for row in rows
    )
    try:
        for ok, item in streaming_bulk(
            client,
            actions,
            chunk_size=chunk_size,
            max_retries=max_retries,
            initial_backoff=2,
            max_backoff=30,
            raise_on_error=False,
            raise_on_exception=False,
            yield_ok=True,
        ):
            if ok:
                successful += 1
            else:
                failed += 1
                if len(error_samples) < 3:
                    error_samples.append(_serialize_error(item))
    finally:
        client.close()

    yield PartitionIndexResult(
        successful_operations=successful,
        failed_operations=failed,
        error_samples=tuple(error_samples),
    )


def _bulk_index_dataframe(
    dataframe: DataFrame,
    *,
    elasticsearch_url: str,
    index_name: str,
    chunk_size: int,
    request_timeout_seconds: int,
    max_retries: int,
) -> tuple[int, int, tuple[str, ...]]:
    partition_results = dataframe.rdd.mapPartitions(
        lambda rows: _index_partition(
            rows,
            elasticsearch_url=elasticsearch_url,
            index_name=index_name,
            chunk_size=chunk_size,
            request_timeout_seconds=request_timeout_seconds,
            max_retries=max_retries,
        )
    ).collect()
    successful = sum(result.successful_operations for result in partition_results)
    failed = sum(result.failed_operations for result in partition_results)
    error_samples = tuple(
        error
        for result in partition_results
        for error in result.error_samples
    )[:10]
    return successful, failed, error_samples


def index_cnefe_parquet(
    spark: SparkSession,
    *,
    client: Elasticsearch,
    elasticsearch_url: str,
    index_name: str,
    parquet_path: Path,
    recreate_index: bool,
    chunk_size: int,
    request_timeout_seconds: int,
    max_retries: int,
) -> CnefeIndexReport:
    """Indexa o Parquet em bulk, sem materializar o CNEFE no driver."""
    if not parquet_path.is_dir():
        raise FileNotFoundError(
            f"Parquet normalizado do CNEFE nao encontrado: {parquet_path}"
        )
    if chunk_size <= 0:
        raise ValueError("chunk_size deve ser maior que zero.")

    check_elasticsearch(client)
    source = spark.read.parquet(str(parquet_path))
    validate_required_columns(
        source.columns,
        CNEFE_INDEX_SOURCE_COLUMNS,
        "CNEFE normalizado para indexacao",
    )
    selected = source.select(*CNEFE_INDEX_SOURCE_COLUMNS)
    source_rows = selected.count()
    if source_rows == 0:
        raise ValueError("CNEFE normalizado nao possui registros para indexar.")

    create_cnefe_index(
        client,
        index_name=index_name,
        recreate=recreate_index,
    )

    LOGGER.info(
        "Iniciando bulk: indice=%s, linhas_fonte=%s, particoes=%s, chunk=%s",
        index_name,
        source_rows,
        selected.rdd.getNumPartitions(),
        chunk_size,
    )
    started = monotonic()
    try:
        successful, failed, error_samples = _bulk_index_dataframe(
            selected,
            elasticsearch_url=elasticsearch_url,
            index_name=index_name,
            chunk_size=chunk_size,
            request_timeout_seconds=request_timeout_seconds,
            max_retries=max_retries,
        )
    finally:
        client.indices.put_settings(
            index=index_name,
            settings={"index": {"refresh_interval": "1s"}},
        )
        client.indices.refresh(index=index_name)
    indexed_documents = int(client.count(index=index_name)["count"])
    duration_seconds = round(monotonic() - started, 2)
    report = CnefeIndexReport(
        generated_at_utc=datetime.now(UTC).isoformat(),
        index_name=index_name,
        mapping_version=MAPPING_VERSION,
        source_rows=source_rows,
        successful_bulk_operations=successful,
        failed_bulk_operations=failed,
        indexed_documents=indexed_documents,
        overwritten_duplicate_operations=max(successful - indexed_documents, 0),
        duration_seconds=duration_seconds,
        error_samples=error_samples,
    )
    LOGGER.info(
        "Bulk concluido: operacoes_ok=%s, falhas=%s, documentos=%s, segundos=%s",
        successful,
        failed,
        indexed_documents,
        duration_seconds,
    )
    return report


def write_index_report(path: Path, report: CnefeIndexReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
