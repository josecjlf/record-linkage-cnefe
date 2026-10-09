"""Pipeline end-to-end que reutiliza CNEFE preparado e indice persistente."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any

from src.config import Settings
from src.indexing.elasticsearch_index import (
    check_elasticsearch,
    create_elasticsearch_client,
)
from src.ingestion.cnefe import create_spark_session
from src.indexing.index_mapping import MAPPING_VERSION
from src.matching.baseline_pipeline import execute_baseline
from src.matching.candidate_retrieval_pipeline import (
    execute_candidate_retrieval,
)
from src.matching.candidate_selection_pipeline import (
    execute_candidate_selection,
)
from src.matching.candidate_selector import SelectionPolicy
from src.output.exporter import ExportOutput, export_results
from src.preprocessing.normalization_pipeline import (
    prepare_normalized_search,
)


LOGGER = logging.getLogger(__name__)


class PipelinePrerequisiteError(RuntimeError):
    """Um artefato pesado precisa ser preparado explicitamente."""


@dataclass(frozen=True)
class EndToEndOutput:
    export: ExportOutput
    indexed_documents: int


def _validate_cnefe_prerequisite(settings: Settings) -> None:
    if not settings.cnefe_normalized_parquet_path.is_dir():
        raise PipelinePrerequisiteError(
            "CNEFE normalizado nao encontrado em "
            f"{settings.cnefe_normalized_parquet_path}. Execute "
            "prepare-cnefe e normalize-cnefe antes do pipeline principal."
        )


def _read_index_report(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PipelinePrerequisiteError(
            f"Relatorio de indexacao nao encontrado: {path}. Execute index-cnefe."
        )
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PipelinePrerequisiteError(
            f"Relatorio de indexacao invalido: {path}. Execute index-cnefe novamente."
        ) from error
    if not isinstance(report, dict):
        raise PipelinePrerequisiteError(
            f"Relatorio de indexacao invalido: {path}."
        )
    return report


def _validate_elasticsearch_prerequisite(
    client: Any,
    *,
    index_name: str,
    report_path: Path,
) -> int:
    """Confirma que índice, mapping e relatório descrevem uma carga completa."""
    if not bool(client.indices.exists(index=index_name)):
        raise PipelinePrerequisiteError(
            f"Indice Elasticsearch nao encontrado: {index_name}. Execute index-cnefe."
        )

    report = _read_index_report(report_path)
    required_fields = {
        "index_name",
        "mapping_version",
        "source_rows",
        "successful_bulk_operations",
        "failed_bulk_operations",
        "indexed_documents",
    }
    missing_fields = sorted(required_fields - set(report))
    if missing_fields:
        raise PipelinePrerequisiteError(
            "Relatorio de indexacao incompleto; campos ausentes: "
            f"{missing_fields}. Execute index-cnefe novamente."
        )
    if report["index_name"] != index_name:
        raise PipelinePrerequisiteError(
            "Relatorio de indexacao pertence a outro indice: "
            f"{report['index_name']!r}."
        )
    if report["mapping_version"] != MAPPING_VERSION:
        raise PipelinePrerequisiteError(
            "Relatorio usa mapping incompatível: "
            f"esperado={MAPPING_VERSION}, encontrado={report['mapping_version']}."
        )
    if int(report["failed_bulk_operations"]) != 0:
        raise PipelinePrerequisiteError(
            "A ultima indexacao registrou falhas bulk. Reconstrua o indice "
            "explicitamente com index-cnefe --recreate-index."
        )
    if int(report["successful_bulk_operations"]) != int(report["source_rows"]):
        raise PipelinePrerequisiteError(
            "A ultima indexacao nao processou todas as linhas do CNEFE."
        )

    mapping_response = client.indices.get_mapping(index=index_name)
    mappings = mapping_response.get(index_name, {}).get("mappings", {})
    actual_mapping_version = mappings.get("_meta", {}).get("mapping_version")
    if actual_mapping_version != MAPPING_VERSION:
        raise PipelinePrerequisiteError(
            "Indice Elasticsearch usa mapping incompatível: "
            f"esperado={MAPPING_VERSION}, encontrado={actual_mapping_version!r}."
        )

    indexed_documents = int(client.count(index=index_name)["count"])
    if indexed_documents == 0:
        raise PipelinePrerequisiteError(f"Indice {index_name} esta vazio.")
    if indexed_documents != int(report["indexed_documents"]):
        raise PipelinePrerequisiteError(
            "Contagem do indice diverge do relatorio da carga: "
            f"indice={indexed_documents}, relatorio={report['indexed_documents']}."
        )
    return indexed_documents


def execute_end_to_end_pipeline(settings: Settings) -> EndToEndOutput:
    """Refaz a busca pequena sem reconstruir CNEFE ou indice automaticamente."""
    LOGGER.info("Iniciando pipeline end-to-end")
    _validate_cnefe_prerequisite(settings)
    client = create_elasticsearch_client(
        settings.elasticsearch_url,
        request_timeout_seconds=settings.elasticsearch_request_timeout_seconds,
    )
    try:
        check_elasticsearch(client)
        indexed_documents = _validate_elasticsearch_prerequisite(
            client,
            index_name=settings.elasticsearch_index_name,
            report_path=settings.elasticsearch_index_report_path,
        )
        LOGGER.info(
            "Pre-requisitos validados: documentos_indexados=%s",
            indexed_documents,
        )

        LOGGER.info("Etapa 1/5: normalizacao da base de busca")
        prepare_normalized_search(
            input_path=settings.search_input_path,
            output_path=settings.search_normalized_path,
            expected_records=settings.expected_search_records,
        )

        LOGGER.info("Etapa 2/5: baseline exata")
        spark = create_spark_session(
            app_name=f"{settings.spark_app_name}-end-to-end",
            master=settings.spark_master,
        )
        try:
            execute_baseline(
                spark,
                cnefe_path=settings.cnefe_normalized_parquet_path,
                search_path=settings.search_normalized_path,
                result_path=settings.baseline_output_path,
                metrics_path=settings.baseline_metrics_path,
            )
        finally:
            spark.stop()

        LOGGER.info("Etapa 3/5: candidate retrieval")
        execute_candidate_retrieval(
            client,
            baseline_path=settings.baseline_output_path,
            index_name=settings.elasticsearch_index_name,
            diagnostics_path=settings.candidate_retrieval_diagnostics_path,
            candidates_path=settings.candidate_retrieval_candidates_path,
            metrics_path=settings.candidate_retrieval_metrics_path,
            analysis_path=settings.candidate_retrieval_analysis_path,
            top_k=settings.candidate_retrieval_top_k,
            minimum_candidates=(
                settings.candidate_retrieval_minimum_candidates
            ),
            expected_records=settings.expected_search_records,
        )

        LOGGER.info("Etapa 4/5: candidate selection")
        execute_candidate_selection(
            baseline_path=settings.baseline_output_path,
            retrieval_diagnostics_path=(
                settings.candidate_retrieval_diagnostics_path
            ),
            audit_path=settings.candidate_selection_audit_path,
            evidence_path=settings.candidate_selection_evidence_path,
            metrics_path=settings.candidate_selection_metrics_path,
            report_path=settings.matching_strategy_report_path,
            policy=SelectionPolicy(
                strong_street_similarity=(
                    settings.candidate_selection_strong_street_similarity
                ),
                plausible_street_similarity=(
                    settings.candidate_selection_plausible_street_similarity
                ),
            ),
            expected_records=settings.expected_search_records,
        )

        LOGGER.info("Etapa 5/5: validacao e exportacao")
        export = export_results(
            original_path=settings.search_input_path,
            selection_audit_path=settings.candidate_selection_audit_path,
            official_path=settings.final_result_path,
            audit_path=settings.final_audit_path,
            metrics_path=settings.final_metrics_path,
            expected_records=settings.expected_search_records,
        )
        LOGGER.info(
            "Pipeline concluido: registros=%s, found=%s, ambiguous=%s, "
            "not_found=%s, arquivo=%s",
            export.metrics.output_records,
            export.metrics.found,
            export.metrics.ambiguous,
            export.metrics.not_found,
            export.official_path,
        )
        return EndToEndOutput(
            export=export,
            indexed_documents=indexed_documents,
        )
    finally:
        client.close()
