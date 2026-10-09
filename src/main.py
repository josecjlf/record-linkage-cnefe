"""Comandos executaveis das etapas implementadas do pipeline."""

from __future__ import annotations

import argparse
import logging
from typing import Sequence

import pandas as pd

from src.config import ConfigurationError, settings
from src.evaluation import execute_evaluation
from src.ingestion.cnefe import create_spark_session, prepare_cnefe_parquet
from src.ingestion.search_addresses import ingest_search_addresses
from src.indexing.elasticsearch_index import (
    BulkIndexingError,
    ElasticsearchUnavailableError,
    IndexAlreadyExistsError,
    check_elasticsearch,
    create_elasticsearch_client,
    index_cnefe_parquet,
    write_index_report,
)
from src.indexing.index_validation import (
    IndexValidationError,
    validate_technical_queries,
    write_validation_report,
)
from src.ingestion.validation import DatasetValidationError
from src.matching.baseline_pipeline import execute_baseline
from src.matching.candidate_retrieval_pipeline import (
    execute_candidate_retrieval,
)
from src.matching.candidate_selection_pipeline import (
    execute_candidate_selection,
)
from src.matching.candidate_selector import SelectionPolicy
from src.output.exporter import ExportValidationError, export_results
from src.pipeline import PipelinePrerequisiteError, execute_end_to_end_pipeline
from src.preprocessing.normalization_analysis import (
    analyze_cnefe_normalization,
    analyze_search_normalization,
    write_normalization_report,
)
from src.preprocessing.normalization_pipeline import (
    NormalizedDatasetResult,
    prepare_normalized_cnefe,
    prepare_normalized_search,
)


LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Constrói a interface de linha de comando das etapas reproduzíveis."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "prepare-cnefe",
        help="Le, valida e converte o CNEFE Bahia para Parquet.",
    )
    subparsers.add_parser(
        "validate-search",
        help="Le, valida e adiciona IDs internos a base de busca em memoria.",
    )
    subparsers.add_parser(
        "normalize-search",
        help="Normaliza e grava a pequena base de busca.",
    )
    subparsers.add_parser(
        "normalize-cnefe",
        help="Normaliza o Parquet processado do CNEFE com Spark.",
    )
    subparsers.add_parser(
        "analyze-normalization",
        help="Recalcula estatisticas e colisoes dos artefatos normalizados.",
    )
    subparsers.add_parser(
        "normalize-data",
        help="Normaliza os dois datasets e gera o relatorio de impacto.",
    )
    subparsers.add_parser(
        "run-baseline",
        help="Executa a baseline exata e exporta resultados e metricas.",
    )
    subparsers.add_parser(
        "check-elasticsearch",
        help="Verifica conexao, versao e saude do Elasticsearch.",
    )
    index_parser = subparsers.add_parser(
        "index-cnefe",
        help="Cria o indice e envia o CNEFE normalizado em bulk.",
    )
    index_parser.add_argument(
        "--recreate-index",
        action="store_true",
        help="Apaga explicitamente um indice existente antes da carga.",
    )
    subparsers.add_parser(
        "validate-index",
        help="Executa consultas term/filter, match e multi_match tecnicas.",
    )
    subparsers.add_parser(
        "retrieve-candidates",
        help="Recupera candidatos Elasticsearch sem decidir o match final.",
    )
    subparsers.add_parser(
        "select-candidates",
        help="Classifica candidatos em FOUND, AMBIGUOUS e NOT_FOUND.",
    )
    subparsers.add_parser(
        "export-results",
        help="Valida e exporta os arquivos oficial e de auditoria.",
    )
    subparsers.add_parser(
        "run-pipeline",
        help="Executa busca normalizada ate a exportacao final.",
    )
    evaluation_parser = subparsers.add_parser(
        "evaluate-results",
        help="Avalia os resultados sem alterar as decisoes oficiais.",
    )
    evaluation_parser.add_argument(
        "--with-relaxed-query-experiment",
        action="store_true",
        help=(
            "Testa uma consulta Elasticsearch mais permissiva somente nos "
            "casos sem candidatos; nao altera result.csv."
        ),
    )
    return parser


def configure_logging() -> None:
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    # O client registra um traceback por tentativa de conexao. A CLI ja converte
    # a falha final em uma mensagem operacional curta.
    logging.getLogger("elastic_transport.transport").setLevel(logging.ERROR)
    logging.getLogger("elastic_transport.node_pool").setLevel(logging.ERROR)


def run_prepare_cnefe() -> None:
    spark = create_spark_session(
        app_name=settings.spark_app_name,
        master=settings.spark_master,
    )
    try:
        prepare_cnefe_parquet(
            spark,
            input_path=settings.cnefe_input_path,
            output_path=settings.cnefe_parquet_path,
            expected_uf_code=settings.expected_uf_code,
        )
    finally:
        spark.stop()


def run_validate_search() -> None:
    dataframe = ingest_search_addresses(
        settings.search_input_path,
        expected_records=settings.expected_search_records,
    )
    LOGGER.info(
        "Base de busca validada: registros=%s, colunas_originais=%s",
        len(dataframe),
        len(dataframe.columns) - 1,
    )


def run_normalize_search() -> NormalizedDatasetResult:
    _, result = prepare_normalized_search(
        input_path=settings.search_input_path,
        output_path=settings.search_normalized_path,
        expected_records=settings.expected_search_records,
    )
    return result


def run_normalize_cnefe() -> NormalizedDatasetResult:
    spark = create_spark_session(
        app_name=settings.spark_app_name,
        master=settings.spark_master,
    )
    try:
        _, result = prepare_normalized_cnefe(
            spark,
            input_path=settings.cnefe_parquet_path,
            output_path=settings.cnefe_normalized_parquet_path,
        )
        return result
    finally:
        spark.stop()


def _read_normalized_search() -> pd.DataFrame:
    if not settings.search_normalized_path.is_file():
        raise FileNotFoundError(
            "CSV normalizado da busca nao encontrado: "
            f"{settings.search_normalized_path}"
        )
    return pd.read_csv(
        settings.search_normalized_path,
        sep=";",
        encoding="utf-8",
        dtype="string",
        keep_default_na=False,
    )


def run_analyze_normalization() -> None:
    search = _read_normalized_search()
    if not settings.cnefe_normalized_parquet_path.is_dir():
        raise FileNotFoundError(
            "Parquet normalizado do CNEFE nao encontrado: "
            f"{settings.cnefe_normalized_parquet_path}"
        )
    spark = create_spark_session(
        app_name=settings.spark_app_name,
        master=settings.spark_master,
    )
    try:
        cnefe = spark.read.parquet(
            str(settings.cnefe_normalized_parquet_path)
        )
        write_normalization_report(
            settings.normalization_report_path,
            search_metrics=analyze_search_normalization(search),
            cnefe_metrics=analyze_cnefe_normalization(cnefe),
        )
        LOGGER.info(
            "Relatorio de normalizacao gravado em %s",
            settings.normalization_report_path,
        )
    finally:
        spark.stop()


def run_normalize_data() -> None:
    search, _ = prepare_normalized_search(
        input_path=settings.search_input_path,
        output_path=settings.search_normalized_path,
        expected_records=settings.expected_search_records,
    )
    spark = create_spark_session(
        app_name=settings.spark_app_name,
        master=settings.spark_master,
    )
    try:
        cnefe, _ = prepare_normalized_cnefe(
            spark,
            input_path=settings.cnefe_parquet_path,
            output_path=settings.cnefe_normalized_parquet_path,
        )
        write_normalization_report(
            settings.normalization_report_path,
            search_metrics=analyze_search_normalization(search),
            cnefe_metrics=analyze_cnefe_normalization(cnefe),
        )
        LOGGER.info(
            "Relatorio de normalizacao gravado em %s",
            settings.normalization_report_path,
        )
    finally:
        spark.stop()


def run_matching_baseline() -> None:
    spark = create_spark_session(
        app_name=settings.spark_app_name,
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


def run_check_elasticsearch() -> None:
    client = create_elasticsearch_client(
        settings.elasticsearch_url,
        request_timeout_seconds=settings.elasticsearch_request_timeout_seconds,
    )
    try:
        status = check_elasticsearch(client)
        LOGGER.info(
            "Elasticsearch disponivel: cluster=%s, versao=%s, saude=%s, nos=%s",
            status["cluster_name"],
            status["version"],
            status["health_status"],
            status["number_of_nodes"],
        )
    finally:
        client.close()


def run_index_cnefe(*, recreate_index: bool) -> None:
    client = create_elasticsearch_client(
        settings.elasticsearch_url,
        request_timeout_seconds=settings.elasticsearch_request_timeout_seconds,
    )
    spark = create_spark_session(
        app_name=f"{settings.spark_app_name}-elasticsearch-index",
        master=settings.spark_master,
    )
    try:
        report = index_cnefe_parquet(
            spark,
            client=client,
            elasticsearch_url=settings.elasticsearch_url,
            index_name=settings.elasticsearch_index_name,
            parquet_path=settings.cnefe_normalized_parquet_path,
            recreate_index=recreate_index,
            chunk_size=settings.elasticsearch_bulk_chunk_size,
            request_timeout_seconds=(
                settings.elasticsearch_request_timeout_seconds
            ),
            max_retries=settings.elasticsearch_bulk_max_retries,
        )
        write_index_report(settings.elasticsearch_index_report_path, report)
        LOGGER.info(
            "Relatorio de indexacao gravado em %s",
            settings.elasticsearch_index_report_path,
        )
        if report.failed_bulk_operations:
            raise BulkIndexingError(
                "Indexacao terminou com "
                f"{report.failed_bulk_operations} falhas. "
                f"Amostras: {report.error_samples}"
            )
    finally:
        spark.stop()
        client.close()


def run_validate_index() -> None:
    client = create_elasticsearch_client(
        settings.elasticsearch_url,
        request_timeout_seconds=settings.elasticsearch_request_timeout_seconds,
    )
    try:
        check_elasticsearch(client)
        report = validate_technical_queries(
            client,
            index_name=settings.elasticsearch_index_name,
        )
        write_validation_report(
            settings.elasticsearch_validation_report_path,
            report,
        )
        LOGGER.info(
            "Consultas tecnicas validadas; relatorio=%s",
            settings.elasticsearch_validation_report_path,
        )
    finally:
        client.close()


def run_retrieve_candidates() -> None:
    client = create_elasticsearch_client(
        settings.elasticsearch_url,
        request_timeout_seconds=settings.elasticsearch_request_timeout_seconds,
    )
    try:
        check_elasticsearch(client)
        if not bool(
            client.indices.exists(index=settings.elasticsearch_index_name)
        ):
            raise FileNotFoundError(
                "Indice Elasticsearch nao encontrado: "
                f"{settings.elasticsearch_index_name}. Execute index-cnefe."
            )
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
    finally:
        client.close()


def run_select_candidates() -> None:
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


def run_export_results() -> None:
    export_results(
        original_path=settings.search_input_path,
        selection_audit_path=settings.candidate_selection_audit_path,
        official_path=settings.final_result_path,
        audit_path=settings.final_audit_path,
        metrics_path=settings.final_metrics_path,
        expected_records=settings.expected_search_records,
    )


def run_pipeline() -> None:
    execute_end_to_end_pipeline(settings)


def run_evaluate_results(*, with_relaxed_query_experiment: bool) -> None:
    client = None
    if with_relaxed_query_experiment:
        client = create_elasticsearch_client(
            settings.elasticsearch_url,
            request_timeout_seconds=(
                settings.elasticsearch_request_timeout_seconds
            ),
        )
    try:
        if client is not None:
            check_elasticsearch(client)
            if not bool(
                client.indices.exists(index=settings.elasticsearch_index_name)
            ):
                raise FileNotFoundError(
                    "Indice Elasticsearch nao encontrado: "
                    f"{settings.elasticsearch_index_name}. Execute index-cnefe."
                )
        execute_evaluation(
            final_audit_path=settings.final_audit_path,
            evidence_path=settings.candidate_selection_evidence_path,
            selection_metrics_path=settings.candidate_selection_metrics_path,
            retrieval_metrics_path=settings.candidate_retrieval_metrics_path,
            normalization_report_path=settings.normalization_report_path,
            diagnostics_path=settings.candidate_retrieval_diagnostics_path,
            normalized_search_path=settings.search_normalized_path,
            cases_path=settings.evaluation_cases_path,
            low_confidence_path=settings.evaluation_low_confidence_path,
            metrics_path=settings.evaluation_metrics_path,
            report_path=settings.evaluation_report_path,
            policy=SelectionPolicy(
                strong_street_similarity=(
                    settings.candidate_selection_strong_street_similarity
                ),
                plausible_street_similarity=(
                    settings.candidate_selection_plausible_street_similarity
                ),
            ),
            near_threshold_band=settings.evaluation_near_threshold_band,
            expected_records=settings.expected_search_records,
            client=client,
            index_name=(
                settings.elasticsearch_index_name if client is not None else None
            ),
            top_k=settings.candidate_retrieval_top_k,
        )
    finally:
        if client is not None:
            client.close()


def main(argv: Sequence[str] | None = None) -> int:
    """Executa o subcomando solicitado e retorna zero quando concluído."""
    configure_logging()
    args = build_parser().parse_args(argv)
    if args.command == "prepare-cnefe":
        run_prepare_cnefe()
    elif args.command == "validate-search":
        run_validate_search()
    elif args.command == "normalize-search":
        run_normalize_search()
    elif args.command == "normalize-cnefe":
        run_normalize_cnefe()
    elif args.command == "analyze-normalization":
        run_analyze_normalization()
    elif args.command == "normalize-data":
        run_normalize_data()
    elif args.command == "run-baseline":
        run_matching_baseline()
    elif args.command == "check-elasticsearch":
        run_check_elasticsearch()
    elif args.command == "index-cnefe":
        run_index_cnefe(recreate_index=args.recreate_index)
    elif args.command == "validate-index":
        run_validate_index()
    elif args.command == "retrieve-candidates":
        run_retrieve_candidates()
    elif args.command == "select-candidates":
        run_select_candidates()
    elif args.command == "export-results":
        run_export_results()
    elif args.command == "run-pipeline":
        run_pipeline()
    elif args.command == "evaluate-results":
        run_evaluate_results(
            with_relaxed_query_experiment=(
                args.with_relaxed_query_experiment
            )
        )
    return 0


EXPECTED_CLI_ERRORS = (
    BulkIndexingError,
    ConfigurationError,
    DatasetValidationError,
    ElasticsearchUnavailableError,
    ExportValidationError,
    FileNotFoundError,
    IndexAlreadyExistsError,
    IndexValidationError,
    PipelinePrerequisiteError,
)


def cli(argv: Sequence[str] | None = None) -> int:
    """Converte falhas operacionais conhecidas em mensagens curtas."""
    try:
        return main(argv)
    except EXPECTED_CLI_ERRORS as error:
        LOGGER.error("%s", error)
        return 2


if __name__ == "__main__":
    raise SystemExit(cli())
