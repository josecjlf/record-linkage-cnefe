"""Configuracao central do projeto, baseada em variaveis de ambiente."""

from __future__ import annotations

from dataclasses import dataclass, fields
from itertools import combinations
import os
from pathlib import Path
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ConfigurationError(ValueError):
    """Uma variavel de ambiente possui valor ausente ou incoerente."""


def _env_text(variable_name: str, default: str) -> str:
    value = os.getenv(variable_name, default).strip()
    if not value:
        raise ConfigurationError(
            f"Configuracao {variable_name} nao pode ser vazia."
        )
    return value


def _env_int(variable_name: str, default: int) -> int:
    raw_value = _env_text(variable_name, str(default))
    try:
        return int(raw_value)
    except ValueError as error:
        raise ConfigurationError(
            f"Configuracao {variable_name} deve ser um inteiro; "
            f"recebido={raw_value!r}."
        ) from error


def _env_float(variable_name: str, default: float) -> float:
    raw_value = _env_text(variable_name, str(default))
    try:
        return float(raw_value)
    except ValueError as error:
        raise ConfigurationError(
            f"Configuracao {variable_name} deve ser numerica; "
            f"recebido={raw_value!r}."
        ) from error


def _project_path(variable_name: str, default: str) -> Path:
    """Resolve um caminho absoluto ou relativo a raiz do projeto."""
    configured_path = Path(_env_text(variable_name, default)).expanduser()
    if configured_path.is_absolute():
        return configured_path
    return PROJECT_ROOT / configured_path


@dataclass(frozen=True)
class Settings:
    """Valores configuraveis compartilhados pelas etapas do pipeline."""

    log_level: str
    cnefe_input_path: Path
    search_input_path: Path
    cnefe_parquet_path: Path
    cnefe_normalized_parquet_path: Path
    search_normalized_path: Path
    normalization_report_path: Path
    baseline_output_path: Path
    baseline_metrics_path: Path
    elasticsearch_url: str
    elasticsearch_index_name: str
    elasticsearch_bulk_chunk_size: int
    elasticsearch_request_timeout_seconds: int
    elasticsearch_bulk_max_retries: int
    elasticsearch_index_report_path: Path
    elasticsearch_validation_report_path: Path
    candidate_retrieval_top_k: int
    candidate_retrieval_minimum_candidates: int
    candidate_retrieval_diagnostics_path: Path
    candidate_retrieval_candidates_path: Path
    candidate_retrieval_metrics_path: Path
    candidate_retrieval_analysis_path: Path
    candidate_selection_strong_street_similarity: float
    candidate_selection_plausible_street_similarity: float
    candidate_selection_audit_path: Path
    candidate_selection_evidence_path: Path
    candidate_selection_metrics_path: Path
    matching_strategy_report_path: Path
    final_result_path: Path
    final_audit_path: Path
    final_metrics_path: Path
    evaluation_cases_path: Path
    evaluation_low_confidence_path: Path
    evaluation_metrics_path: Path
    evaluation_report_path: Path
    evaluation_near_threshold_band: float
    expected_uf_code: str
    expected_search_records: int
    spark_master: str
    spark_app_name: str

    @classmethod
    def from_env(cls) -> "Settings":
        configured = cls(
            log_level=_env_text("LOG_LEVEL", "INFO").upper(),
            cnefe_input_path=_project_path("CNEFE_INPUT_PATH", "29_BA.csv"),
            search_input_path=_project_path(
                "SEARCH_INPUT_PATH", "2026_teste_tecnico_relinke_cidacs(in).csv"
            ),
            cnefe_parquet_path=_project_path(
                "CNEFE_PARQUET_PATH", "data/processed/cnefe_bahia.parquet"
            ),
            cnefe_normalized_parquet_path=_project_path(
                "CNEFE_NORMALIZED_PARQUET_PATH",
                "data/processed/cnefe_bahia_normalized.parquet",
            ),
            search_normalized_path=_project_path(
                "SEARCH_NORMALIZED_PATH",
                "data/processed/search_addresses_normalized.csv",
            ),
            normalization_report_path=_project_path(
                "NORMALIZATION_REPORT_PATH",
                "data/processed/normalization_report.json",
            ),
            baseline_output_path=_project_path(
                "BASELINE_OUTPUT_PATH", "data/output/baseline_matches.csv"
            ),
            baseline_metrics_path=_project_path(
                "BASELINE_METRICS_PATH", "data/output/baseline_metrics.json"
            ),
            elasticsearch_url=_env_text(
                "ELASTICSEARCH_URL", "http://localhost:9200"
            ),
            elasticsearch_index_name=_env_text(
                "ELASTICSEARCH_INDEX_NAME", "cnefe-bahia-addresses-v1"
            ),
            elasticsearch_bulk_chunk_size=_env_int(
                "ELASTICSEARCH_BULK_CHUNK_SIZE", 2000
            ),
            elasticsearch_request_timeout_seconds=_env_int(
                "ELASTICSEARCH_REQUEST_TIMEOUT_SECONDS", 60
            ),
            elasticsearch_bulk_max_retries=_env_int(
                "ELASTICSEARCH_BULK_MAX_RETRIES", 3
            ),
            elasticsearch_index_report_path=_project_path(
                "ELASTICSEARCH_INDEX_REPORT_PATH",
                "data/output/elasticsearch_index_report.json",
            ),
            elasticsearch_validation_report_path=_project_path(
                "ELASTICSEARCH_VALIDATION_REPORT_PATH",
                "data/output/elasticsearch_validation_report.json",
            ),
            candidate_retrieval_top_k=_env_int(
                "CANDIDATE_RETRIEVAL_TOP_K", 10
            ),
            candidate_retrieval_minimum_candidates=_env_int(
                "CANDIDATE_RETRIEVAL_MINIMUM_CANDIDATES", 5
            ),
            candidate_retrieval_diagnostics_path=_project_path(
                "CANDIDATE_RETRIEVAL_DIAGNOSTICS_PATH",
                "data/output/candidate_retrieval_diagnostics.jsonl",
            ),
            candidate_retrieval_candidates_path=_project_path(
                "CANDIDATE_RETRIEVAL_CANDIDATES_PATH",
                "data/output/candidate_retrieval_candidates.csv",
            ),
            candidate_retrieval_metrics_path=_project_path(
                "CANDIDATE_RETRIEVAL_METRICS_PATH",
                "data/output/candidate_retrieval_metrics.json",
            ),
            candidate_retrieval_analysis_path=_project_path(
                "CANDIDATE_RETRIEVAL_ANALYSIS_PATH",
                "docs/candidate_retrieval_analysis.md",
            ),
            candidate_selection_strong_street_similarity=_env_float(
                "CANDIDATE_SELECTION_STRONG_STREET_SIMILARITY", 0.90
            ),
            candidate_selection_plausible_street_similarity=_env_float(
                "CANDIDATE_SELECTION_PLAUSIBLE_STREET_SIMILARITY", 0.85
            ),
            candidate_selection_audit_path=_project_path(
                "CANDIDATE_SELECTION_AUDIT_PATH",
                "data/output/candidate_selection_audit.csv",
            ),
            candidate_selection_evidence_path=_project_path(
                "CANDIDATE_SELECTION_EVIDENCE_PATH",
                "data/output/candidate_selection_evidence.csv",
            ),
            candidate_selection_metrics_path=_project_path(
                "CANDIDATE_SELECTION_METRICS_PATH",
                "data/output/candidate_selection_metrics.json",
            ),
            matching_strategy_report_path=_project_path(
                "MATCHING_STRATEGY_REPORT_PATH",
                "docs/matching_strategy.md",
            ),
            final_result_path=_project_path(
                "FINAL_RESULT_PATH", "data/output/result.csv"
            ),
            final_audit_path=_project_path(
                "FINAL_AUDIT_PATH", "data/output/result_audit.csv"
            ),
            final_metrics_path=_project_path(
                "FINAL_METRICS_PATH", "data/output/result_metrics.json"
            ),
            evaluation_cases_path=_project_path(
                "EVALUATION_CASES_PATH",
                "data/output/evaluation_cases.csv",
            ),
            evaluation_low_confidence_path=_project_path(
                "EVALUATION_LOW_CONFIDENCE_PATH",
                "data/output/low_confidence_cases.csv",
            ),
            evaluation_metrics_path=_project_path(
                "EVALUATION_METRICS_PATH",
                "data/output/evaluation_metrics.json",
            ),
            evaluation_report_path=_project_path(
                "EVALUATION_REPORT_PATH", "docs/evaluation.md"
            ),
            evaluation_near_threshold_band=_env_float(
                "EVALUATION_NEAR_THRESHOLD_BAND", 0.05
            ),
            expected_uf_code=_env_text("EXPECTED_UF_CODE", "29"),
            expected_search_records=_env_int(
                "EXPECTED_SEARCH_RECORDS", 107
            ),
            spark_master=_env_text("SPARK_MASTER", "local[4]"),
            spark_app_name=_env_text(
                "SPARK_APP_NAME", "cnefe-bahia-ingestion"
            ),
        )
        configured.validate()
        return configured

    def validate(self) -> None:
        """Falha cedo com mensagens ligadas as configuracoes publicas."""
        configured_paths = {
            field.name.upper(): getattr(self, field.name).resolve()
            for field in fields(self)
            if field.name.endswith("_path")
        }
        for name in ("FINAL_RESULT_PATH", "FINAL_AUDIT_PATH", "FINAL_METRICS_PATH"):
            output_path = configured_paths[name]
            configured_paths[f"{name} (temporario)"] = output_path.with_suffix(
                f"{output_path.suffix}.tmp"
            ).resolve()
        for (left_name, left_path), (right_name, right_path) in combinations(
            configured_paths.items(), 2
        ):
            if (
                left_path == right_path
                or left_path in right_path.parents
                or right_path in left_path.parents
            ):
                raise ConfigurationError(
                    f"Caminhos conflitantes: {left_name}={left_path} e "
                    f"{right_name}={right_path}. Entradas e artefatos devem "
                    "usar caminhos distintos, sem conter uns aos outros."
                )
        parsed_url = urlparse(self.elasticsearch_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ConfigurationError(
                "ELASTICSEARCH_URL deve ser uma URL HTTP(S) completa; "
                f"recebido={self.elasticsearch_url!r}."
            )
        if self.elasticsearch_index_name.lower() != self.elasticsearch_index_name:
            raise ConfigurationError(
                "ELASTICSEARCH_INDEX_NAME deve conter somente letras minusculas."
            )
        if self.elasticsearch_index_name in {".", ".."} or (
            self.elasticsearch_index_name.startswith(("-", "_", "+"))
        ):
            raise ConfigurationError(
                "ELASTICSEARCH_INDEX_NAME nao pode ser '.', '..' nem iniciar "
                "com '-', '_' ou '+'."
            )
        if any(
            character in self.elasticsearch_index_name
            for character in ('\\', '/', '*', '?', '"', '<', '>', '|', ' ', ',', '#')
        ):
            raise ConfigurationError(
                "ELASTICSEARCH_INDEX_NAME contem caractere invalido: "
                f"{self.elasticsearch_index_name!r}."
            )
        positive_integers = {
            "ELASTICSEARCH_BULK_CHUNK_SIZE": self.elasticsearch_bulk_chunk_size,
            "ELASTICSEARCH_REQUEST_TIMEOUT_SECONDS": (
                self.elasticsearch_request_timeout_seconds
            ),
            "CANDIDATE_RETRIEVAL_TOP_K": self.candidate_retrieval_top_k,
            "CANDIDATE_RETRIEVAL_MINIMUM_CANDIDATES": (
                self.candidate_retrieval_minimum_candidates
            ),
            "EXPECTED_SEARCH_RECORDS": self.expected_search_records,
        }
        for name, value in positive_integers.items():
            if value <= 0:
                raise ConfigurationError(
                    f"Configuracao {name} deve ser maior que zero; recebido={value}."
                )
        if self.elasticsearch_bulk_max_retries < 0:
            raise ConfigurationError(
                "ELASTICSEARCH_BULK_MAX_RETRIES nao pode ser negativo."
            )
        if (
            self.candidate_retrieval_minimum_candidates
            > self.candidate_retrieval_top_k
        ):
            raise ConfigurationError(
                "CANDIDATE_RETRIEVAL_MINIMUM_CANDIDATES nao pode superar "
                "CANDIDATE_RETRIEVAL_TOP_K."
            )
        thresholds = {
            "CANDIDATE_SELECTION_STRONG_STREET_SIMILARITY": (
                self.candidate_selection_strong_street_similarity
            ),
            "CANDIDATE_SELECTION_PLAUSIBLE_STREET_SIMILARITY": (
                self.candidate_selection_plausible_street_similarity
            ),
        }
        for name, value in thresholds.items():
            if not 0 <= value <= 1:
                raise ConfigurationError(
                    f"Configuracao {name} deve estar entre zero e um."
                )
        if (
            self.candidate_selection_plausible_street_similarity
            > self.candidate_selection_strong_street_similarity
        ):
            raise ConfigurationError(
                "O threshold plausivel nao pode superar o threshold forte."
            )
        if self.evaluation_near_threshold_band < 0:
            raise ConfigurationError(
                "EVALUATION_NEAR_THRESHOLD_BAND nao pode ser negativo."
            )
        if not (len(self.expected_uf_code) == 2 and self.expected_uf_code.isdigit()):
            raise ConfigurationError(
                "EXPECTED_UF_CODE deve conter exatamente dois digitos."
            )


try:
    settings = Settings.from_env()
except ConfigurationError as error:
    raise SystemExit(f"Configuracao invalida: {error}") from None
