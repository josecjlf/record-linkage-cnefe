"""Orquestracao, metricas e exportacao da baseline exata."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import logging
from pathlib import Path
from typing import Sequence

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StringType, StructField, StructType

from src.ingestion.validation import (
    DatasetValidationError,
    validate_required_columns,
)
from src.matching.baseline_strategies import (
    BASELINE_STRATEGIES,
    ExactMatchStrategy,
)
from src.matching.baseline_analysis import (
    add_audit_diagnostics,
    diagnose_not_found,
    summarize_audit,
)
from src.matching.candidate_retriever import retrieve_candidate_summaries
from src.matching.candidate_selector import (
    MatchDecision,
    MatchStatus,
    not_found_decision,
    select_candidate,
)


LOGGER = logging.getLogger(__name__)

SEARCH_MATCH_COLUMNS = (
    "input_id",
    "municipio_codigo_normalizado",
    "cep_normalizado",
    "logradouro_normalizado",
    "numero_comparavel_normalizado",
    "bairro_normalizado",
)

CNEFE_MATCH_COLUMNS = (
    "COD_UNICO_ENDERECO",
    "COD_SETOR",
    "municipio_codigo_normalizado",
    "cep_normalizado",
    "logradouro_normalizado",
    "numero_comparavel_normalizado",
    "localidade_normalizada",
)


@dataclass(frozen=True)
class BaselineRun:
    decisions: tuple[MatchDecision, ...]
    metrics: dict[str, object]


@dataclass(frozen=True)
class BaselineOutput:
    result_path: Path
    metrics_path: Path
    total_records: int


def _status_count(decisions: Sequence[MatchDecision], status: MatchStatus) -> int:
    return sum(decision.status_match == status for decision in decisions)


def run_baseline(
    cnefe: DataFrame,
    search: DataFrame,
    *,
    strategies: Sequence[ExactMatchStrategy] = BASELINE_STRATEGIES,
) -> BaselineRun:
    """Executa regras em cascata e encerra ao encontrar candidatos exatos."""
    validate_required_columns(cnefe.columns, CNEFE_MATCH_COLUMNS, "CNEFE")
    validate_required_columns(search.columns, SEARCH_MATCH_COLUMNS, "busca")
    input_ids = [row["input_id"] for row in search.select("input_id").collect()]
    if not input_ids:
        raise DatasetValidationError("Base de busca normalizada esta vazia.")
    if len(input_ids) != len(set(input_ids)):
        raise DatasetValidationError("Base de busca possui input_id duplicado.")

    unresolved = set(input_ids)
    decisions: dict[str, MatchDecision] = {}
    strategy_metrics: list[dict[str, object]] = []
    for order, strategy in enumerate(strategies, start=1):
        retrieval = retrieve_candidate_summaries(
            cnefe,
            search,
            strategy,
            input_ids=unresolved,
        )
        strategy_decisions = [
            select_candidate(summary) for summary in retrieval.summaries
        ]
        for decision in strategy_decisions:
            decisions[decision.input_id] = decision
            unresolved.discard(decision.input_id)

        found = _status_count(strategy_decisions, MatchStatus.FOUND)
        ambiguous = _status_count(strategy_decisions, MatchStatus.AMBIGUOUS)
        strategy_metrics.append(
            {
                "order": order,
                "strategy": strategy.name,
                "eligible_inputs": retrieval.eligible_input_count,
                "inputs_with_candidates": len(strategy_decisions),
                "found": found,
                "ambiguous": ambiguous,
            }
        )
        LOGGER.info(
            "Baseline %s: elegiveis=%s, candidatos=%s, found=%s, ambiguous=%s",
            strategy.name,
            retrieval.eligible_input_count,
            len(strategy_decisions),
            found,
            ambiguous,
        )

    for input_id in unresolved:
        decisions[input_id] = not_found_decision(input_id)

    ordered_decisions = tuple(decisions[input_id] for input_id in input_ids)
    total = len(ordered_decisions)
    found = _status_count(ordered_decisions, MatchStatus.FOUND)
    ambiguous = _status_count(ordered_decisions, MatchStatus.AMBIGUOUS)
    not_found = _status_count(ordered_decisions, MatchStatus.NOT_FOUND)
    metrics = {
        "total": total,
        "found": found,
        "not_found": not_found,
        "ambiguous": ambiguous,
        "found_percentage": round((found / total) * 100, 2),
        "strategies": strategy_metrics,
    }
    return BaselineRun(decisions=ordered_decisions, metrics=metrics)


def read_normalized_search(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Busca normalizada nao encontrada: {path}")
    dataframe = pd.read_csv(
        path,
        sep=";",
        encoding="utf-8",
        dtype="string",
        keep_default_na=False,
    )
    validate_required_columns(
        dataframe.columns,
        (
            "search_id",
            "municipio_codigo_normalizado",
            "cep_normalizado",
            "logradouro_normalizado",
            "numero_comparavel_normalizado",
            "bairro_normalizado",
        ),
        "busca normalizada",
    )
    if dataframe.empty:
        raise DatasetValidationError("Busca normalizada nao possui registros.")
    return dataframe


def search_pandas_to_spark(
    spark: SparkSession, dataframe: pd.DataFrame
) -> DataFrame:
    selected = dataframe.assign(input_id=dataframe["search_id"])
    records = [
        tuple(
            str(value) if not pd.isna(value) and value != "" else None
            for value in row
        )
        for row in selected.loc[:, list(SEARCH_MATCH_COLUMNS)].itertuples(
            index=False, name=None
        )
    ]
    schema = StructType(
        [StructField(column, StringType(), nullable=True) for column in SEARCH_MATCH_COLUMNS]
    )
    return spark.createDataFrame(records, schema=schema)


def execute_baseline(
    spark: SparkSession,
    *,
    cnefe_path: Path,
    search_path: Path,
    result_path: Path,
    metrics_path: Path,
    strategies: Sequence[ExactMatchStrategy] = BASELINE_STRATEGIES,
) -> BaselineOutput:
    if not cnefe_path.is_dir():
        raise FileNotFoundError(f"CNEFE normalizado nao encontrado: {cnefe_path}")
    search_pandas = read_normalized_search(search_path)
    search_spark = search_pandas_to_spark(spark, search_pandas)
    cnefe = spark.read.parquet(str(cnefe_path)).select(*CNEFE_MATCH_COLUMNS)
    run = run_baseline(cnefe, search_spark, strategies=strategies)

    decision_frame = pd.DataFrame(
        [decision.as_record() for decision in run.decisions]
    )
    result = search_pandas.copy(deep=True)
    result.insert(1, "input_id", result["search_id"])
    result = result.merge(
        decision_frame,
        on="input_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if len(result) != len(search_pandas):
        raise DatasetValidationError("Baseline perdeu registros da busca.")

    not_found_ids = [
        decision.input_id
        for decision in run.decisions
        if decision.status_match == MatchStatus.NOT_FOUND
    ]
    diagnostics = diagnose_not_found(cnefe, search_spark, not_found_ids)
    result = add_audit_diagnostics(result, diagnostics)

    result_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(result_path, sep=";", encoding="utf-8", index=False)
    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        **run.metrics,
        "audit": summarize_audit(result),
    }
    metrics_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    LOGGER.info(
        "Baseline concluida: total=%s, found=%s, not_found=%s, ambiguous=%s",
        run.metrics["total"],
        run.metrics["found"],
        run.metrics["not_found"],
        run.metrics["ambiguous"],
    )
    return BaselineOutput(
        result_path=result_path,
        metrics_path=metrics_path,
        total_records=len(result),
    )
