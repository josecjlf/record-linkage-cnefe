"""Diagnosticos exatos para auditar os resultados da baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd
from pyspark.sql import DataFrame, functions as F


@dataclass(frozen=True)
class NotFoundDiagnostic:
    input_id: str
    category: str
    exact_street_candidate_rows: int
    exact_street_postal_code_rows: int


def _has_value(value: str | None) -> bool:
    return value is not None and value.strip() != ""


def diagnose_not_found(
    cnefe: DataFrame,
    search: DataFrame,
    input_ids: Sequence[str],
) -> tuple[NotFoundDiagnostic, ...]:
    """Explica falhas por igualdade exata, sem criar uma regra de matching."""
    if not input_ids:
        return ()

    search_subset = (
        search.where(F.col("input_id").isin(sorted(input_ids)))
        .select(
            "input_id",
            "municipio_codigo_normalizado",
            "cep_normalizado",
            "logradouro_normalizado",
            "numero_comparavel_normalizado",
        )
        .dropDuplicates(["input_id"])
    )
    input_rows = {
        row["input_id"]: row.asDict(recursive=False)
        for row in search_subset.collect()
    }

    search_alias = F.broadcast(search_subset).alias("search")
    cnefe_alias = cnefe.alias("cnefe")
    same_street = cnefe_alias.join(
        search_alias,
        (
            F.col("cnefe.municipio_codigo_normalizado")
            == F.col("search.municipio_codigo_normalizado")
        )
        & (
            F.col("cnefe.logradouro_normalizado")
            == F.col("search.logradouro_normalizado")
        ),
        "inner",
    )
    aggregates = (
        same_street.groupBy(F.col("search.input_id").alias("input_id"))
        .agg(
            F.count(F.lit(1)).alias("exact_street_candidate_rows"),
            F.sum(
                F.when(
                    F.col("cnefe.cep_normalizado")
                    == F.col("search.cep_normalizado"),
                    1,
                ).otherwise(0)
            ).alias("exact_street_postal_code_rows"),
            F.sum(
                F.when(
                    F.col("cnefe.numero_comparavel_normalizado")
                    == F.col("search.numero_comparavel_normalizado"),
                    1,
                ).otherwise(0)
            ).alias("exact_street_number_rows"),
        )
        .collect()
    )
    aggregate_by_input = {
        row["input_id"]: row.asDict(recursive=False) for row in aggregates
    }

    diagnostics: list[NotFoundDiagnostic] = []
    for input_id in input_ids:
        values = input_rows[input_id]
        aggregate = aggregate_by_input.get(input_id, {})
        street_rows = int(aggregate.get("exact_street_candidate_rows", 0))
        postal_code_rows = int(
            aggregate.get("exact_street_postal_code_rows", 0)
        )
        number_rows = int(aggregate.get("exact_street_number_rows", 0))

        if not _has_value(values["municipio_codigo_normalizado"]) or not _has_value(
            values["logradouro_normalizado"]
        ):
            category = "MISSING_MUNICIPIO_OR_LOGRADOURO"
        elif street_rows == 0:
            category = "NO_EXACT_MUNICIPIO_LOGRADOURO"
        elif number_rows == 0 and postal_code_rows > 0:
            category = "EXACT_LOGRADOURO_CEP_BUT_NUMBER_DIFFERS"
        elif number_rows == 0:
            category = "EXACT_LOGRADOURO_BUT_NUMBER_DIFFERS"
        else:
            category = "OTHER_EXACT_CONFLICT"

        diagnostics.append(
            NotFoundDiagnostic(
                input_id=input_id,
                category=category,
                exact_street_candidate_rows=street_rows,
                exact_street_postal_code_rows=postal_code_rows,
            )
        )
    return tuple(diagnostics)


def add_audit_diagnostics(
    result: pd.DataFrame,
    not_found_diagnostics: Sequence[NotFoundDiagnostic],
) -> pd.DataFrame:
    """Adiciona um motivo auditavel e preserva todas as linhas da busca."""
    not_found_by_id = {
        diagnostic.input_id: diagnostic for diagnostic in not_found_diagnostics
    }

    def diagnostic_for_row(row: pd.Series) -> str:
        if row["status_match"] == "NOT_FOUND":
            return not_found_by_id[row["input_id"]].category
        if row["status_match"] == "AMBIGUOUS":
            return "MULTIPLE_CANDIDATE_SECTORS"
        if int(row["quantidade_candidatos"]) == 1:
            return "UNIQUE_CANDIDATE"
        return "MULTIPLE_CANDIDATES_SAME_SECTOR"

    audited = result.copy(deep=True)
    audited["diagnostico_baseline"] = audited.apply(diagnostic_for_row, axis=1)
    return audited


def summarize_audit(result: pd.DataFrame) -> dict[str, object]:
    """Resume categorias observadas sem inferir similaridade textual."""
    not_found = result.loc[result["status_match"] == "NOT_FOUND"]
    found = result.loc[result["status_match"] == "FOUND"]
    ambiguous = result.loc[result["status_match"] == "AMBIGUOUS"]

    not_found_patterns = {
        "numero_ausente_ou_sn": int(
            not_found["numero_comparavel_normalizado"].fillna("").isin(["", "sn"]).sum()
        ),
        "cep_ausente": int(
            not_found["cep_normalizado"].fillna("").eq("").sum()
        ),
        "logradouro_com_digito": int(
            not_found["logradouro_normalizado"]
            .fillna("")
            .str.contains(r"\d", regex=True)
            .sum()
        ),
        "bairro_ausente": int(
            not_found["bairro_normalizado"].fillna("").eq("").sum()
        ),
        "complemento_presente": int(
            not_found["complemento_normalizado"].fillna("").ne("").sum()
        ),
    }
    return {
        "found": {
            "unique_candidate": int(
                found["quantidade_candidatos"].eq(1).sum()
            ),
            "multiple_candidates_same_sector": int(
                found["quantidade_candidatos"].gt(1).sum()
            ),
        },
        "not_found": {
            "categories": {
                str(category): int(count)
                for category, count in not_found[
                    "diagnostico_baseline"
                ].value_counts().sort_index().items()
            },
            "patterns": not_found_patterns,
        },
        "ambiguous": {
            "candidate_count_min": (
                int(ambiguous["quantidade_candidatos"].min())
                if not ambiguous.empty
                else 0
            ),
            "candidate_count_max": (
                int(ambiguous["quantidade_candidatos"].max())
                if not ambiguous.empty
                else 0
            ),
            "sector_count_min": (
                int(ambiguous["quantidade_setores_candidatos"].min())
                if not ambiguous.empty
                else 0
            ),
            "sector_count_max": (
                int(ambiguous["quantidade_setores_candidatos"].max())
                if not ambiguous.empty
                else 0
            ),
        },
    }
