"""Constroi e valida os arquivos oficial e de auditoria."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
import logging
from pathlib import Path
import re

import pandas as pd

from src.ingestion.schemas import SEARCH_ID_COLUMN
from src.ingestion.search_addresses import (
    add_stable_search_id,
    read_search_addresses,
    validate_search_addresses,
)
from src.ingestion.validation import validate_required_columns


LOGGER = logging.getLogger(__name__)

FINAL_DECISION_COLUMNS = (
    "search_id",
    "setor_censitario",
    "status_match",
    "metodo_match",
)
VALID_STATUSES = frozenset({"FOUND", "AMBIGUOUS", "NOT_FOUND"})
SECTOR_CODE_PATTERN = re.compile(r"^[0-9]{15}[A-Z]?$", re.ASCII)


class ExportValidationError(ValueError):
    """Uma invariante da entrega final nao foi satisfeita."""


@dataclass(frozen=True)
class ExportMetrics:
    generated_at_utc: str
    input_records: int
    output_records: int
    original_column_count: int
    official_column_count: int
    found: int
    ambiguous: int
    not_found: int
    found_percentage: float
    official_path: str
    audit_path: str
    validations: tuple[str, ...]


@dataclass(frozen=True)
class ExportOutput:
    official_path: Path
    audit_path: Path
    metrics_path: Path
    metrics: ExportMetrics


def read_selection_audit(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Auditoria da candidate selection nao encontrada: {path}"
        )
    dataframe = pd.read_csv(
        path,
        sep=";",
        encoding="utf-8",
        dtype="string",
        keep_default_na=False,
    )
    validate_required_columns(
        dataframe.columns,
        FINAL_DECISION_COLUMNS,
        "auditoria da candidate selection",
    )
    if dataframe.empty:
        raise ExportValidationError("Auditoria da selection esta vazia.")
    return dataframe


def _validate_original_values(
    original: pd.DataFrame,
    result: pd.DataFrame,
) -> None:
    missing = [column for column in original.columns if column not in result]
    if missing:
        raise ExportValidationError(
            f"Colunas originais ausentes na saida: {missing}."
        )
    expected = original.reset_index(drop=True).astype("string").fillna("")
    actual = (
        result.loc[:, list(original.columns)]
        .reset_index(drop=True)
        .astype("string")
        .fillna("")
    )
    if not expected.equals(actual):
        raise ExportValidationError(
            "Valores ou ordem das colunas originais foram alterados."
        )


def validate_final_result(
    original: pd.DataFrame,
    result: pd.DataFrame,
) -> None:
    if len(result) != len(original):
        raise ExportValidationError(
            "Cardinalidade divergente: "
            f"entrada={len(original)}, saida={len(result)}."
        )
    validate_required_columns(
        result.columns, FINAL_DECISION_COLUMNS, "resultado final"
    )
    _validate_original_values(original, result)

    identifiers = result[SEARCH_ID_COLUMN].astype("string")
    if identifiers.isna().any() or (identifiers.str.strip() == "").any():
        raise ExportValidationError("A saida possui search_id vazio.")
    if identifiers.duplicated().any():
        raise ExportValidationError("A saida possui search_id duplicado.")

    statuses = result["status_match"].astype("string")
    invalid_statuses = sorted(set(statuses) - VALID_STATUSES)
    if invalid_statuses:
        raise ExportValidationError(
            f"Status de match invalidos: {invalid_statuses}."
        )
    methods = result["metodo_match"].astype("string")
    if methods.isna().any() or (methods.str.strip() == "").any():
        raise ExportValidationError("A saida possui metodo_match vazio.")

    sectors = result["setor_censitario"].astype("string").fillna("")
    sector_present = sectors.str.strip() != ""
    found = statuses == "FOUND"
    unresolved = statuses.isin(("AMBIGUOUS", "NOT_FOUND"))
    if (~sector_present[found]).any():
        raise ExportValidationError("Existe FOUND sem setor censitario.")
    if sector_present[unresolved].any():
        raise ExportValidationError(
            "AMBIGUOUS ou NOT_FOUND nao pode possuir setor censitario."
        )
    invalid_sector = sector_present & ~sectors.str.fullmatch(
        SECTOR_CODE_PATTERN
    )
    if invalid_sector.any():
        samples = list(sectors[invalid_sector].head(5))
        raise ExportValidationError(
            f"Codigo de setor com formato incoerente: {samples}."
        )


def build_export_frames(
    original: pd.DataFrame,
    selection_audit: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    original_with_id = add_stable_search_id(original)
    if selection_audit[SEARCH_ID_COLUMN].duplicated().any():
        raise ExportValidationError(
            "Auditoria da selection possui search_id duplicado."
        )
    if set(original_with_id[SEARCH_ID_COLUMN]) != set(
        selection_audit[SEARCH_ID_COLUMN]
    ):
        raise ExportValidationError(
            "A entrada original e a selection possuem chaves diferentes."
        )

    decisions = selection_audit.loc[:, list(FINAL_DECISION_COLUMNS)]
    official = original_with_id.merge(
        decisions,
        on=SEARCH_ID_COLUMN,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    official = official.loc[
        :,
        [
            *original.columns,
            SEARCH_ID_COLUMN,
            "setor_censitario",
            "status_match",
            "metodo_match",
        ],
    ]

    diagnostic_columns = [
        column
        for column in selection_audit.columns
        if column not in original.columns and column != SEARCH_ID_COLUMN
    ]
    audit_details = selection_audit.loc[
        :, [SEARCH_ID_COLUMN, *diagnostic_columns]
    ]
    audit = original_with_id.merge(
        audit_details,
        on=SEARCH_ID_COLUMN,
        how="left",
        validate="one_to_one",
        sort=False,
    )

    validate_final_result(original, official)
    validate_final_result(original, audit)
    return official, audit


def _write_csv_atomically(dataframe: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    dataframe.to_csv(
        temporary_path,
        sep=";",
        encoding="utf-8",
        index=False,
    )
    temporary_path.replace(path)


def _write_json_atomically(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def export_results(
    *,
    original_path: Path,
    selection_audit_path: Path,
    official_path: Path,
    audit_path: Path,
    metrics_path: Path,
    expected_records: int | None = None,
) -> ExportOutput:
    """Valida e exporta os arquivos oficial, de auditoria e de métricas."""
    original = read_search_addresses(original_path)
    validate_search_addresses(original, expected_records=expected_records)
    selection_audit = read_selection_audit(selection_audit_path)
    official, audit = build_export_frames(original, selection_audit)

    status_counts = official["status_match"].value_counts().to_dict()
    metrics = ExportMetrics(
        generated_at_utc=datetime.now(UTC).isoformat(),
        input_records=len(original),
        output_records=len(official),
        original_column_count=len(original.columns),
        official_column_count=len(official.columns),
        found=int(status_counts.get("FOUND", 0)),
        ambiguous=int(status_counts.get("AMBIGUOUS", 0)),
        not_found=int(status_counts.get("NOT_FOUND", 0)),
        found_percentage=round(
            int(status_counts.get("FOUND", 0)) / len(official) * 100,
            2,
        ),
        official_path=str(official_path),
        audit_path=str(audit_path),
        validations=(
            "INPUT_OUTPUT_CARDINALITY_EQUAL",
            "ALL_ORIGINAL_COLUMNS_AND_VALUES_PRESERVED",
            "SEARCH_ID_UNIQUE_AND_ORDER_PRESERVED",
            "FOUND_HAS_VALID_SECTOR",
            "UNRESOLVED_HAS_NO_SECTOR",
            "MATCH_STATUS_AND_METHOD_VALID",
        ),
    )
    _write_csv_atomically(official, official_path)
    _write_csv_atomically(audit, audit_path)
    _write_json_atomically(asdict(metrics), metrics_path)
    LOGGER.info(
        "Exportacao concluida: entrada=%s, saida=%s, found=%s, "
        "ambiguous=%s, not_found=%s, arquivo=%s",
        len(original),
        len(official),
        metrics.found,
        metrics.ambiguous,
        metrics.not_found,
        official_path,
    )
    return ExportOutput(
        official_path=official_path,
        audit_path=audit_path,
        metrics_path=metrics_path,
        metrics=metrics,
    )
