"""Ingestao da pequena base de enderecos de busca com Pandas."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.ingestion.schemas import SEARCH_COLUMNS, SEARCH_ID_COLUMN
from src.ingestion.validation import (
    DatasetValidationError,
    read_csv_header,
    validate_input_file,
    validate_required_columns,
)


def read_search_addresses(path: Path) -> pd.DataFrame:
    """Le o CSV preservando todas as colunas e valores como texto."""
    return pd.read_csv(
        path,
        sep=";",
        encoding="utf-8-sig",
        dtype="string",
        keep_default_na=False,
    )


def validate_search_addresses(
    dataframe: pd.DataFrame, *, expected_records: int | None
) -> None:
    validate_required_columns(
        dataframe.columns, SEARCH_COLUMNS, "base de busca"
    )
    if dataframe.empty:
        raise DatasetValidationError("Base de busca nao possui registros.")
    if expected_records is not None and len(dataframe) != expected_records:
        raise DatasetValidationError(
            "Quantidade inesperada na base de busca: "
            f"esperado={expected_records}, encontrado={len(dataframe)}."
        )


def add_stable_search_id(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Adiciona um ID deterministico baseado na ordem original do arquivo."""
    if SEARCH_ID_COLUMN in dataframe.columns:
        raise DatasetValidationError(
            f"A entrada ja possui a coluna reservada {SEARCH_ID_COLUMN}."
        )
    result = dataframe.copy(deep=True)
    identifiers = pd.Series(
        [f"search_{position:06d}" for position in range(1, len(result) + 1)],
        index=result.index,
        dtype="string",
    )
    result.insert(0, SEARCH_ID_COLUMN, identifiers)
    return result


def ingest_search_addresses(
    path: Path, *, expected_records: int | None
) -> pd.DataFrame:
    """Orquestra leitura, validacao e adicao do identificador interno."""
    validate_input_file(path, "base de busca")
    header = read_csv_header(path)
    validate_required_columns(header, SEARCH_COLUMNS, "base de busca")
    dataframe = read_search_addresses(path)
    validate_search_addresses(dataframe, expected_records=expected_records)
    return add_stable_search_id(dataframe)

