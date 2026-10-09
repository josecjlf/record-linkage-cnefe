"""Validacoes deterministicas compartilhadas pelos leitores."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Collection, Sequence


class DatasetValidationError(ValueError):
    """Indica que um arquivo existe, mas viola o contrato do dataset."""


def validate_input_file(path: Path, dataset_name: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Arquivo de {dataset_name} nao encontrado: {path}. "
            "Posicione o arquivo conforme o README ou ajuste o caminho "
            "correspondente no .env."
        )
    if path.stat().st_size == 0:
        raise DatasetValidationError(
            f"Arquivo de {dataset_name} esta vazio: {path}"
        )


def read_csv_header(
    path: Path, *, delimiter: str = ";", encoding: str = "utf-8-sig"
) -> tuple[str, ...]:
    with path.open("r", encoding=encoding, errors="strict", newline="") as source:
        header = next(csv.reader(source, delimiter=delimiter), None)
    if not header:
        raise DatasetValidationError(f"CSV sem cabecalho: {path}")
    return tuple(header)


def validate_required_columns(
    actual_columns: Collection[str],
    required_columns: Collection[str],
    dataset_name: str,
) -> None:
    missing = sorted(set(required_columns) - set(actual_columns))
    if missing:
        raise DatasetValidationError(
            f"{dataset_name} sem colunas obrigatorias: {', '.join(missing)}"
        )


def validate_exact_schema_order(
    actual_columns: Sequence[str],
    expected_columns: Sequence[str],
    dataset_name: str,
) -> None:
    if tuple(actual_columns) == tuple(expected_columns):
        return
    validate_required_columns(actual_columns, expected_columns, dataset_name)
    raise DatasetValidationError(
        f"Ordem ou quantidade de colunas inesperada em {dataset_name}. "
        "O schema explicito do CSV depende da ordem definida no contrato de entrada."
    )
