"""Contrato explicito dos documentos e do mapping do CNEFE."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


MAPPING_VERSION = 1

CNEFE_INDEX_SOURCE_COLUMNS = (
    "COD_UNICO_ENDERECO",
    "COD_UF",
    "COD_MUNICIPIO",
    "COD_SETOR",
    "municipio_codigo_normalizado",
    "cep_normalizado",
    "tipo_logradouro_normalizado",
    "titulo_logradouro_normalizado",
    "nome_logradouro_normalizado",
    "logradouro_normalizado",
    "numero_normalizado",
    "numero_comparavel_normalizado",
    "sem_numero",
    "localidade_normalizada",
    "modificador_normalizado",
    "complemento_normalizado",
    "endereco_normalizado",
)

SOURCE_TO_DOCUMENT_FIELDS = {
    "COD_UNICO_ENDERECO": "id_cnefe",
    "COD_UF": "codigo_uf",
    "COD_MUNICIPIO": "codigo_municipio_ibge",
    "COD_SETOR": "setor_censitario",
    "municipio_codigo_normalizado": "codigo_municipio",
    "cep_normalizado": "cep",
    "tipo_logradouro_normalizado": "tipo_logradouro",
    "titulo_logradouro_normalizado": "titulo_logradouro",
    "nome_logradouro_normalizado": "nome_logradouro",
    "logradouro_normalizado": "logradouro",
    "numero_normalizado": "numero",
    "numero_comparavel_normalizado": "numero_comparavel",
    "sem_numero": "sem_numero",
    "localidade_normalizada": "localidade",
    "modificador_normalizado": "modificador",
    "complemento_normalizado": "complemento",
    "endereco_normalizado": "endereco_normalizado",
}


def _searchable_text(*, ignore_above: int = 512) -> dict[str, object]:
    return {
        "type": "text",
        "analyzer": "standard",
        "fields": {
            "exact": {
                "type": "keyword",
                "ignore_above": ignore_above,
            }
        },
    }


CNEFE_INDEX_SETTINGS: dict[str, object] = {
    "number_of_shards": 1,
    "number_of_replicas": 0,
    "refresh_interval": "-1",
}

CNEFE_INDEX_MAPPINGS: dict[str, object] = {
    "dynamic": "strict",
    "_meta": {
        "mapping_version": MAPPING_VERSION,
        "description": "Enderecos normalizados do CNEFE Bahia",
    },
    "properties": {
        "id_cnefe": {"type": "keyword"},
        "codigo_uf": {"type": "keyword"},
        "codigo_municipio_ibge": {"type": "keyword"},
        "codigo_municipio": {"type": "keyword"},
        "setor_censitario": {"type": "keyword"},
        "cep": {"type": "keyword"},
        "tipo_logradouro": {"type": "keyword"},
        "titulo_logradouro": _searchable_text(ignore_above=256),
        "nome_logradouro": _searchable_text(),
        "logradouro": _searchable_text(),
        "numero": {"type": "keyword"},
        "numero_comparavel": {"type": "keyword"},
        "sem_numero": {"type": "boolean"},
        "localidade": _searchable_text(),
        "modificador": _searchable_text(),
        "complemento": _searchable_text(ignore_above=1024),
        "endereco_normalizado": _searchable_text(ignore_above=2048),
    },
}


class InvalidCnefeDocumentError(ValueError):
    """Documento sem os identificadores obrigatorios para indexacao."""


def build_document(record: Mapping[str, Any]) -> dict[str, Any]:
    """Converte uma linha normalizada no documento minimo do indice."""
    document = {
        target: record.get(source)
        for source, target in SOURCE_TO_DOCUMENT_FIELDS.items()
        if record.get(source) is not None
    }
    for required in ("id_cnefe", "setor_censitario", "codigo_municipio"):
        value = document.get(required)
        if value is None or str(value).strip() == "":
            raise InvalidCnefeDocumentError(
                f"Documento CNEFE sem campo obrigatorio: {required}."
            )
    return document


def build_bulk_action(
    record: Mapping[str, Any], *, index_name: str
) -> dict[str, Any]:
    document = build_document(record)
    return {
        "_op_type": "index",
        "_index": index_name,
        "_id": document["id_cnefe"],
        "_source": document,
    }
