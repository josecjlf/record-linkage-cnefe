"""Consultas tecnicas sem decisao automatica de candidato."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

from elasticsearch import Elasticsearch


class IndexValidationError(RuntimeError):
    """O indice nao respondeu como esperado as consultas tecnicas."""


def build_exact_filter_query(document: dict[str, Any]) -> dict[str, Any]:
    filters = [
        {"term": {"codigo_municipio": document["codigo_municipio"]}},
        {"term": {"logradouro.exact": document["logradouro"]}},
    ]
    for field in ("cep", "numero_comparavel"):
        if document.get(field) is not None:
            filters.append({"term": {field: document[field]}})
    return {"bool": {"filter": filters}}


def build_match_query(logradouro: str) -> dict[str, Any]:
    return {
        "match": {
            "logradouro": {
                "query": logradouro,
                "operator": "and",
            }
        }
    }


def build_multi_match_query(logradouro: str) -> dict[str, Any]:
    return {
        "multi_match": {
            "query": logradouro,
            "fields": [
                "logradouro",
                "nome_logradouro",
                "localidade",
                "endereco_normalizado",
            ],
            "type": "best_fields",
            "operator": "and",
        }
    }


def _total_hits(response: Any) -> int:
    total = response["hits"]["total"]
    return int(total["value"] if isinstance(total, dict) else total)


def _top_hits(response: Any) -> list[dict[str, Any]]:
    return [
        {
            "id_cnefe": hit["_source"]["id_cnefe"],
            "setor_censitario": hit["_source"]["setor_censitario"],
            "score": hit.get("_score"),
        }
        for hit in response["hits"]["hits"]
    ]


def validate_technical_queries(
    client: Elasticsearch, *, index_name: str
) -> dict[str, Any]:
    """Executa term/filter, match e multi_match sem selecionar o top 1."""
    indexed_documents = int(client.count(index=index_name)["count"])
    if indexed_documents == 0:
        raise IndexValidationError(f"Indice '{index_name}' esta vazio.")

    # A consulta exata tambem exige o multi-field keyword, nao so texto.
    sample_response = client.search(
        index=index_name,
        size=1,
        query={
            "bool": {
                "filter": [{"exists": {"field": "logradouro.exact"}}],
                "must_not": [{"term": {"logradouro.exact": ""}}],
            }
        },
    )
    if _total_hits(sample_response) == 0:
        raise IndexValidationError(
            f"Indice '{index_name}' possui documentos, mas nenhum tem "
            "logradouro nao vazio com campo exact indexado para validar "
            "as consultas tecnicas. Documentos sem logradouro sao permitidos."
        )

    sample_hit = sample_response["hits"]["hits"][0]
    sample = sample_hit["_source"]
    logradouro = sample.get("logradouro")
    if not isinstance(logradouro, str) or not logradouro.strip():
        raise IndexValidationError(
            "A amostra de validacao nao possui logradouro textual adequado."
        )
    queries = {
        "exact_term_filter": build_exact_filter_query(sample),
        "match_logradouro": build_match_query(logradouro),
        "multi_match_address_fields": build_multi_match_query(logradouro),
    }

    query_reports: dict[str, Any] = {}
    for name, query in queries.items():
        response = client.search(index=index_name, size=5, query=query)
        total = _total_hits(response)
        if total == 0:
            raise IndexValidationError(
                f"Consulta tecnica '{name}' nao recuperou documentos."
            )
        query_reports[name] = {
            "query": query,
            "total_hits": total,
            "top_hits_for_inspection": _top_hits(response),
        }

    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "index_name": index_name,
        "indexed_documents": indexed_documents,
        "sample_document_id": sample_hit["_id"],
        "queries": query_reports,
        "note": "Scores sao diagnosticos; nenhum top hit foi aceito como match.",
    }


def write_validation_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
