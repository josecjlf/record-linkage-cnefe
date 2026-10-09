"""Recuperacao de candidatos, sem decidir se algum deles e o match final."""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from operator import and_
from typing import Any, Mapping, Protocol

from pyspark.sql import DataFrame, functions as F

from src.matching.baseline_strategies import (
    ExactMatchStrategy,
    NumberRequirement,
)
from src.matching.candidate_selector import CandidateSummary


@dataclass(frozen=True)
class RetrievalResult:
    summaries: tuple[CandidateSummary, ...]
    eligible_input_count: int


@dataclass(frozen=True)
class StructuredAddress:
    """Campos normalizados de uma entrada usados nas consultas Elasticsearch."""

    input_id: str
    municipality_code: str | None
    postal_code: str | None
    street: str | None
    number: str | None
    neighborhood: str | None
    complement: str | None
    normalized_address: str | None

    @classmethod
    def from_mapping(cls, record: Mapping[str, Any]) -> "StructuredAddress":
        return cls(
            input_id=_optional_text(record.get("input_id")) or "",
            municipality_code=_optional_text(
                record.get("municipio_codigo_normalizado")
            ),
            postal_code=_optional_text(record.get("cep_normalizado")),
            street=_optional_text(record.get("logradouro_normalizado")),
            number=_optional_text(
                record.get("numero_comparavel_normalizado")
            ),
            neighborhood=_optional_text(record.get("bairro_normalizado")),
            complement=_optional_text(record.get("complemento_normalizado")),
            normalized_address=_optional_text(
                record.get("endereco_normalizado")
            ),
        )


@dataclass(frozen=True)
class RetrievalQuery:
    strategy: str
    order: int
    query: dict[str, Any]


@dataclass(frozen=True)
class RetrievedCandidate:
    rank: int
    id_cnefe: str
    setor_censitario: str
    codigo_municipio: str | None
    cep: str | None
    logradouro: str | None
    numero_comparavel: str | None
    localidade: str | None
    complemento: str | None
    endereco_normalizado: str | None
    score: float | None
    query_strategy: str
    strategy_order: int

    def as_record(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "id_cnefe": self.id_cnefe,
            "setor_censitario": self.setor_censitario,
            "codigo_municipio": self.codigo_municipio,
            "cep": self.cep,
            "logradouro": self.logradouro,
            "numero_comparavel": self.numero_comparavel,
            "localidade": self.localidade,
            "complemento": self.complemento,
            "endereco_normalizado": self.endereco_normalizado,
            "score": self.score,
            "query_strategy": self.query_strategy,
            "strategy_order": self.strategy_order,
        }


@dataclass(frozen=True)
class QueryAttempt:
    strategy: str
    order: int
    query: dict[str, Any]
    total_hits: int
    returned_hits: int
    max_score: float | None


@dataclass(frozen=True)
class CandidateRetrieval:
    input_id: str
    candidates: tuple[RetrievedCandidate, ...]
    attempts: tuple[QueryAttempt, ...]


class SearchClient(Protocol):
    def search(self, **kwargs: Any) -> Mapping[str, Any]: ...


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _geographic_filters(
    address: StructuredAddress, *, include_postal_code: bool
) -> list[dict[str, Any]]:
    filters: list[dict[str, Any]] = []
    if address.municipality_code:
        filters.append(
            {"term": {"codigo_municipio": address.municipality_code}}
        )
    if include_postal_code and address.postal_code:
        filters.append({"term": {"cep": address.postal_code}})
    return filters


def _ranking_should_clauses(
    address: StructuredAddress,
    *,
    include_postal_code: bool,
    include_number: bool = True,
) -> list[dict[str, Any]]:
    clauses: list[dict[str, Any]] = []
    if include_postal_code and address.postal_code:
        clauses.append({"term": {"cep": address.postal_code}})
    if include_number and address.number:
        clauses.append({"term": {"numero_comparavel": address.number}})
    if address.neighborhood:
        clauses.append(
            {
                "match": {
                    "localidade": {
                        "query": address.neighborhood,
                        "operator": "and",
                    }
                }
            }
        )
    if address.complement:
        clauses.append(
            {
                "match": {
                    "complemento": {
                        "query": address.complement,
                        "operator": "and",
                    }
                }
            }
        )
    return clauses


def build_retrieval_queries(
    address: StructuredAddress,
) -> tuple[RetrievalQuery, ...]:
    """Monta ate tres queries; sem municipio, a textual pode ficar sem recorte.

    A query exata exige municipio ou CEP. A textual filtra municipio quando
    informado; nessa etapa, CEP influencia apenas o ranking. O fallback fuzzy
    exige municipio ou, na sua ausencia, CEP como filtro geografico.
    """
    if not address.input_id:
        raise ValueError("input_id e obrigatorio para recuperar candidatos.")
    if not address.street:
        return ()

    exact_filters = _geographic_filters(
        address, include_postal_code=True
    )
    if not exact_filters:
        return ()
    exact_filters.append(
        {"term": {"logradouro.exact": address.street}}
    )
    if address.number:
        exact_filters.append(
            {"term": {"numero_comparavel": address.number}}
        )
    queries = [
        RetrievalQuery(
            strategy="EXACT_STRUCTURED",
            order=1,
            query={"bool": {"filter": exact_filters}},
        )
    ]

    geographic_filters = _geographic_filters(
        address, include_postal_code=False
    )
    number_is_informative = bool(
        address.number and address.number != "sn"
    )
    if number_is_informative:
        geographic_filters.append(
            {"term": {"numero_comparavel": address.number}}
        )
    text_bool: dict[str, Any] = {
        "filter": geographic_filters,
        "must": [
            {
                "match": {
                    "logradouro": {
                        "query": address.street,
                        "operator": "and",
                    }
                }
            }
        ],
    }
    should = _ranking_should_clauses(
        address,
        include_postal_code=True,
        include_number=not number_is_informative,
    )
    if should:
        text_bool["should"] = should
    queries.append(
        RetrievalQuery(
            strategy="TEXT_WITH_GEOGRAPHY",
            order=2,
            query={"bool": text_bool},
        )
    )

    relaxed_filters = _geographic_filters(
        address, include_postal_code=False
    )
    if not relaxed_filters and address.postal_code:
        relaxed_filters = [{"term": {"cep": address.postal_code}}]
    if not relaxed_filters:
        return tuple(queries)
    fuzzy_bool: dict[str, Any] = {
        "filter": relaxed_filters,
        "must": [
            {
                "multi_match": {
                    "query": address.street,
                    "fields": [
                        "logradouro^3",
                        "nome_logradouro^2",
                        "endereco_normalizado",
                    ],
                    "type": "best_fields",
                    "operator": "and",
                    "fuzziness": "AUTO",
                    "prefix_length": 1,
                    "max_expansions": 25,
                }
            }
        ],
    }
    fuzzy_should = _ranking_should_clauses(
        address, include_postal_code=True
    )
    if fuzzy_should:
        fuzzy_bool["should"] = fuzzy_should
    queries.append(
        RetrievalQuery(
            strategy="FUZZY_STREET_WITH_MUNICIPALITY",
            order=3,
            query={"bool": fuzzy_bool},
        )
    )
    return tuple(queries)


def _total_hits(response: Mapping[str, Any]) -> int:
    total = response.get("hits", {}).get("total", 0)
    if isinstance(total, Mapping):
        return int(total.get("value", 0))
    return int(total)


def _candidate_from_hit(
    hit: Mapping[str, Any], *, strategy: RetrievalQuery, rank: int
) -> RetrievedCandidate:
    source = hit.get("_source", {})
    candidate_id = _optional_text(source.get("id_cnefe")) or _optional_text(
        hit.get("_id")
    )
    sector = _optional_text(source.get("setor_censitario"))
    if not candidate_id or not sector:
        raise ValueError(
            "Documento Elasticsearch sem id_cnefe ou setor_censitario."
        )
    raw_score = hit.get("_score")
    return RetrievedCandidate(
        rank=rank,
        id_cnefe=candidate_id,
        setor_censitario=sector,
        codigo_municipio=_optional_text(source.get("codigo_municipio")),
        cep=_optional_text(source.get("cep")),
        logradouro=_optional_text(source.get("logradouro")),
        numero_comparavel=_optional_text(source.get("numero_comparavel")),
        localidade=_optional_text(source.get("localidade")),
        complemento=_optional_text(source.get("complemento")),
        endereco_normalizado=_optional_text(
            source.get("endereco_normalizado")
        ),
        score=float(raw_score) if raw_score is not None else None,
        query_strategy=strategy.strategy,
        strategy_order=strategy.order,
    )


def retrieve_candidates(
    client: SearchClient,
    *,
    index_name: str,
    address: StructuredAddress,
    top_k: int = 10,
    minimum_candidates: int = 5,
) -> CandidateRetrieval:
    """Executa fallbacks sob demanda e devolve candidatos, sem selecao final."""
    if top_k <= 0:
        raise ValueError("top_k deve ser maior que zero.")
    if not 1 <= minimum_candidates <= top_k:
        raise ValueError(
            "minimum_candidates deve estar entre 1 e top_k."
        )

    candidates: list[RetrievedCandidate] = []
    seen_ids: set[str] = set()
    attempts: list[QueryAttempt] = []
    for strategy in build_retrieval_queries(address):
        response = client.search(
            index=index_name,
            size=top_k,
            query=strategy.query,
            sort=["_score:desc", "id_cnefe:asc"],
            track_scores=True,
            track_total_hits=True,
        )
        hits = list(response.get("hits", {}).get("hits", []))
        max_score = response.get("hits", {}).get("max_score")
        attempts.append(
            QueryAttempt(
                strategy=strategy.strategy,
                order=strategy.order,
                query=strategy.query,
                total_hits=_total_hits(response),
                returned_hits=len(hits),
                max_score=(
                    float(max_score) if max_score is not None else None
                ),
            )
        )
        for hit in hits:
            candidate = _candidate_from_hit(
                hit, strategy=strategy, rank=len(candidates) + 1
            )
            if candidate.id_cnefe in seen_ids:
                continue
            seen_ids.add(candidate.id_cnefe)
            candidates.append(candidate)
            if len(candidates) == top_k:
                break
        if len(candidates) >= minimum_candidates:
            break

    return CandidateRetrieval(
        input_id=address.input_id,
        candidates=tuple(candidates),
        attempts=tuple(attempts),
    )


def _is_present(column_name: str):
    column = F.col(column_name)
    return column.isNotNull() & (F.trim(column) != "")


def eligible_search_inputs(
    search: DataFrame,
    strategy: ExactMatchStrategy,
    *,
    input_ids: set[str] | None = None,
) -> DataFrame:
    """Filtra somente entradas que possuem os componentes exigidos pela regra."""
    eligible = search
    if input_ids is not None:
        if not input_ids:
            return search.limit(0)
        eligible = eligible.where(F.col("input_id").isin(sorted(input_ids)))

    required_search_columns = tuple(
        dict.fromkeys(search_column for search_column, _ in strategy.key_pairs)
    )
    for column_name in required_search_columns:
        eligible = eligible.where(_is_present(column_name))

    number = F.col("numero_comparavel_normalizado")
    number_available = _is_present("numero_comparavel_normalizado") & (
        number != "sn"
    )
    if strategy.number_requirement == NumberRequirement.PRESENT:
        eligible = eligible.where(number_available)
    elif strategy.number_requirement == NumberRequirement.MISSING_OR_SN:
        eligible = eligible.where(~number_available)
    return eligible


def retrieve_candidate_summaries(
    cnefe: DataFrame,
    search: DataFrame,
    strategy: ExactMatchStrategy,
    *,
    input_ids: set[str] | None = None,
) -> RetrievalResult:
    """Retorna apenas agregados pequenos; nenhum candidato individual e coletado."""
    eligible = eligible_search_inputs(
        search, strategy, input_ids=input_ids
    ).select("input_id", *(pair[0] for pair in strategy.key_pairs))
    eligible = eligible.dropDuplicates(["input_id"])
    eligible_input_count = eligible.count()
    if eligible_input_count == 0:
        return RetrievalResult(summaries=(), eligible_input_count=0)

    search_alias = F.broadcast(eligible).alias("search")
    cnefe_alias = cnefe.alias("cnefe")
    conditions = [
        F.col(f"search.{search_column}")
        == F.col(f"cnefe.{cnefe_column}")
        for search_column, cnefe_column in strategy.key_pairs
    ]
    join_condition = reduce(and_, conditions)
    matches = cnefe_alias.join(search_alias, join_condition, "inner").select(
        F.col("search.input_id").alias("input_id"),
        F.col("cnefe.COD_UNICO_ENDERECO").alias("candidate_id"),
        F.col("cnefe.COD_SETOR").alias("candidate_sector"),
    )
    rows = (
        matches.groupBy("input_id")
        .agg(
            F.count(F.lit(1)).alias("candidate_row_count"),
            F.countDistinct("candidate_id").alias("candidate_count"),
            F.sort_array(F.collect_set("candidate_sector")).alias(
                "candidate_sectors"
            ),
            F.min("candidate_id").alias("minimum_candidate_id"),
        )
        .collect()
    )
    summaries = tuple(
        CandidateSummary(
            input_id=row["input_id"],
            strategy_name=strategy.name,
            candidate_count=int(row["candidate_count"]),
            candidate_row_count=int(row["candidate_row_count"]),
            candidate_sectors=tuple(row["candidate_sectors"]),
            minimum_candidate_id=row["minimum_candidate_id"],
        )
        for row in rows
    )
    return RetrievalResult(
        summaries=summaries,
        eligible_input_count=eligible_input_count,
    )
