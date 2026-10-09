"""Avaliacao reproduzivel dos resultados, sem alterar decisoes de matching."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import logging
from pathlib import Path
import re
import unicodedata
from typing import Any, Mapping, Protocol

import pandas as pd

from src.ingestion.schemas import SEARCH_COLUMNS
from src.ingestion.validation import (
    DatasetValidationError,
    validate_required_columns,
)
from src.matching.candidate_selector import (
    MatchStatus,
    SelectionAddress,
    SelectionCandidate,
    SelectionPolicy,
    select_retrieved_candidates,
)
from src.preprocessing.address_normalizer import split_street_type
from src.preprocessing.pandas_address_normalizer import (
    SEARCH_NORMALIZED_COLUMNS,
    normalize_search_addresses,
)


LOGGER = logging.getLogger(__name__)

AUDIT_REQUIRED_COLUMNS = (
    "input_id",
    "consulta_logradouro",
    "consulta_numero",
    "consulta_bairro",
    "consulta_cep",
    "consulta_complemento",
    "consulta_municipio",
    "logradouro_normalizado",
    "numero_ausente",
    "sem_numero",
    "bairro_normalizado",
    "cep_normalizado",
    "status_match",
    "metodo_match",
    "setor_censitario",
    "baseline_status_match",
    "baseline_metodo_match",
    "diagnostico_baseline",
    "candidate_count",
    "candidate_sector_count",
    "strong_candidate_count",
    "plausible_candidate_count",
)

EVIDENCE_REQUIRED_COLUMNS = (
    "input_id",
    "sector",
    "retrieval_strategy",
    "street_similarity",
    "postal_code_match",
    "number_match",
    "neighborhood_match",
    "evidence_strength",
)

_PLACEHOLDER_TOKEN = re.compile(r"(?:^|\s)(?:none|null|nan)(?:\s|$)", re.I)
_DIGIT = re.compile(r"\d")


class EvaluationSearchClient(Protocol):
    def search(self, **kwargs: Any) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class EvaluationOutput:
    cases_path: Path
    low_confidence_path: Path
    metrics_path: Path
    report_path: Path
    total_records: int


def _read_csv(
    path: Path,
    required: tuple[str, ...],
    label: str,
    *,
    allow_empty: bool = False,
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Artefato de {label} nao encontrado: {path}")
    try:
        dataframe = pd.read_csv(
            path,
            sep=";",
            encoding="utf-8",
            dtype="string",
            keep_default_na=False,
        )
    except pd.errors.EmptyDataError as error:
        raise DatasetValidationError(
            f"Artefato de {label} sem cabecalho: {path}. "
            "Execute run-pipeline novamente para gerar os artefatos."
        ) from error
    validate_required_columns(dataframe.columns, required, label)
    if dataframe.empty and not allow_empty:
        raise DatasetValidationError(f"Artefato de {label} esta vazio.")
    return dataframe


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Artefato de {label} nao encontrado: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise DatasetValidationError(f"Artefato de {label} esta vazio.")
    return payload


def _read_diagnostics(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Diagnostico de candidate retrieval nao encontrado: {path}"
        )
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    indexed = {str(record["input_id"]): record for record in records}
    if not indexed or len(indexed) != len(records):
        raise DatasetValidationError(
            "Diagnostico de retrieval vazio ou com input_id duplicado."
        )
    return indexed


def _text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _truth(value: object) -> bool:
    return _text(value).lower() == "true"


def _integer(value: object) -> int:
    text = _text(value)
    return int(float(text)) if text else 0


def _maximum_float(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.max()) if not numeric.empty else None


def _any_true(frame: pd.DataFrame, column: str) -> bool:
    return bool(frame[column].map(_truth).any()) if not frame.empty else False


def _quality_flags(row: Mapping[str, object]) -> tuple[str, ...]:
    flags: list[str] = []
    if _truth(row["numero_ausente"]):
        flags.append("NUMBER_MISSING")
    elif _truth(row["sem_numero"]):
        flags.append("WITHOUT_NUMBER_MARKER")
    if not _text(row["bairro_normalizado"]):
        flags.append("NEIGHBORHOOD_MISSING")
    if not _text(row["cep_normalizado"]):
        flags.append("POSTAL_CODE_MISSING")
    raw_street = _text(row["consulta_logradouro"])
    if _DIGIT.search(raw_street):
        flags.append("DIGIT_IN_STREET")
    if _PLACEHOLDER_TOKEN.search(raw_street):
        flags.append("PLACEHOLDER_TOKEN_IN_STREET")
    if _text(row["consulta_complemento"]):
        flags.append("COMPLEMENT_PRESENT")
    return tuple(flags)


def _evidence_summary(group: pd.DataFrame) -> dict[str, object]:
    strong = group[group["evidence_strength"] == "STRONG"]
    plausible = group[group["evidence_strength"] == "PLAUSIBLE"]
    return {
        "max_street_similarity": _maximum_float(group["street_similarity"]),
        "max_strong_street_similarity": _maximum_float(
            strong["street_similarity"]
        ),
        "strong_sector_count": int(strong["sector"].nunique()),
        "plausible_sector_count": int(plausible["sector"].nunique()),
        "any_postal_code_match": _any_true(group, "postal_code_match"),
        "any_number_match": _any_true(group, "number_match"),
        "any_neighborhood_match": _any_true(group, "neighborhood_match"),
        "retrieval_strategies": "|".join(
            sorted(set(group["retrieval_strategy"].dropna().astype(str)))
        ),
    }


def _classify_outcome(
    row: Mapping[str, object],
    summary: Mapping[str, object],
    *,
    policy: SelectionPolicy,
    near_threshold_band: float,
) -> tuple[str, str, tuple[str, ...]]:
    status = _text(row["status_match"])
    method = _text(row["metodo_match"])
    review_flags: list[str] = []

    if status == MatchStatus.FOUND.value and method == "BASELINE_FOUND_PRESERVED":
        diagnostic = _text(row["diagnostico_baseline"])
        category = (
            "BASELINE_EXACT_UNIQUE_CANDIDATE"
            if diagnostic == "UNIQUE_CANDIDATE"
            else "BASELINE_EXACT_MULTIPLE_DOCUMENTS_SAME_SECTOR"
        )
        return category, "HIGH_EXACT_SECTOR", ()

    if status == MatchStatus.FOUND.value:
        similarity = summary["max_strong_street_similarity"]
        if similarity is None:
            raise DatasetValidationError(
                f"FOUND Elasticsearch sem evidencia STRONG: {row['input_id']}."
            )
        if float(similarity) - policy.strong_street_similarity <= (
            near_threshold_band + 1e-12
        ):
            confidence = "REVIEW_NEAR_STRONG_THRESHOLD"
            category = "ELASTICSEARCH_STRONG_NEAR_THRESHOLD"
            review_flags.append("STREET_SIMILARITY_NEAR_STRONG_THRESHOLD")
        else:
            confidence = "HIGH_STRUCTURED"
            category = "ELASTICSEARCH_STRONG_SINGLE_SECTOR"
        if not bool(summary["any_postal_code_match"]):
            review_flags.append("NO_POSTAL_CODE_AGREEMENT")
        if _text(row["bairro_normalizado"]) and not bool(
            summary["any_neighborhood_match"]
        ):
            review_flags.append("NO_NEIGHBORHOOD_AGREEMENT")
        return category, confidence, tuple(review_flags)

    if status == MatchStatus.AMBIGUOUS.value:
        if method == "BASELINE_AMBIGUOUS_PRESERVED":
            return (
                "EXACT_ADDRESS_MULTIPLE_SECTORS",
                "UNRESOLVED_AMBIGUOUS",
                ("EXACT_CONFLICT_IN_CNEFE",),
            )
        if method == "ES_STRONG_EVIDENCE_MULTIPLE_SECTORS":
            return (
                "STRONG_CANDIDATES_MULTIPLE_SECTORS",
                "UNRESOLVED_AMBIGUOUS",
                ("STRONG_SECTOR_CONFLICT",),
            )
        if _integer(row["candidate_sector_count"]) > 1:
            category = "PLAUSIBLE_CANDIDATES_MULTIPLE_SECTORS"
        else:
            category = "PLAUSIBLE_SINGLE_SECTOR_INSUFFICIENT_ANCHORS"
        return category, "UNRESOLVED_AMBIGUOUS", ("PLAUSIBLE_ONLY",)

    if method == "ES_NO_CANDIDATES":
        return (
            "NO_CANDIDATE_AFTER_PROGRESSIVE_RETRIEVAL",
            "UNRESOLVED_NO_EVIDENCE",
            ("NO_ELASTICSEARCH_CANDIDATE",),
        )
    if (
        summary["max_street_similarity"] is not None
        and float(summary["max_street_similarity"])
        >= policy.plausible_street_similarity
    ):
        category = "TEXT_SIMILAR_BUT_WITHOUT_EXACT_ANCHOR"
    else:
        category = "CANDIDATES_BELOW_PLAUSIBLE_THRESHOLD"
    return category, "UNRESOLVED_WEAK_EVIDENCE", ("WEAK_EVIDENCE_ONLY",)


def build_case_evaluation(
    audit: pd.DataFrame,
    evidence: pd.DataFrame,
    *,
    policy: SelectionPolicy,
    near_threshold_band: float = 0.05,
) -> pd.DataFrame:
    """Classifica os resultados por regras gerais e auditaveis."""
    validate_required_columns(audit.columns, AUDIT_REQUIRED_COLUMNS, "auditoria")
    validate_required_columns(
        evidence.columns, EVIDENCE_REQUIRED_COLUMNS, "evidencias"
    )
    if audit["input_id"].duplicated().any():
        raise DatasetValidationError("Auditoria possui input_id duplicado.")
    if near_threshold_band < 0:
        raise ValueError("near_threshold_band nao pode ser negativo.")

    evidence_by_id = {
        str(input_id): group
        for input_id, group in evidence.groupby("input_id", sort=False)
    }
    records: list[dict[str, object]] = []
    empty_evidence = evidence.iloc[0:0]
    for row in audit.to_dict(orient="records"):
        input_id = _text(row["input_id"])
        group = evidence_by_id.get(input_id, empty_evidence)
        summary = _evidence_summary(group)
        category, confidence, review_flags = _classify_outcome(
            row,
            summary,
            policy=policy,
            near_threshold_band=near_threshold_band,
        )
        quality_flags = _quality_flags(row)
        records.append(
            {
                "input_id": input_id,
                "consulta_logradouro": _text(row["consulta_logradouro"]),
                "consulta_numero": _text(row["consulta_numero"]),
                "consulta_bairro": _text(row["consulta_bairro"]),
                "consulta_cep": _text(row["consulta_cep"]),
                "consulta_complemento": _text(
                    row["consulta_complemento"]
                ),
                "consulta_municipio": _text(row["consulta_municipio"]),
                "setor_censitario": _text(row["setor_censitario"]),
                "status_match": _text(row["status_match"]),
                "metodo_match": _text(row["metodo_match"]),
                "baseline_status_match": _text(
                    row["baseline_status_match"]
                ),
                "baseline_metodo_match": _text(
                    row["baseline_metodo_match"]
                ),
                "match_origin": (
                    "BASELINE_EXACT"
                    if _text(row["metodo_match"]).startswith("BASELINE_")
                    else "ELASTICSEARCH"
                ),
                "confidence_band": confidence,
                "outcome_category": category,
                "review_flags": "|".join(review_flags),
                "quality_flags": "|".join(quality_flags),
                "candidate_count": _integer(row["candidate_count"]),
                "candidate_sector_count": _integer(
                    row["candidate_sector_count"]
                ),
                "strong_candidate_count": _integer(
                    row["strong_candidate_count"]
                ),
                "plausible_candidate_count": _integer(
                    row["plausible_candidate_count"]
                ),
                **summary,
            }
        )
    return pd.DataFrame(records)


def _strip_accents(value: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFD", value)
        if unicodedata.category(character) != "Mn"
    )


def analyze_normalization_stability(search: pd.DataFrame) -> dict[str, object]:
    """Testa variacoes irrelevantes de caixa, acento, espaco e pontuacao."""
    validate_required_columns(search.columns, SEARCH_COLUMNS, "base de busca")
    original = search.loc[:, list(SEARCH_COLUMNS)].copy(deep=True)
    original_normalized = normalize_search_addresses(original)
    variant = original.copy(deep=True)

    for column in (
        "consulta_logradouro",
        "consulta_bairro",
        "consulta_complemento",
    ):
        variant[column] = variant[column].map(
            lambda value: value
            if pd.isna(value) or not str(value).strip()
            else f"  {_strip_accents(str(value)).swapcase()} !!!  "
        )
    variant["consulta_numero"] = variant["consulta_numero"].map(
        lambda value: value
        if pd.isna(value) or not str(value).strip()
        else f"  {str(value).swapcase()}  "
    )
    variant["consulta_cep"] = variant["consulta_cep"].map(
        lambda value: value
        if pd.isna(value) or len(re.sub(r"\D", "", str(value))) != 8
        else (
            re.sub(r"\D", "", str(value))[:5]
            + "-"
            + re.sub(r"\D", "", str(value))[5:]
        )
    )
    variant_normalized = normalize_search_addresses(variant)
    equality = original_normalized.loc[:, SEARCH_NORMALIZED_COLUMNS].fillna("").eq(
        variant_normalized.loc[:, SEARCH_NORMALIZED_COLUMNS].fillna("")
    )
    stable_rows = equality.all(axis=1)
    return {
        "tested_records": len(search),
        "stable_records": int(stable_rows.sum()),
        "changed_records": int((~stable_rows).sum()),
        "stability_percentage": round(float(stable_rows.mean() * 100), 2),
        "tested_variations": [
            "case",
            "accent_removal",
            "outer_whitespace",
            "trailing_punctuation",
            "postal_code_punctuation",
        ],
        "changed_input_ids": [
            _text(value)
            for value in search.loc[~stable_rows, "search_id"].tolist()
        ]
        if "search_id" in search.columns
        else [],
    }


def build_relaxed_retrieval_query(
    address: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Query experimental: relaxa termos longos, preservando o municipio."""
    municipality = _text(address.get("municipality_code"))
    street = _text(address.get("street"))
    if not municipality or not street:
        return None
    _, street_name = split_street_type(street)
    query_text = street_name or street
    minimum_should_match = "100%" if len(query_text.split()) <= 2 else "75%"
    should: list[dict[str, object]] = []
    for field, address_field in (
        ("cep", "postal_code"),
        ("numero_comparavel", "number"),
    ):
        value = _text(address.get(address_field))
        if value:
            should.append({"term": {field: value}})
    neighborhood = _text(address.get("neighborhood"))
    if neighborhood:
        should.append(
            {
                "match": {
                    "localidade": {
                        "query": neighborhood,
                        "operator": "and",
                    }
                }
            }
        )
    body: dict[str, Any] = {
        "filter": [{"term": {"codigo_municipio": municipality}}],
        "must": [
            {
                "multi_match": {
                    "query": query_text,
                    "fields": [
                        "nome_logradouro^3",
                        "logradouro^2",
                        "endereco_normalizado",
                    ],
                    "type": "best_fields",
                    "operator": "or",
                    "minimum_should_match": minimum_should_match,
                    "fuzziness": "AUTO",
                    "prefix_length": 1,
                    "max_expansions": 25,
                }
            }
        ],
    }
    if should:
        body["should"] = should
    return {"bool": body}


def _selection_address(record: Mapping[str, Any]) -> SelectionAddress:
    return SelectionAddress(
        input_id=_text(record.get("input_id")),
        municipality_code=_text(record.get("municipality_code")) or None,
        postal_code=_text(record.get("postal_code")) or None,
        street=_text(record.get("street")) or None,
        number=_text(record.get("number")) or None,
        neighborhood=_text(record.get("neighborhood")) or None,
        complement=_text(record.get("complement")) or None,
    )


def _selection_candidate(
    hit: Mapping[str, Any], *, rank: int
) -> SelectionCandidate | None:
    source = hit.get("_source", {})
    candidate_id = _text(source.get("id_cnefe")) or _text(hit.get("_id"))
    sector = _text(source.get("setor_censitario"))
    if not candidate_id or not sector:
        return None
    raw_score = hit.get("_score")
    return SelectionCandidate(
        candidate_id=candidate_id,
        sector=sector,
        municipality_code=_text(source.get("codigo_municipio")) or None,
        postal_code=_text(source.get("cep")) or None,
        street=_text(source.get("logradouro")) or None,
        number=_text(source.get("numero_comparavel")) or None,
        neighborhood=_text(source.get("localidade")) or None,
        complement=_text(source.get("complemento")) or None,
        es_score=float(raw_score) if raw_score is not None else None,
        retrieval_strategy="EVALUATION_RELAXED_TEXT",
        retrieval_rank=rank,
    )


def run_relaxed_retrieval_experiment(
    client: EvaluationSearchClient,
    *,
    index_name: str,
    cases: pd.DataFrame,
    diagnostics: Mapping[str, Mapping[str, Any]],
    policy: SelectionPolicy,
    top_k: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Testa query menos restritiva sem modificar os resultados oficiais."""
    targets = cases[
        cases["outcome_category"]
        == "NO_CANDIDATE_AFTER_PROGRESSIVE_RETRIEVAL"
    ]
    rows: list[dict[str, object]] = []
    for input_id in targets["input_id"]:
        diagnostic = diagnostics[str(input_id)]
        address_record = diagnostic["address"]
        query = build_relaxed_retrieval_query(address_record)
        if query is None:
            rows.append(
                {
                    "input_id": input_id,
                    "relaxed_total_hits": 0,
                    "relaxed_candidate_count": 0,
                    "relaxed_status": "NOT_FOUND",
                    "relaxed_sector": "",
                    "relaxed_method": "QUERY_NOT_APPLICABLE",
                }
            )
            continue
        response = client.search(
            index=index_name,
            size=top_k,
            query=query,
            sort=["_score:desc", "id_cnefe:asc"],
            track_scores=True,
            track_total_hits=True,
        )
        raw_hits = list(response.get("hits", {}).get("hits", []))
        candidates = tuple(
            candidate
            for rank, hit in enumerate(raw_hits, start=1)
            if (candidate := _selection_candidate(hit, rank=rank)) is not None
        )
        decision = select_retrieved_candidates(
            _selection_address(address_record), candidates, policy=policy
        )
        raw_total = response.get("hits", {}).get("total", 0)
        total_hits = (
            int(raw_total.get("value", 0))
            if isinstance(raw_total, Mapping)
            else int(raw_total)
        )
        rows.append(
            {
                "input_id": input_id,
                "relaxed_total_hits": total_hits,
                "relaxed_candidate_count": len(candidates),
                "relaxed_status": decision.status_match.value,
                "relaxed_sector": decision.setor_censitario or "",
                "relaxed_method": decision.metodo_match,
            }
        )
    frame = pd.DataFrame(
        rows,
        columns=(
            "input_id",
            "relaxed_total_hits",
            "relaxed_candidate_count",
            "relaxed_status",
            "relaxed_sector",
            "relaxed_method",
        ),
    )
    frame["input_id"] = frame["input_id"].astype("string")
    status_counts = Counter(frame["relaxed_status"]) if not frame.empty else Counter()
    metrics = {
        "executed": True,
        "changes_official_results": False,
        "target_no_candidate_cases": len(targets),
        "cases_with_hits": int((frame["relaxed_candidate_count"] > 0).sum()),
        "candidate_count_total": int(frame["relaxed_candidate_count"].sum()),
        "proposed_status": {
            status.value: status_counts.get(status.value, 0)
            for status in MatchStatus
        },
        "query_design": {
            "geographic_filter": "exact municipality code",
            "text": "street name without known type",
            "short_names_minimum_should_match": "100%",
            "long_names_minimum_should_match": "75%",
            "fuzziness": "AUTO",
            "scope": "evaluation only",
        },
    }
    return frame, metrics


def _counts(values: pd.Series) -> dict[str, int]:
    counter = Counter(str(value) for value in values)
    return dict(sorted(counter.items()))


def _status_counts(cases: pd.DataFrame) -> dict[str, int]:
    counts = Counter(cases["status_match"])
    return {
        status.value: counts.get(status.value, 0) for status in MatchStatus
    }


def build_evaluation_metrics(
    cases: pd.DataFrame,
    *,
    selection_metrics: Mapping[str, Any],
    retrieval_metrics: Mapping[str, Any],
    normalization_report: Mapping[str, Any],
    diagnostics: Mapping[str, Mapping[str, Any]],
    evidence: pd.DataFrame,
    stability: Mapping[str, Any],
    relaxed_experiment: Mapping[str, Any],
    near_threshold_band: float = 0.05,
    top_k: int = 10,
) -> dict[str, object]:
    """Consolida métricas de avaliação sem recalcular decisões de match."""
    current = _status_counts(cases)
    quality_by_status: dict[str, dict[str, int]] = {}
    for status, group in cases.groupby("status_match"):
        flags = Counter(
            flag
            for packed in group["quality_flags"]
            for flag in str(packed).split("|")
            if flag
        )
        quality_by_status[str(status)] = dict(sorted(flags.items()))

    fuzzy = evidence[
        evidence["retrieval_strategy"] == "FUZZY_STREET_WITH_MUNICIPALITY"
    ]
    fuzzy_strength = _counts(fuzzy["evidence_strength"])
    zero_candidate_ids = set(
        cases.loc[
            cases["outcome_category"]
            == "NO_CANDIDATE_AFTER_PROGRESSIVE_RETRIEVAL",
            "input_id",
        ]
    )
    zero_with_municipality_only_fallback = 0
    for input_id in zero_candidate_ids:
        attempts = diagnostics[str(input_id)].get("attempts", [])
        if not attempts:
            continue
        final_query = attempts[-1].get("query", {}).get("bool", {})
        filters = final_query.get("filter", [])
        if (
            attempts[-1].get("strategy")
            == "FUZZY_STREET_WITH_MUNICIPALITY"
            and filters
            and all("codigo_municipio" in item.get("term", {}) for item in filters)
        ):
            zero_with_municipality_only_fallback += 1

    cnefe_normalization = normalization_report["cnefe"]
    collapsed = int(cnefe_normalization["collapsed_distinct_addresses"])
    original_distinct = int(cnefe_normalization["original_address_distinct"])
    near_found = int(
        (
            cases["confidence_band"]
            == "REVIEW_NEAR_STRONG_THRESHOLD"
        ).sum()
    )
    low_confidence = cases[
        cases["status_match"].eq("AMBIGUOUS")
        | cases["confidence_band"].eq("REVIEW_NEAR_STRONG_THRESHOLD")
    ]
    found_review_flags = Counter(
        flag
        for packed in cases.loc[
            cases["status_match"].eq("FOUND"), "review_flags"
        ]
        for flag in str(packed).split("|")
        if flag
    )
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "total": len(cases),
        "comparison": [
            {"version": "BASELINE", **selection_metrics["baseline"]},
            {"version": "FINAL", **current},
        ],
        "comparison_scope": "BASELINE_VS_FINAL_SAME_RUN",
        "changes_official_results": False,
        "found_percentage_is_accuracy": False,
        "ground_truth_available": False,
        "near_threshold_band": near_threshold_band,
        "retrieval_top_k": retrieval_metrics.get("top_k", top_k),
        "current_by_method": _counts(cases["metodo_match"]),
        "methods_by_version": {
            "BASELINE": _counts(cases["baseline_metodo_match"]),
            "FINAL": _counts(cases["metodo_match"]),
        },
        "found": {
            "total": current["FOUND"],
            "baseline_exact": int(
                (
                    cases["status_match"].eq("FOUND")
                    & cases["match_origin"].eq("BASELINE_EXACT")
                ).sum()
            ),
            "elasticsearch_structured": int(
                (
                    cases["status_match"].eq("FOUND")
                    & cases["match_origin"].eq("ELASTICSEARCH")
                ).sum()
            ),
            "near_strong_threshold": near_found,
            "with_review_flags": int(
                (
                    cases["status_match"].eq("FOUND")
                    & cases["review_flags"].ne("")
                ).sum()
            ),
            "review_flag_counts": dict(sorted(found_review_flags.items())),
        },
        "ambiguous_by_category": _counts(
            cases.loc[cases["status_match"] == "AMBIGUOUS", "outcome_category"]
        ),
        "not_found_by_category": _counts(
            cases.loc[cases["status_match"] == "NOT_FOUND", "outcome_category"]
        ),
        "low_confidence_cases": len(low_confidence),
        "quality_patterns_by_status": quality_by_status,
        "normalization_risk": {
            "search_collision_groups": normalization_report["search"][
                "normalization_collision_groups"
            ],
            "cnefe_collapsed_distinct_addresses": collapsed,
            "cnefe_original_distinct_addresses": original_distinct,
            "cnefe_collapsed_percentage": round(
                collapsed / original_distinct * 100, 4
            ),
            "cnefe_normalized_groups_with_multiple_sectors": (
                cnefe_normalization["normalized_groups_with_multiple_sectors"]
            ),
        },
        "retrieval_review": {
            "no_candidate_cases": len(zero_candidate_ids),
            "no_candidate_cases_after_municipality_only_fallback": (
                zero_with_municipality_only_fallback
            ),
            "fuzzy_retained_candidates": len(fuzzy),
            "fuzzy_evidence_strength": fuzzy_strength,
            "fuzzy_only_addresses": len(
                retrieval_metrics["address_ids_with_only_fuzzy_candidates"]
            ),
            "fuzzy_weak_percentage": round(
                fuzzy_strength.get("WEAK", 0) / len(fuzzy) * 100, 2
            )
            if len(fuzzy)
            else 0.0,
        },
        "threshold_sensitivity": selection_metrics["sensitivity"],
        "score_margin_analysis": selection_metrics["score_margin_analysis"],
        "stability": dict(stability),
        "relaxed_retrieval_experiment": dict(relaxed_experiment),
    }


def write_evaluation_report(path: Path, metrics: Mapping[str, Any]) -> None:
    """Grava o resumo auditável da avaliação em Markdown."""
    comparison = metrics["comparison"]
    found = metrics["found"]
    normalization = metrics["normalization_risk"]
    retrieval = metrics["retrieval_review"]
    relaxed = metrics["relaxed_retrieval_experiment"]
    missing_postal_codes = sum(
        patterns.get("POSTAL_CODE_MISSING", 0)
        for patterns in metrics["quality_patterns_by_status"].values()
    )
    margin_median = metrics["score_margin_analysis"][
        "absolute_gap_distribution"
    ]["median"]
    margin_description = (
        str(margin_median)
        if margin_median is not None
        else "indisponível por ausência de pares de scores"
    )
    lines = [
        "# Avaliação dos resultados",
        "",
        "## Resultado executivo",
        "",
        f"A avaliação analisou {metrics['total']} registros. Este comando não "
        "altera as decisões oficiais; o experimento opcional é separado da "
        "saída final. Não há ground truth para medir precisão.",
        "",
        "`FOUND percentage` mede cobertura, não acurácia. Sem rótulos legítimos, "
        "não é possível estimar precisão ou falsos positivos confirmados.",
        "",
        "## Comparação",
        "",
        "A comparação é entre etapas da mesma execução, não entre versões "
        "históricas do projeto.",
        "",
        "| Etapa | FOUND | AMBIGUOUS | NOT_FOUND |",
        "|---|---:|---:|---:|",
    ]
    for version in comparison:
        lines.append(
            f"| {version['version']} | {version['FOUND']} | "
            f"{version['AMBIGUOUS']} | {version['NOT_FOUND']} |"
        )
    lines.extend(
        [
            "",
            "## FOUND e baixa confiança",
            "",
            f"- {found['baseline_exact']} `FOUND` vieram de igualdade exata e "
            "setor único na baseline.",
            f"- {found['elasticsearch_structured']} vieram de evidência "
            "estruturada forte do Elasticsearch.",
            f"- {found['near_strong_threshold']} resultados `FOUND` ficaram "
            f"dentro da banda de auditoria de {metrics['near_threshold_band']:g} "
            "acima do threshold forte; a banda não muda a decisão.",
            f"- O arquivo `low_confidence_cases.csv` contém "
            f"{metrics['low_confidence_cases']} casos: ambiguidades e `FOUND` "
            "próximos do threshold.",
            "",
            "Os flags de revisão indicam ausência de alguma concordância, não "
            "um falso positivo. Todos os `FOUND` Elasticsearch mantiveram "
            "município, logradouro, número/ausência de número e as âncoras "
            "estruturadas exigidas pela política.",
            "",
            "Flags nos `FOUND` que merecem inspeção:",
            "",
        ]
    )
    for flag, count in found["review_flag_counts"].items():
        lines.append(f"- `{flag}`: {count}")
    lines.extend(
        [
            "",
            f"{found['with_review_flags']} resultados `FOUND` possuem flags de "
            "revisão. Sem rótulos, não é possível confirmar a ausência de falsos "
            "positivos. As flags indicam pontos de revisão, não justificam "
            "sozinhas alterar o setor nem aceitar um resultado pelo `_score`.",
            "",
            "## AMBIGUOUS",
            "",
        ]
    )
    for category, count in metrics["ambiguous_by_category"].items():
        lines.append(f"- `{category}`: {count}")
    lines.extend(["", "## NOT_FOUND", ""])
    for category, count in metrics["not_found_by_category"].items():
        lines.append(f"- `{category}`: {count}")
    lines.extend(
        [
            "",
            "### Padrões de qualidade por status",
            "",
            "| Status | Padrão | Casos |",
            "|---|---|---:|",
        ]
    )
    for status in ("FOUND", "AMBIGUOUS", "NOT_FOUND"):
        for flag, count in metrics["quality_patterns_by_status"].get(
            status, {}
        ).items():
            lines.append(f"| {status} | `{flag}` | {count} |")
    lines.extend(
        [
            "",
            "Os padrões são flags independentes e podem coexistir no mesmo "
            f"registro. Há {missing_postal_codes} registros com CEP normalizado "
            "ausente ou inválido. Essas contagens descrevem a entrada; não "
            "comprovam, isoladamente, a causa de uma falha de matching.",
            "",
            "## Normalização",
            "",
            f"A busca teve {normalization['search_collision_groups']} colisões. "
            f"No CNEFE, {normalization['cnefe_collapsed_distinct_addresses']} "
            "representações originais foram colapsadas, "
            f"{normalization['cnefe_collapsed_percentage']:.4f}% das "
            "representações distintas. Existem "
            f"{normalization['cnefe_normalized_groups_with_multiple_sectors']} "
            "grupos normalizados em múltiplos setores; por isso a pipeline "
            "preserva ambiguidades e não deduplica por endereço normalizado.",
            "",
            f"O teste de estabilidade manteve "
            f"{metrics['stability']['stable_records']}/"
            f"{metrics['stability']['tested_records']} resultados normalizados "
            "sob mudanças de caixa, acento, espaço e pontuação.",
            "",
            "## Elasticsearch: filtros e fuzzy",
            "",
            f"Entre os {retrieval['no_candidate_cases']} casos sem candidatos, "
            f"{retrieval['no_candidate_cases_after_municipality_only_fallback']} "
            "chegaram ao fallback com filtro apenas de município. Nesses casos, "
            "CEP e número não estavam nos filtros da última consulta; o "
            "requisito textual `AND` permanece um ponto a investigar.",
            "",
            f"O fuzzy reteve {retrieval['fuzzy_retained_candidates']} candidatos; "
            f"{retrieval['fuzzy_evidence_strength'].get('WEAK', 0)} "
            f"({retrieval['fuzzy_weak_percentage']:.2f}%) foram fracos e rejeitados "
            "pelo selector. A recuperação fuzzy não promove automaticamente "
            "candidatos a `FOUND`.",
            "",
            "## Experimento de consulta relaxada",
            "",
        ]
    )
    if relaxed.get("executed"):
        proposed = relaxed["proposed_status"]
        lines.extend(
            [
                f"A consulta experimental foi aplicada somente aos "
                f"{relaxed['target_no_candidate_cases']} casos sem candidatos e "
                f"encontrou hits em {relaxed['cases_with_hits']}.",
                "",
                f"Proposta isolada: FOUND={proposed['FOUND']}, "
                f"AMBIGUOUS={proposed['AMBIGUOUS']}, "
                f"NOT_FOUND={proposed['NOT_FOUND']}. A experiência não altera "
                "o resultado oficial porque não há ground truth para medir o "
                "efeito sobre precisão.",
            ]
        )
    else:
        lines.append(
            "Não executado. Use `evaluate-results "
            "--with-relaxed-query-experiment` com Elasticsearch disponível."
        )
    lines.extend(
        [
            "",
            "## Sensibilidade e decisão",
            "",
            "| Cenário | Forte | Plausível | FOUND | AMBIGUOUS | NOT_FOUND |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for scenario in metrics["threshold_sensitivity"]:
        lines.append(
            f"| {scenario['scenario']} | "
            f"{scenario['strong_street_similarity']:.2f} | "
            f"{scenario['plausible_street_similarity']:.2f} | "
            f"{scenario['FOUND']} | {scenario['AMBIGUOUS']} | "
            f"{scenario['NOT_FOUND']} |"
        )
    lines.extend(
        [
            "",
            "A tabela registra a cobertura de cada política nesta execução. "
            f"A mediana da margem absoluta de `_score` foi {margin_description}. Sem "
            "rótulos, alterar thresholds aumentaria ou reduziria cobertura sem "
            "demonstrar ganho de precisão.",
            "",
            "### Métodos por etapa",
            "",
            "| Etapa | Método | Casos |",
            "|---|---|---:|",
        ]
    )
    for version, methods in metrics["methods_by_version"].items():
        for method, count in methods.items():
            lines.append(f"| {version} | `{method}` | {count} |")
    lines.extend(
        [
            "",
            "## Limitações e próximos passos",
            "",
            "- ausência de ground truth;",
            "- bairro da busca é comparado à localidade CNEFE apenas como "
            "evidência, sem equivalência semântica geral;",
            f"- top {metrics['retrieval_top_k']} pode ocultar setores adicionais;",
            "- endereços rurais, sem número e com componentes misturados "
            "continuam difíceis;",
            "- o experimento relaxado precisa de validação rotulada antes de "
            "entrar no retrieval oficial.",
            "",
            "As melhorias futuras devem ser avaliadas com dados rotulados; sem "
            "ground truth, aumentar cobertura não demonstra maior corretude.",
            "",
            "## Reprodução",
            "",
            "```bash",
            "docker compose up -d --wait elasticsearch",
            "docker compose run --rm app python -m src.main evaluate-results "
            "--with-relaxed-query-experiment",
            "```",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def execute_evaluation(
    *,
    final_audit_path: Path,
    evidence_path: Path,
    selection_metrics_path: Path,
    retrieval_metrics_path: Path,
    normalization_report_path: Path,
    diagnostics_path: Path,
    normalized_search_path: Path,
    cases_path: Path,
    low_confidence_path: Path,
    metrics_path: Path,
    report_path: Path,
    policy: SelectionPolicy,
    near_threshold_band: float,
    expected_records: int | None,
    client: EvaluationSearchClient | None = None,
    index_name: str | None = None,
    top_k: int = 10,
) -> EvaluationOutput:
    """Avalia a saída oficial; o experimento opcional não a modifica."""
    audit = _read_csv(final_audit_path, AUDIT_REQUIRED_COLUMNS, "auditoria final")
    evidence = _read_csv(
        evidence_path,
        EVIDENCE_REQUIRED_COLUMNS,
        "evidencias de candidatos",
        allow_empty=True,
    )
    selection_metrics = _read_json(
        selection_metrics_path, "metricas de candidate selection"
    )
    retrieval_metrics = _read_json(
        retrieval_metrics_path, "metricas de candidate retrieval"
    )
    normalization_report = _read_json(
        normalization_report_path, "relatorio de normalizacao"
    )
    diagnostics = _read_diagnostics(diagnostics_path)
    normalized_search = _read_csv(
        normalized_search_path, SEARCH_COLUMNS, "busca normalizada"
    )
    if expected_records is not None and len(audit) != expected_records:
        raise DatasetValidationError(
            f"Avaliacao esperava {expected_records} registros; recebeu {len(audit)}."
        )
    cases = build_case_evaluation(
        audit,
        evidence,
        policy=policy,
        near_threshold_band=near_threshold_band,
    )
    stability = analyze_normalization_stability(normalized_search)
    relaxed_metrics: dict[str, object] = {
        "executed": False,
        "changes_official_results": False,
    }
    if client is not None:
        if not index_name:
            raise ValueError("index_name e obrigatorio no experimento online.")
        experiment, relaxed_metrics = run_relaxed_retrieval_experiment(
            client,
            index_name=index_name,
            cases=cases,
            diagnostics=diagnostics,
            policy=policy,
            top_k=top_k,
        )
        cases = cases.merge(
            experiment, on="input_id", how="left", validate="one_to_one"
        )

    metrics = build_evaluation_metrics(
        cases,
        selection_metrics=selection_metrics,
        retrieval_metrics=retrieval_metrics,
        normalization_report=normalization_report,
        diagnostics=diagnostics,
        evidence=evidence,
        stability=stability,
        relaxed_experiment=relaxed_metrics,
        near_threshold_band=near_threshold_band,
        top_k=top_k,
    )
    low_confidence = cases[
        cases["status_match"].eq("AMBIGUOUS")
        | cases["confidence_band"].eq("REVIEW_NEAR_STRONG_THRESHOLD")
    ]
    for output_path in (
        cases_path,
        low_confidence_path,
        metrics_path,
        report_path,
    ):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    cases.to_csv(cases_path, sep=";", encoding="utf-8", index=False)
    low_confidence.to_csv(
        low_confidence_path, sep=";", encoding="utf-8", index=False
    )
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_evaluation_report(report_path, metrics)
    LOGGER.info(
        "Avaliacao concluida: total=%s, baixa_confianca=%s, relatorio=%s",
        len(cases),
        len(low_confidence),
        report_path,
    )
    return EvaluationOutput(
        cases_path=cases_path,
        low_confidence_path=low_confidence_path,
        metrics_path=metrics_path,
        report_path=report_path,
        total_records=len(cases),
    )
