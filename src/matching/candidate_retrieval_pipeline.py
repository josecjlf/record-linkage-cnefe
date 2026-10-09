"""Orquestracao e analise do candidate retrieval no Elasticsearch."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
import logging
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

import pandas as pd

from src.ingestion.validation import (
    DatasetValidationError,
    validate_required_columns,
)
from src.matching.candidate_retriever import (
    CandidateRetrieval,
    SearchClient,
    StructuredAddress,
    retrieve_candidates,
)


LOGGER = logging.getLogger(__name__)

BASELINE_RETRIEVAL_COLUMNS = (
    "input_id",
    "municipio_codigo_normalizado",
    "cep_normalizado",
    "logradouro_normalizado",
    "numero_comparavel_normalizado",
    "bairro_normalizado",
    "complemento_normalizado",
    "endereco_normalizado",
    "setor_censitario",
    "status_match",
    "metodo_match",
)

CANDIDATE_OUTPUT_COLUMNS = (
    "input_id",
    "baseline_status",
    "baseline_setor_censitario",
    "logradouro_busca",
    "numero_busca",
    "candidate_rank",
    "id_cnefe",
    "setor_censitario",
    "codigo_municipio",
    "cep",
    "logradouro",
    "numero_comparavel",
    "localidade",
    "complemento",
    "endereco_normalizado",
    "score",
    "query_strategy",
    "strategy_order",
)


@dataclass(frozen=True)
class AddressRetrievalRecord:
    address: StructuredAddress
    baseline_status: str
    baseline_sector: str | None
    baseline_method: str
    retrieval: CandidateRetrieval


@dataclass(frozen=True)
class CandidateRetrievalRun:
    records: tuple[AddressRetrievalRecord, ...]
    metrics: dict[str, Any]


@dataclass(frozen=True)
class CandidateRetrievalOutput:
    diagnostics_path: Path
    candidates_path: Path
    metrics_path: Path
    analysis_path: Path
    total_records: int


def read_baseline_results(
    path: Path, *, expected_records: int | None = None
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Resultado da baseline nao encontrado: {path}")
    dataframe = pd.read_csv(
        path,
        sep=";",
        encoding="utf-8",
        dtype="string",
        keep_default_na=False,
    )
    validate_required_columns(
        dataframe.columns,
        BASELINE_RETRIEVAL_COLUMNS,
        "resultado da baseline para candidate retrieval",
    )
    if dataframe.empty:
        raise DatasetValidationError("Resultado da baseline esta vazio.")
    if dataframe["input_id"].duplicated().any():
        raise DatasetValidationError("Resultado da baseline possui input_id duplicado.")
    if expected_records is not None and len(dataframe) != expected_records:
        raise DatasetValidationError(
            "Quantidade inesperada no resultado da baseline: "
            f"esperado={expected_records}, encontrado={len(dataframe)}."
        )
    return dataframe


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "minimum": None,
            "p25": None,
            "median": None,
            "p75": None,
            "maximum": None,
            "mean": None,
        }
    series = pd.Series(values, dtype="float64")
    return {
        "count": len(values),
        "minimum": round(float(series.min()), 6),
        "p25": round(float(series.quantile(0.25)), 6),
        "median": round(float(series.median()), 6),
        "p75": round(float(series.quantile(0.75)), 6),
        "maximum": round(float(series.max()), 6),
        "mean": round(float(mean(values)), 6),
    }


def _group_metrics(
    records: Sequence[AddressRetrievalRecord], status: str
) -> dict[str, int | float]:
    selected = [record for record in records if record.baseline_status == status]
    with_candidates = sum(bool(record.retrieval.candidates) for record in selected)
    candidate_total = sum(len(record.retrieval.candidates) for record in selected)
    return {
        "total": len(selected),
        "with_candidates": with_candidates,
        "without_candidates": len(selected) - with_candidates,
        "coverage_percentage": (
            round((with_candidates / len(selected)) * 100, 2)
            if selected
            else 0.0
        ),
        "average_candidates": (
            round(candidate_total / len(selected), 2) if selected else 0.0
        ),
    }


def build_retrieval_metrics(
    records: Sequence[AddressRetrievalRecord],
    *,
    index_name: str,
    top_k: int,
    minimum_candidates: int,
) -> dict[str, Any]:
    candidate_counts = [len(record.retrieval.candidates) for record in records]
    without_candidates = [
        record.address.input_id
        for record in records
        if not record.retrieval.candidates
    ]
    multiple_sectors = [
        record.address.input_id
        for record in records
        if len(
            {
                candidate.setor_censitario
                for candidate in record.retrieval.candidates
            }
        )
        > 1
    ]
    fuzzy_only = [
        record.address.input_id
        for record in records
        if record.retrieval.candidates
        and all(
            candidate.query_strategy == "FUZZY_STREET_WITH_MUNICIPALITY"
            for candidate in record.retrieval.candidates
        )
    ]

    attempt_metrics: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "attempted_addresses": 0,
            "addresses_with_hits": 0,
            "total_hits": 0,
            "returned_hits": 0,
        }
    )
    candidate_strategy_counts: Counter[str] = Counter()
    score_values: dict[str, list[float]] = defaultdict(list)
    top_score_gaps: dict[str, list[float]] = defaultdict(list)
    tied_top_scores: Counter[str] = Counter()
    for record in records:
        for attempt in record.retrieval.attempts:
            strategy_metrics = attempt_metrics[attempt.strategy]
            strategy_metrics["attempted_addresses"] += 1
            strategy_metrics["addresses_with_hits"] += int(
                attempt.returned_hits > 0
            )
            strategy_metrics["total_hits"] += attempt.total_hits
            strategy_metrics["returned_hits"] += attempt.returned_hits
        for candidate in record.retrieval.candidates:
            candidate_strategy_counts[candidate.query_strategy] += 1
            if candidate.score is not None:
                score_values[candidate.query_strategy].append(candidate.score)
        candidates = record.retrieval.candidates
        if (
            len(candidates) >= 2
            and candidates[0].query_strategy == candidates[1].query_strategy
            and candidates[0].score is not None
            and candidates[1].score is not None
        ):
            strategy = candidates[0].query_strategy
            gap = candidates[0].score - candidates[1].score
            top_score_gaps[strategy].append(gap)
            if abs(gap) < 1e-9:
                tied_top_scores[strategy] += 1

    baseline_found = [
        record for record in records if record.baseline_status == "FOUND"
    ]
    target_unresolved = [
        record for record in records if record.baseline_status != "FOUND"
    ]
    target_with_candidates = sum(
        bool(record.retrieval.candidates) for record in target_unresolved
    )
    baseline_sector_retrieved = sum(
        bool(record.baseline_sector)
        and any(
            candidate.setor_censitario == record.baseline_sector
            for candidate in record.retrieval.candidates
        )
        for record in baseline_found
    )
    total = len(records)
    with_candidates = total - len(without_candidates)
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "index_name": index_name,
        "top_k": top_k,
        "fallback_minimum_candidates": minimum_candidates,
        "total_addresses": total,
        "target_unresolved_addresses": sum(
            record.baseline_status != "FOUND" for record in records
        ),
        "target_unresolved_with_candidates": target_with_candidates,
        "target_unresolved_coverage_percentage": (
            round(
                (target_with_candidates / len(target_unresolved)) * 100,
                2,
            )
            if target_unresolved
            else 0.0
        ),
        "addresses_with_candidates": with_candidates,
        "addresses_without_candidates": len(without_candidates),
        "coverage_percentage": (
            round((with_candidates / total) * 100, 2) if total else 0.0
        ),
        "candidate_count_distribution": dict(
            sorted(Counter(candidate_counts).items())
        ),
        "by_baseline_status": {
            status: _group_metrics(records, status)
            for status in ("FOUND", "AMBIGUOUS", "NOT_FOUND")
        },
        "attempts_by_strategy": dict(attempt_metrics),
        "retained_candidates_by_strategy": dict(candidate_strategy_counts),
        "retained_candidates_total": sum(candidate_strategy_counts.values()),
        "score_distribution_by_strategy": {
            strategy: _distribution(values)
            for strategy, values in score_values.items()
        },
        "top_score_gap_distribution_by_strategy": {
            strategy: _distribution(values)
            for strategy, values in top_score_gaps.items()
        },
        "tied_top_scores_by_strategy": dict(tied_top_scores),
        "addresses_with_multiple_candidate_sectors": len(multiple_sectors),
        "addresses_with_only_fuzzy_candidates": len(fuzzy_only),
        "baseline_found_control": {
            "total": len(baseline_found),
            "baseline_sector_retrieved": baseline_sector_retrieved,
            "recall_percentage": (
                round(
                    (baseline_sector_retrieved / len(baseline_found)) * 100,
                    2,
                )
                if baseline_found
                else 0.0
            ),
        },
        "address_ids_without_candidates": without_candidates,
        "address_ids_with_multiple_candidate_sectors": multiple_sectors,
        "address_ids_with_only_fuzzy_candidates": fuzzy_only,
    }


def run_candidate_retrieval(
    client: SearchClient,
    baseline: pd.DataFrame,
    *,
    index_name: str,
    top_k: int,
    minimum_candidates: int,
) -> CandidateRetrievalRun:
    validate_required_columns(
        baseline.columns,
        BASELINE_RETRIEVAL_COLUMNS,
        "resultado da baseline para candidate retrieval",
    )
    records: list[AddressRetrievalRecord] = []
    for position, row in enumerate(baseline.to_dict(orient="records"), start=1):
        address = StructuredAddress.from_mapping(row)
        retrieval = retrieve_candidates(
            client,
            index_name=index_name,
            address=address,
            top_k=top_k,
            minimum_candidates=minimum_candidates,
        )
        records.append(
            AddressRetrievalRecord(
                address=address,
                baseline_status=str(row["status_match"]),
                baseline_sector=(
                    str(row["setor_censitario"]).strip() or None
                ),
                baseline_method=str(row["metodo_match"]),
                retrieval=retrieval,
            )
        )
        if position % 10 == 0 or position == len(baseline):
            LOGGER.info(
                "Candidate retrieval: processados=%s/%s",
                position,
                len(baseline),
            )
    metrics = build_retrieval_metrics(
        records,
        index_name=index_name,
        top_k=top_k,
        minimum_candidates=minimum_candidates,
    )
    return CandidateRetrievalRun(records=tuple(records), metrics=metrics)


def _diagnostic_record(record: AddressRetrievalRecord) -> dict[str, Any]:
    return {
        "input_id": record.address.input_id,
        "baseline": {
            "status_match": record.baseline_status,
            "setor_censitario": record.baseline_sector,
            "metodo_match": record.baseline_method,
        },
        "address": asdict(record.address),
        "attempts": [asdict(attempt) for attempt in record.retrieval.attempts],
        "candidates": [
            candidate.as_record()
            for candidate in record.retrieval.candidates
        ],
    }


def write_retrieval_artifacts(
    run: CandidateRetrievalRun,
    *,
    diagnostics_path: Path,
    candidates_path: Path,
    metrics_path: Path,
) -> None:
    for path in (diagnostics_path, candidates_path, metrics_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_text(
        "\n".join(
            json.dumps(
                _diagnostic_record(record), ensure_ascii=False, default=str
            )
            for record in run.records
        )
        + "\n",
        encoding="utf-8",
    )

    rows: list[dict[str, object]] = []
    for record in run.records:
        for candidate in record.retrieval.candidates:
            rows.append(
                {
                    "input_id": record.address.input_id,
                    "baseline_status": record.baseline_status,
                    "baseline_setor_censitario": record.baseline_sector,
                    "logradouro_busca": record.address.street,
                    "numero_busca": record.address.number,
                    "candidate_rank": candidate.rank,
                    **{
                        key: value
                        for key, value in candidate.as_record().items()
                        if key != "rank"
                    },
                }
            )
    pd.DataFrame(rows, columns=CANDIDATE_OUTPUT_COLUMNS).to_csv(
        candidates_path,
        sep=";",
        encoding="utf-8",
        index=False,
    )
    metrics_path.write_text(
        json.dumps(run.metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _markdown_cell(value: object) -> str:
    return str(value if value is not None else "").replace("|", "\\|")


def _representative_examples(
    records: Sequence[AddressRetrievalRecord],
) -> list[AddressRetrievalRecord]:
    examples: list[AddressRetrievalRecord] = []
    selected_ids: set[str] = set()
    for strategy in (
        "EXACT_STRUCTURED",
        "TEXT_WITH_GEOGRAPHY",
        "FUZZY_STREET_WITH_MUNICIPALITY",
    ):
        match = next(
            (
                record
                for record in records
                if record.retrieval.candidates
                and record.retrieval.candidates[0].query_strategy == strategy
            ),
            None,
        )
        if match is not None and match.address.input_id not in selected_ids:
            examples.append(match)
            selected_ids.add(match.address.input_id)
    difficult = next(
        (
            record
            for record in records
            if len(
                {
                    candidate.setor_censitario
                    for candidate in record.retrieval.candidates
                }
            )
            > 1
            and record.address.input_id not in selected_ids
        ),
        None,
    )
    if difficult is not None:
        examples.append(difficult)
        selected_ids.add(difficult.address.input_id)
    no_candidates = next(
        (
            record
            for record in records
            if not record.retrieval.candidates
            and record.address.input_id not in selected_ids
        ),
        None,
    )
    if no_candidates is not None:
        examples.append(no_candidates)
    return examples


def write_analysis_report(
    path: Path,
    run: CandidateRetrievalRun,
) -> None:
    metrics = run.metrics
    status_rows = metrics["by_baseline_status"]
    score_rows = metrics["score_distribution_by_strategy"]
    lines = [
        "# Recuperacao de candidatos com Elasticsearch",
        "",
        "## Escopo",
        "",
        "O Elasticsearch recupera ate `top_k` documentos plausiveis por entrada, "
        "mas nao decide `FOUND`, `AMBIGUOUS` ou `NOT_FOUND`. Os resultados da "
        "baseline permanecem inalterados. Os casos nao resolvidos sao o alvo; os "
        "`FOUND` da baseline foram consultados apenas como controle.",
        "",
        "## Queries progressivas",
        "",
        "| Ordem | Estrategia | Restricao | Papel |",
        "|---:|---|---|---|",
        "| 1 | `EXACT_STRUCTURED` | municipio/CEP + logradouro exato + numero, quando presente | Evidencia estrutural forte |",
        "| 2 | `TEXT_WITH_GEOGRAPHY` | municipio e numero informado em `filter`; todos os tokens do logradouro em `must`; CEP pontua | Variacao textual sem exigir CEP igual |",
        "| 3 | `FUZZY_STREET_WITH_MUNICIPALITY` | municipio em `filter`; fuzzy somente em campos textuais de logradouro | Ultimo fallback para erros simples |",
        "",
        "Os filtros usam os campos disponiveis: a query exata exige municipio "
        "ou CEP; a textual usa municipio quando informado, mas CEP somente "
        "pontua nessa etapa. Sem municipio, a textual pode ficar sem recorte "
        "geografico. O fallback fuzzy usa municipio ou, na sua ausencia, CEP "
        "como filtro. Nenhuma dessas consultas decide o setor final.",
        "",
        "A proxima query so roda enquanto houver menos de "
        f"{metrics['fallback_minimum_candidates']} candidatos unicos. O limite "
        f"final e {metrics['top_k']}. Candidatos repetidos entre queries sao "
        "mantidos uma vez, associados a primeira estrategia que os encontrou.",
        "",
        "`filter` aplica restricoes obrigatorias sem influenciar `_score`; `must` "
        "exige correspondencia e pontua; `should` adiciona evidencia de numero, "
        "CEP, bairro/localidade ou complemento, sem torna-la obrigatoria. Nao foi "
        "definido `minimum_should_match`: com `must`/`filter`, os `should` servem "
        "apenas para ranking. CEP e numero usam `term`, nunca fuzziness.",
        "",
        "## Cobertura observada",
        "",
        f"Indice: `{metrics['index_name']}`. Foram consultados "
        f"{metrics['total_addresses']} enderecos, dos quais "
        f"{metrics['target_unresolved_addresses']} nao eram `FOUND` na baseline.",
        "",
        "| Grupo da baseline | Total | Com candidatos | Sem candidatos | Cobertura | Media de candidatos |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for status in ("FOUND", "AMBIGUOUS", "NOT_FOUND"):
        group = status_rows[status]
        lines.append(
            f"| `{status}` | {group['total']} | {group['with_candidates']} | "
            f"{group['without_candidates']} | {group['coverage_percentage']:.2f}% | "
            f"{group['average_candidates']:.2f} |"
        )
    lines.extend(
        [
            "",
            f"Cobertura total: **{metrics['addresses_with_candidates']}/"
            f"{metrics['total_addresses']} ({metrics['coverage_percentage']:.2f}%)**. "
            f"No alvo nao resolvido, foram recuperados candidatos para "
            f"**{metrics['target_unresolved_with_candidates']}/"
            f"{metrics['target_unresolved_addresses']} "
            f"({metrics['target_unresolved_coverage_percentage']:.2f}%)**. "
            f"O CSV reteve {metrics['retained_candidates_total']} candidatos. "
            f"Ha {metrics['addresses_with_multiple_candidate_sectors']} entradas "
            "cujos candidatos recuperados abrangem mais de um setor; isso e "
            "evidencia de ambiguidade, nao uma decisao final.",
            "",
            "## Distribuicao de scores",
            "",
            "Scores so devem ser comparados dentro da mesma query. Uma query exata "
            "em contexto de `filter` pode produzir score zero, o que nao reduz a "
            "forca da igualdade estrutural. A distribuicao agregada abaixo e "
            "apenas descritiva e nao deve virar threshold automaticamente.",
            "",
            "| Estrategia | N | Min | P25 | Mediana | P75 | Max | Media |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for strategy, distribution in score_rows.items():
        lines.append(
            f"| `{strategy}` | {distribution['count']} | "
            f"{distribution['minimum']} | {distribution['p25']} | "
            f"{distribution['median']} | {distribution['p75']} | "
            f"{distribution['maximum']} | {distribution['mean']} |"
        )
    lines.extend(
        [
            "",
            "## Exemplos representativos",
            "",
            "A selecao abaixo e automatica: primeiro exemplo de cada estrategia, "
            "um caso com setores multiplos e um caso sem candidatos, quando existem.",
            "",
            "| Input | Baseline | Logradouro de busca | Candidatos | Setores | Primeira estrategia | Top score |",
            "|---|---|---|---:|---:|---|---:|",
        ]
    )
    for record in _representative_examples(run.records):
        candidates = record.retrieval.candidates
        sectors = len({candidate.setor_censitario for candidate in candidates})
        strategy = candidates[0].query_strategy if candidates else "SEM_CANDIDATOS"
        top_score = candidates[0].score if candidates else None
        lines.append(
            "| {input_id} | `{status}` | {street} | {count} | {sectors} | "
            "`{strategy}` | {score} |".format(
                input_id=_markdown_cell(record.address.input_id),
                status=_markdown_cell(record.baseline_status),
                street=_markdown_cell(record.address.street),
                count=len(candidates),
                sectors=sectors,
                strategy=_markdown_cell(strategy),
                score=_markdown_cell(top_score),
            )
        )
    lines.extend(
        [
            "",
            "## Casos dificeis e limites",
            "",
            f"- {metrics['addresses_without_candidates']} entradas continuaram sem "
            "candidatos; podem envolver informacao ausente, estrutura muito "
            "diferente ou grafia fora do alcance do fallback.",
            f"- {metrics['addresses_with_only_fuzzy_candidates']} entradas tiveram "
            "apenas candidatos vindos do fallback fuzzy; por isso o selector exige "
            "evidencias estruturadas antes de promover qualquer uma a `FOUND`.",
            f"- {metrics['addresses_with_multiple_candidate_sectors']} entradas "
            "recuperaram mais de um setor e nao podem ser resolvidas por `top 1`.",
            "- `_score` nao e probabilidade e scores de queries diferentes nao sao "
            "diretamente comparaveis.",
            "- Bairro da busca e localidade do CNEFE continuam como evidencias de "
            "ranking, sem pressupor equivalencia semantica perfeita.",
            "",
            "## Artefatos e reproducao",
            "",
            "- `data/output/candidate_retrieval_diagnostics.jsonl`: endereco, "
            "baseline, query completa, total de hits e candidatos por entrada;",
            "- `data/output/candidate_retrieval_candidates.csv`: candidatos em "
            "formato tabular;",
            "- `data/output/candidate_retrieval_metrics.json`: metricas agregadas.",
            "",
            "```bash",
            "docker compose up -d --wait elasticsearch",
            "docker compose run --rm app python -m src.main retrieve-candidates",
            "```",
            "",
            "Referencias: [bool query](https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-bool-query), "
            "[multi_match](https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-multi-match-query) e "
            "[term-level queries](https://www.elastic.co/docs/reference/query-languages/query-dsl/term-level-queries).",
            "",
            "O candidate selector avalia esses candidatos com regras explicitas "
            "documentadas em `docs/matching_strategy.md`. Este relatorio permanece "
            "restrito a retrieval e nao promove candidato a resultado final.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def execute_candidate_retrieval(
    client: SearchClient,
    *,
    baseline_path: Path,
    index_name: str,
    diagnostics_path: Path,
    candidates_path: Path,
    metrics_path: Path,
    analysis_path: Path,
    top_k: int,
    minimum_candidates: int,
    expected_records: int | None = None,
) -> CandidateRetrievalOutput:
    """Executa retrieval e grava candidatos e diagnósticos, sem selecioná-los."""
    baseline = read_baseline_results(
        baseline_path, expected_records=expected_records
    )
    run = run_candidate_retrieval(
        client,
        baseline,
        index_name=index_name,
        top_k=top_k,
        minimum_candidates=minimum_candidates,
    )
    write_retrieval_artifacts(
        run,
        diagnostics_path=diagnostics_path,
        candidates_path=candidates_path,
        metrics_path=metrics_path,
    )
    write_analysis_report(analysis_path, run)
    LOGGER.info(
        "Retrieval concluido: total=%s, com_candidatos=%s, sem_candidatos=%s",
        run.metrics["total_addresses"],
        run.metrics["addresses_with_candidates"],
        run.metrics["addresses_without_candidates"],
    )
    return CandidateRetrievalOutput(
        diagnostics_path=diagnostics_path,
        candidates_path=candidates_path,
        metrics_path=metrics_path,
        analysis_path=analysis_path,
        total_records=len(run.records),
    )
