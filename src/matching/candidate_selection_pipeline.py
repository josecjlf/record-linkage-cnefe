"""Orquestracao auditavel da selecao final entre candidatos recuperados."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import logging
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from src.ingestion.validation import (
    DatasetValidationError,
    validate_required_columns,
)
from src.matching.candidate_selector import (
    DEFAULT_SELECTION_POLICY,
    MatchStatus,
    SelectionAddress,
    SelectionCandidate,
    SelectionDecision,
    SelectionPolicy,
    select_retrieved_candidates,
)


LOGGER = logging.getLogger(__name__)

SELECTION_REQUIRED_COLUMNS = (
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

BASELINE_RESULT_COLUMNS = (
    "setor_censitario",
    "status_match",
    "metodo_match",
    "quantidade_candidatos",
    "quantidade_linhas_candidatas",
    "quantidade_setores_candidatos",
    "score_match",
    "id_candidato_cnefe",
    "setores_candidatos",
)


EVIDENCE_OUTPUT_COLUMNS = (
    "input_id",
    "baseline_status",
    "es_proposed_status",
    "final_status",
    "final_sector",
    "input_municipality_code",
    "input_postal_code",
    "input_street",
    "input_number",
    "input_neighborhood",
    "candidate_municipality_code",
    "candidate_postal_code",
    "candidate_street",
    "candidate_number",
    "candidate_neighborhood",
    "candidate_complement",
    "candidate_id",
    "sector",
    "retrieval_strategy",
    "retrieval_rank",
    "es_score",
    "municipality_match",
    "postal_code_available",
    "postal_code_match",
    "street_exact",
    "street_similarity",
    "street_input_token_coverage",
    "street_candidate_extra_tokens",
    "informative_number_available",
    "number_match",
    "no_number_marker_match",
    "neighborhood_available",
    "neighborhood_match",
    "complement_available",
    "complement_match",
    "evidence_strength",
    "evidence_reasons",
)


@dataclass(frozen=True)
class SelectionInput:
    baseline: dict[str, Any]
    address: SelectionAddress
    candidates: tuple[SelectionCandidate, ...]


@dataclass(frozen=True)
class FinalSelection:
    input_id: str
    baseline_status: MatchStatus
    baseline_sector: str | None
    baseline_method: str
    es_decision: SelectionDecision
    final_status: MatchStatus
    final_sector: str | None
    final_method: str
    final_candidate_id: str | None

    def as_record(self) -> dict[str, object]:
        es = self.es_decision
        return {
            "input_id": self.input_id,
            "setor_censitario": self.final_sector,
            "status_match": self.final_status.value,
            "metodo_match": self.final_method,
            "score_match": (
                es.top_score if self.final_method.startswith("ES_") else None
            ),
            "id_candidato_cnefe": self.final_candidate_id,
            "top_score": es.top_score,
            "second_score": es.second_score,
            "score_gap": es.score_gap,
            "score_gap_ratio": es.score_gap_ratio,
            "scores_comparable": es.scores_comparable,
            "candidate_count": es.candidate_count,
            "strong_candidate_count": es.strong_candidate_count,
            "plausible_candidate_count": es.plausible_candidate_count,
            "candidate_sector_count": es.candidate_sector_count,
            "candidate_sectors": "|".join(es.candidate_sectors),
            "supporting_candidate_ids": "|".join(
                es.supporting_candidate_ids
            ),
            "top_evidence_strength": (
                es.top_evidence_strength.value
                if es.top_evidence_strength
                else None
            ),
            "evidence": "|".join(es.evidence),
            "es_proposed_status": es.status_match.value,
            "es_proposed_sector": es.setor_censitario,
            "es_proposed_method": es.metodo_match,
        }


@dataclass(frozen=True)
class CandidateSelectionRun:
    selections: tuple[FinalSelection, ...]
    metrics: dict[str, Any]


@dataclass(frozen=True)
class CandidateSelectionOutput:
    audit_path: Path
    evidence_path: Path
    metrics_path: Path
    report_path: Path
    total_records: int


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def read_baseline_for_selection(
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
        SELECTION_REQUIRED_COLUMNS,
        "baseline para candidate selection",
    )
    if dataframe.empty:
        raise DatasetValidationError("Resultado da baseline esta vazio.")
    if dataframe["input_id"].duplicated().any():
        raise DatasetValidationError("Baseline possui input_id duplicado.")
    if expected_records is not None and len(dataframe) != expected_records:
        raise DatasetValidationError(
            "Quantidade inesperada na baseline: "
            f"esperado={expected_records}, encontrado={len(dataframe)}."
        )
    return dataframe


def read_retrieval_diagnostics(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Diagnostico de candidate retrieval nao encontrado: {path}"
        )
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise DatasetValidationError("Diagnostico de retrieval esta vazio.")
    indexed = {record["input_id"]: record for record in records}
    if len(indexed) != len(records):
        raise DatasetValidationError(
            "Diagnostico de retrieval possui input_id duplicado."
        )
    return indexed


def _address_from_diagnostic(record: dict[str, Any]) -> SelectionAddress:
    address = record["address"]
    return SelectionAddress(
        input_id=record["input_id"],
        municipality_code=_optional_text(address.get("municipality_code")),
        postal_code=_optional_text(address.get("postal_code")),
        street=_optional_text(address.get("street")),
        number=_optional_text(address.get("number")),
        neighborhood=_optional_text(address.get("neighborhood")),
        complement=_optional_text(address.get("complement")),
    )


def _candidate_from_diagnostic(record: dict[str, Any]) -> SelectionCandidate:
    return SelectionCandidate(
        candidate_id=str(record["id_cnefe"]),
        sector=str(record["setor_censitario"]),
        municipality_code=_optional_text(record.get("codigo_municipio")),
        postal_code=_optional_text(record.get("cep")),
        street=_optional_text(record.get("logradouro")),
        number=_optional_text(record.get("numero_comparavel")),
        neighborhood=_optional_text(record.get("localidade")),
        complement=_optional_text(record.get("complemento")),
        es_score=(
            float(record["score"])
            if record.get("score") is not None
            else None
        ),
        retrieval_strategy=str(record["query_strategy"]),
        retrieval_rank=int(record["rank"]),
    )


def _validate_diagnostic_matches_baseline(
    baseline_row: dict[str, Any], diagnostic: dict[str, Any]
) -> None:
    input_id = str(baseline_row["input_id"])
    diagnostic_baseline = diagnostic.get("baseline", {})
    for column in ("status_match", "setor_censitario", "metodo_match"):
        if column not in diagnostic_baseline or _optional_text(
            diagnostic_baseline[column]
        ) != _optional_text(baseline_row[column]):
            raise DatasetValidationError(
                f"Baseline e diagnostico divergem em {column}: {input_id}. "
                "Execute retrieve-candidates novamente a partir da baseline atual."
            )

    diagnostic_address = diagnostic.get("address", {})
    address_columns = {
        "input_id": "input_id",
        "municipality_code": "municipio_codigo_normalizado",
        "postal_code": "cep_normalizado",
        "street": "logradouro_normalizado",
        "number": "numero_comparavel_normalizado",
        "neighborhood": "bairro_normalizado",
        "complement": "complemento_normalizado",
        "normalized_address": "endereco_normalizado",
    }
    for diagnostic_column, baseline_column in address_columns.items():
        if diagnostic_column not in diagnostic_address or _optional_text(
            diagnostic_address[diagnostic_column]
        ) != _optional_text(baseline_row[baseline_column]):
            raise DatasetValidationError(
                f"Endereco da baseline diverge em {baseline_column} "
                f"no diagnostico: {input_id}. Execute retrieve-candidates "
                "novamente a partir da baseline atual."
            )


def build_selection_inputs(
    baseline: pd.DataFrame,
    diagnostics: dict[str, dict[str, Any]],
) -> tuple[SelectionInput, ...]:
    baseline_ids = list(baseline["input_id"])
    if set(baseline_ids) != set(diagnostics):
        missing = sorted(set(baseline_ids) - set(diagnostics))
        extra = sorted(set(diagnostics) - set(baseline_ids))
        raise DatasetValidationError(
            "Baseline e retrieval possuem IDs diferentes: "
            f"ausentes={missing[:5]}, extras={extra[:5]}."
        )
    inputs: list[SelectionInput] = []
    for baseline_row in baseline.to_dict(orient="records"):
        input_id = str(baseline_row["input_id"])
        diagnostic = diagnostics[input_id]
        _validate_diagnostic_matches_baseline(baseline_row, diagnostic)
        inputs.append(
            SelectionInput(
                baseline=baseline_row,
                address=_address_from_diagnostic(diagnostic),
                candidates=tuple(
                    _candidate_from_diagnostic(candidate)
                    for candidate in diagnostic["candidates"]
                ),
            )
        )
    return tuple(inputs)


def _finalize_selection(
    item: SelectionInput,
    es_decision: SelectionDecision,
) -> FinalSelection:
    baseline_status = MatchStatus(str(item.baseline["status_match"]))
    baseline_sector = _optional_text(item.baseline.get("setor_censitario"))
    baseline_method = str(item.baseline["metodo_match"])
    if baseline_status == MatchStatus.FOUND:
        if not baseline_sector:
            raise DatasetValidationError(
                f"Baseline FOUND sem setor: {item.address.input_id}."
            )
        return FinalSelection(
            input_id=item.address.input_id,
            baseline_status=baseline_status,
            baseline_sector=baseline_sector,
            baseline_method=baseline_method,
            es_decision=es_decision,
            final_status=MatchStatus.FOUND,
            final_sector=baseline_sector,
            final_method="BASELINE_FOUND_PRESERVED",
            final_candidate_id=_optional_text(
                item.baseline.get("id_candidato_cnefe")
            ),
        )
    if baseline_status == MatchStatus.AMBIGUOUS:
        return FinalSelection(
            input_id=item.address.input_id,
            baseline_status=baseline_status,
            baseline_sector=None,
            baseline_method=baseline_method,
            es_decision=es_decision,
            final_status=MatchStatus.AMBIGUOUS,
            final_sector=None,
            final_method="BASELINE_AMBIGUOUS_PRESERVED",
            final_candidate_id=None,
        )
    return FinalSelection(
        input_id=item.address.input_id,
        baseline_status=baseline_status,
        baseline_sector=None,
        baseline_method=baseline_method,
        es_decision=es_decision,
        final_status=es_decision.status_match,
        final_sector=es_decision.setor_censitario,
        final_method=es_decision.metodo_match,
        final_candidate_id=es_decision.selected_candidate_id,
    )


def apply_selection_policy(
    inputs: Sequence[SelectionInput],
    *,
    policy: SelectionPolicy = DEFAULT_SELECTION_POLICY,
) -> tuple[FinalSelection, ...]:
    return tuple(
        _finalize_selection(
            item,
            select_retrieved_candidates(
                item.address, item.candidates, policy=policy
            ),
        )
        for item in inputs
    )


def _status_counts(values: Sequence[MatchStatus]) -> dict[str, int]:
    counts = Counter(value.value for value in values)
    return {
        status.value: counts.get(status.value, 0)
        for status in MatchStatus
    }


def _transition_counts(
    selections: Sequence[FinalSelection], *, target: str
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for selection in selections:
        destination = (
            selection.es_decision.status_match
            if target == "elasticsearch"
            else selection.final_status
        )
        counts[
            f"{selection.baseline_status.value}->{destination.value}"
        ] += 1
    return dict(sorted(counts.items()))


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "minimum": None,
            "p25": None,
            "median": None,
            "p75": None,
            "maximum": None,
        }
    series = pd.Series(values, dtype="float64")
    return {
        "count": len(values),
        "minimum": round(float(series.min()), 6),
        "p25": round(float(series.quantile(0.25)), 6),
        "median": round(float(series.median()), 6),
        "p75": round(float(series.quantile(0.75)), 6),
        "maximum": round(float(series.max()), 6),
    }


def _score_margin_analysis(
    selections: Sequence[FinalSelection],
) -> dict[str, object]:
    gaps = [
        selection.es_decision.score_gap
        for selection in selections
        if selection.es_decision.score_gap is not None
    ]
    ratios = [
        selection.es_decision.score_gap_ratio
        for selection in selections
        if selection.es_decision.score_gap_ratio is not None
    ]
    return {
        "absolute_gap_distribution": _distribution(gaps),
        "relative_gap_distribution": _distribution(ratios),
        "relative_margin_sensitivity": {
            str(threshold): sum(ratio >= threshold for ratio in ratios)
            for threshold in (0.0, 0.02, 0.05, 0.10, 0.20)
        },
        "used_for_decision": False,
        "reason": (
            "A politica usa evidencias estruturadas, nao uma margem de _score. "
            "Scores de estrategias diferentes nao sao diretamente comparaveis "
            "e nao ha rotulos para calibrar um threshold de aceitacao."
        ),
    }


SENSITIVITY_POLICIES = (
    ("LENIENT", SelectionPolicy(0.80, 0.75)),
    ("MODERATE", SelectionPolicy(0.85, 0.80)),
    ("SELECTED", DEFAULT_SELECTION_POLICY),
    ("STRICT", SelectionPolicy(0.95, 0.90)),
)


def build_selection_metrics(
    inputs: Sequence[SelectionInput],
    selections: Sequence[FinalSelection],
    *,
    policy: SelectionPolicy,
) -> dict[str, Any]:
    methods = Counter(selection.final_method for selection in selections)
    sensitivity = []
    for name, candidate_policy in SENSITIVITY_POLICIES:
        if name == "SELECTED":
            candidate_policy = policy
        scenario = apply_selection_policy(inputs, policy=candidate_policy)
        sensitivity.append(
            {
                "scenario": name,
                "strong_street_similarity": (
                    candidate_policy.strong_street_similarity
                ),
                "plausible_street_similarity": (
                    candidate_policy.plausible_street_similarity
                ),
                **_status_counts(
                    [selection.final_status for selection in scenario]
                ),
            }
        )
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "total": len(selections),
        "policy": {
            "strong_street_similarity": policy.strong_street_similarity,
            "plausible_street_similarity": (
                policy.plausible_street_similarity
            ),
            "elasticsearch_score_threshold": None,
        },
        "baseline": _status_counts(
            [selection.baseline_status for selection in selections]
        ),
        "elasticsearch_proposal": _status_counts(
            [selection.es_decision.status_match for selection in selections]
        ),
        "final": _status_counts(
            [selection.final_status for selection in selections]
        ),
        "final_found_percentage": round(
            sum(
                selection.final_status == MatchStatus.FOUND
                for selection in selections
            )
            / len(selections)
            * 100,
            2,
        ),
        "final_by_method": dict(sorted(methods.items())),
        "baseline_to_elasticsearch": _transition_counts(
            selections, target="elasticsearch"
        ),
        "baseline_to_final": _transition_counts(
            selections, target="final"
        ),
        "elasticsearch_added_found": sum(
            selection.baseline_status == MatchStatus.NOT_FOUND
            and selection.final_status == MatchStatus.FOUND
            for selection in selections
        ),
        "found_with_multiple_strong_candidates_same_sector": sum(
            selection.final_status == MatchStatus.FOUND
            and selection.es_decision.strong_candidate_count > 1
            and selection.final_method
            == "ES_STRONG_EVIDENCE_SINGLE_SECTOR"
            for selection in selections
        ),
        "score_margin_analysis": _score_margin_analysis(selections),
        "sensitivity": sensitivity,
    }


def run_candidate_selection(
    inputs: Sequence[SelectionInput],
    *,
    policy: SelectionPolicy = DEFAULT_SELECTION_POLICY,
) -> CandidateSelectionRun:
    selections = apply_selection_policy(inputs, policy=policy)
    metrics = build_selection_metrics(
        inputs, selections, policy=policy
    )
    return CandidateSelectionRun(
        selections=selections,
        metrics=metrics,
    )


def write_selection_artifacts(
    baseline: pd.DataFrame,
    run: CandidateSelectionRun,
    *,
    inputs: Sequence[SelectionInput],
    audit_path: Path,
    evidence_path: Path,
    metrics_path: Path,
) -> None:
    for path in (audit_path, evidence_path, metrics_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    renamed = baseline.rename(
        columns={
            column: f"baseline_{column}"
            for column in BASELINE_RESULT_COLUMNS
            if column in baseline.columns
        }
    )
    decisions = pd.DataFrame(
        [selection.as_record() for selection in run.selections]
    )
    audit = renamed.merge(
        decisions,
        on="input_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if len(audit) != len(baseline):
        raise DatasetValidationError(
            "Candidate selection perdeu registros da base de busca."
        )
    audit.to_csv(audit_path, sep=";", encoding="utf-8", index=False)

    input_by_id = {item.address.input_id: item for item in inputs}
    evidence_rows: list[dict[str, object]] = []
    for selection in run.selections:
        item = input_by_id[selection.input_id]
        candidate_by_id = {
            candidate.candidate_id: candidate
            for candidate in item.candidates
        }
        for assessment in selection.es_decision.assessments:
            candidate = candidate_by_id[assessment.evidence.candidate_id]
            evidence_rows.append(
                {
                    "input_id": selection.input_id,
                    "baseline_status": selection.baseline_status.value,
                    "es_proposed_status": (
                        selection.es_decision.status_match.value
                    ),
                    "final_status": selection.final_status.value,
                    "final_sector": selection.final_sector,
                    "input_municipality_code": (
                        item.address.municipality_code
                    ),
                    "input_postal_code": item.address.postal_code,
                    "input_street": item.address.street,
                    "input_number": item.address.number,
                    "input_neighborhood": item.address.neighborhood,
                    "candidate_municipality_code": (
                        candidate.municipality_code
                    ),
                    "candidate_postal_code": candidate.postal_code,
                    "candidate_street": candidate.street,
                    "candidate_number": candidate.number,
                    "candidate_neighborhood": candidate.neighborhood,
                    "candidate_complement": candidate.complement,
                    **assessment.as_record(),
                }
            )
    pd.DataFrame(evidence_rows, columns=EVIDENCE_OUTPUT_COLUMNS).to_csv(
        evidence_path,
        sep=";",
        encoding="utf-8",
        index=False,
    )
    metrics_path.write_text(
        json.dumps(run.metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_matching_strategy_report(
    path: Path, run: CandidateSelectionRun
) -> None:
    metrics = run.metrics
    final = metrics["final"]
    baseline = metrics["baseline"]
    policy = metrics["policy"]
    methods = metrics["final_by_method"]
    baseline_ambiguous = methods.get("BASELINE_AMBIGUOUS_PRESERVED", 0)
    plausible_ambiguous = methods.get("ES_PLAUSIBLE_EVIDENCE_ONLY", 0)
    strong_ambiguous = methods.get("ES_STRONG_EVIDENCE_MULTIPLE_SECTORS", 0)
    no_candidates = methods.get("ES_NO_CANDIDATES", 0)
    insufficient = methods.get("ES_INSUFFICIENT_EVIDENCE", 0)
    lines = [
        "# Estrategia de matching",
        "",
        "## Separacao entre retrieval e selection",
        "",
        "O Elasticsearch recupera candidatos; o selector compara novamente os "
        "componentes normalizados e toma uma decisao explicita. `_score` e "
        "registrado, mas nao e usado como probabilidade nem como threshold.",
        "",
        "O retrieval usa ate tres consultas progressivas: igualdade estruturada, "
        "busca textual com filtro de municipio quando informado e, por ultimo, "
        "fuzzy somente no "
        "logradouro. Um limite configuravel de documentos unicos e mantido.",
        "",
        "A baseline `FOUND` e preservada porque deriva de igualdade exata e setor "
        "unico. Uma baseline `AMBIGUOUS` tambem e preservada: ela avaliou o "
        "conjunto exato completo, enquanto o retrieval mantem um top K e "
        "nao pode provar que setores conflitantes deixaram de existir.",
        "",
        "## Classes de evidencia",
        "",
        "- `STRONG` com numero informativo: municipio igual, similaridade do "
        f"logradouro >= {policy['strong_street_similarity']:.2f}, numero igual "
        "e pelo menos CEP ou bairro/localidade igual.",
        "- `STRONG` sem numero informativo: municipio igual, similaridade do "
        f"logradouro >= {policy['strong_street_similarity']:.2f}, CEP igual e "
        "bairro/localidade igual.",
        "- `PLAUSIBLE`: municipio igual, similaridade do logradouro >= "
        f"{policy['plausible_street_similarity']:.2f} e "
        "pelo menos uma ancora entre logradouro exato, CEP, numero ou "
        "bairro/localidade.",
        "- `WEAK`: qualquer combinacao abaixo desses requisitos.",
        "",
        "Nao ha soma extensa de pesos. A similaridade usa `SequenceMatcher` "
        "somente no logradouro normalizado; cada igualdade estrutural permanece "
        "visivel na auditoria.",
        "",
        "## Regras de decisao",
        "",
        "1. Sem candidatos, ou somente candidatos `WEAK`: `NOT_FOUND`.",
        "2. Sem evidencia forte, mas com candidato `PLAUSIBLE`: `AMBIGUOUS`.",
        "3. Evidencias `STRONG` em mais de um setor: `AMBIGUOUS`.",
        "4. Uma ou varias evidencias `STRONG`, todas no mesmo setor: `FOUND`.",
        "5. Multiplos documentos no mesmo setor apoiam o setor, mas nenhum ID "
        "individual e escolhido arbitrariamente.",
        "",
        "## Resultados",
        "",
        "| Fonte | FOUND | AMBIGUOUS | NOT_FOUND |",
        "|---|---:|---:|---:|",
        f"| Baseline | {baseline['FOUND']} | {baseline['AMBIGUOUS']} | {baseline['NOT_FOUND']} |",
        f"| Final | {final['FOUND']} | {final['AMBIGUOUS']} | {final['NOT_FOUND']} |",
        "",
        f"O selector adicionou {metrics['elasticsearch_added_found']} `FOUND` "
        f"aos {baseline['FOUND']} da baseline. O percentual final de encontrados e "
        f"{metrics['final_found_percentage']:.2f}%.",
        f"Em {metrics['found_with_multiple_strong_candidates_same_sector']} "
        "desses novos resultados, varios documentos fortes apoiaram o mesmo "
        "setor sem escolha arbitraria de um ID.",
        "",
        "### Casos ambiguos e nao encontrados",
        "",
        f"- `AMBIGUOUS`: {final['AMBIGUOUS']} casos — "
        f"{baseline_ambiguous} ambiguidades exatas preservadas, "
        f"{strong_ambiguous} casos com evidencias fortes em setores diferentes e "
        f"{plausible_ambiguous} casos com evidencia apenas plausivel.",
        f"- `NOT_FOUND`: {final['NOT_FOUND']} casos — {no_candidates} sem "
        f"candidatos e {insufficient} com candidatos, mas somente evidencia fraca.",
        "",
        "## Sensibilidade",
        "",
        "| Cenario | Similaridade forte | Similaridade plausivel | FOUND | AMBIGUOUS | NOT_FOUND |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for scenario in metrics["sensitivity"]:
        lines.append(
            f"| {scenario['scenario']} | "
            f"{scenario['strong_street_similarity']:.2f} | "
            f"{scenario['plausible_street_similarity']:.2f} | "
            f"{scenario['FOUND']} | {scenario['AMBIGUOUS']} | "
            f"{scenario['NOT_FOUND']} |"
        )
    margin = metrics["score_margin_analysis"]
    gap_distribution = margin["absolute_gap_distribution"]
    margin_summary = (
        "Nao houve pares de scores comparaveis para calcular a margem. "
        if gap_distribution["count"] == 0
        else "A mediana da margem absoluta entre top 1 e top 2 foi "
        f"{gap_distribution['median']} "
        f"({gap_distribution['count']} pares comparaveis). "
    )
    lines.extend(
        [
            "",
            "A tabela registra o efeito das politicas avaliadas; variar cobertura "
            "sem ground truth nao demonstra maior precisao. "
            f"{margin_summary}"
            "Nenhuma margem de `_score` participa da decisao; ela permanece "
            "diagnostica, independentemente da distribuicao observada.",
            "",
            "| Margem relativa minima | Casos que atingem |",
            "|---:|---:|",
        ]
    )
    for threshold, count in margin["relative_margin_sensitivity"].items():
        lines.append(f"| {float(threshold):.2f} | {count} |")
    lines.extend(
        [
            "",
            "Sem rotulos, a distribuicao nao permite calibrar uma probabilidade "
            "ou threshold de aceitacao. `_score` permanece apenas diagnostico.",
            "",
            "## Auditabilidade e limitacoes",
            "",
            "- `candidate_selection_audit.csv` preserva os campos originais, a "
            "decisao da baseline, a proposta Elasticsearch e a decisao final.",
            "- `candidate_selection_evidence.csv` contem uma linha por candidato "
            "com todas as comparacoes e a classe de evidencia.",
            f"- Nao existe rotulo ouro disponivel. Os {baseline['FOUND']} "
            "`FOUND` exatos servem somente como controle positivo; as metricas nao representam "
            "acuracia.",
            "- Igualdade entre bairro e localidade e tratada como evidencia "
            "adicional, nao como equivalencia geral entre os campos.",
            "- O top K pode omitir candidatos; por isso ambiguidades exatas da "
            "baseline nunca sao anuladas.",
            "",
            "## Avaliacao",
            "",
            "O estado reproduzivel do experimento de consulta relaxada fica em "
            "`docs/evaluation.md`. Ele so e executado quando a flag "
            "`--with-relaxed-query-experiment` e informada e nunca altera a "
            "saida oficial.",
            "",
            "## Reproducao",
            "",
            "```bash",
            "docker compose up -d --wait elasticsearch",
            "docker compose run --rm app python -m src.main select-candidates",
            "docker compose run --rm app python -m src.main evaluate-results "
            "--with-relaxed-query-experiment",
            "```",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def execute_candidate_selection(
    *,
    baseline_path: Path,
    retrieval_diagnostics_path: Path,
    audit_path: Path,
    evidence_path: Path,
    metrics_path: Path,
    report_path: Path,
    policy: SelectionPolicy = DEFAULT_SELECTION_POLICY,
    expected_records: int | None = None,
) -> CandidateSelectionOutput:
    """Aplica a política explícita e grava decisão, evidências e métricas."""
    baseline = read_baseline_for_selection(
        baseline_path, expected_records=expected_records
    )
    diagnostics = read_retrieval_diagnostics(retrieval_diagnostics_path)
    inputs = build_selection_inputs(baseline, diagnostics)
    run = run_candidate_selection(inputs, policy=policy)
    write_selection_artifacts(
        baseline,
        run,
        inputs=inputs,
        audit_path=audit_path,
        evidence_path=evidence_path,
        metrics_path=metrics_path,
    )
    write_matching_strategy_report(report_path, run)
    LOGGER.info(
        "Candidate selection: total=%s, found=%s, ambiguous=%s, not_found=%s",
        run.metrics["total"],
        run.metrics["final"]["FOUND"],
        run.metrics["final"]["AMBIGUOUS"],
        run.metrics["final"]["NOT_FOUND"],
    )
    return CandidateSelectionOutput(
        audit_path=audit_path,
        evidence_path=evidence_path,
        metrics_path=metrics_path,
        report_path=report_path,
        total_records=len(run.selections),
    )
