"""Evidencias e decisoes deterministicas sobre candidatos de endereco."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import Enum


class MatchStatus(str, Enum):
    FOUND = "FOUND"
    NOT_FOUND = "NOT_FOUND"
    AMBIGUOUS = "AMBIGUOUS"


@dataclass(frozen=True)
class SelectionAddress:
    input_id: str
    municipality_code: str | None
    postal_code: str | None
    street: str | None
    number: str | None
    neighborhood: str | None
    complement: str | None


@dataclass(frozen=True)
class SelectionCandidate:
    candidate_id: str
    sector: str
    municipality_code: str | None
    postal_code: str | None
    street: str | None
    number: str | None
    neighborhood: str | None
    complement: str | None
    es_score: float | None
    retrieval_strategy: str
    retrieval_rank: int


@dataclass(frozen=True)
class CandidateEvidence:
    candidate_id: str
    sector: str
    retrieval_strategy: str
    retrieval_rank: int
    es_score: float | None
    municipality_match: bool
    postal_code_available: bool
    postal_code_match: bool
    street_exact: bool
    street_similarity: float
    street_input_token_coverage: float
    street_candidate_extra_tokens: int
    informative_number_available: bool
    number_match: bool
    no_number_marker_match: bool
    neighborhood_available: bool
    neighborhood_match: bool
    complement_available: bool
    complement_match: bool

    def as_record(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "sector": self.sector,
            "retrieval_strategy": self.retrieval_strategy,
            "retrieval_rank": self.retrieval_rank,
            "es_score": self.es_score,
            "municipality_match": self.municipality_match,
            "postal_code_available": self.postal_code_available,
            "postal_code_match": self.postal_code_match,
            "street_exact": self.street_exact,
            "street_similarity": self.street_similarity,
            "street_input_token_coverage": self.street_input_token_coverage,
            "street_candidate_extra_tokens": (
                self.street_candidate_extra_tokens
            ),
            "informative_number_available": (
                self.informative_number_available
            ),
            "number_match": self.number_match,
            "no_number_marker_match": self.no_number_marker_match,
            "neighborhood_available": self.neighborhood_available,
            "neighborhood_match": self.neighborhood_match,
            "complement_available": self.complement_available,
            "complement_match": self.complement_match,
        }


def _same_present(left: str | None, right: str | None) -> bool:
    return bool(left and right and left == right)


def _street_metrics(
    input_street: str | None, candidate_street: str | None
) -> tuple[bool, float, float, int]:
    if not input_street or not candidate_street:
        return False, 0.0, 0.0, 0
    input_tokens = set(input_street.split())
    candidate_tokens = set(candidate_street.split())
    common_tokens = input_tokens & candidate_tokens
    coverage = (
        len(common_tokens) / len(input_tokens) if input_tokens else 0.0
    )
    return (
        input_street == candidate_street,
        SequenceMatcher(None, input_street, candidate_street).ratio(),
        coverage,
        len(candidate_tokens - input_tokens),
    )


class EvidenceStrength(str, Enum):
    STRONG = "STRONG"
    PLAUSIBLE = "PLAUSIBLE"
    WEAK = "WEAK"


@dataclass(frozen=True)
class SelectionPolicy:
    strong_street_similarity: float = 0.90
    plausible_street_similarity: float = 0.85

    def __post_init__(self) -> None:
        if not 0 <= self.plausible_street_similarity <= 1:
            raise ValueError(
                "plausible_street_similarity deve estar entre zero e um."
            )
        if not 0 <= self.strong_street_similarity <= 1:
            raise ValueError(
                "strong_street_similarity deve estar entre zero e um."
            )
        if self.plausible_street_similarity > self.strong_street_similarity:
            raise ValueError(
                "O threshold plausivel nao pode superar o threshold forte."
            )


DEFAULT_SELECTION_POLICY = SelectionPolicy()


@dataclass(frozen=True)
class EvidenceAssessment:
    evidence: CandidateEvidence
    strength: EvidenceStrength
    reasons: tuple[str, ...]

    def as_record(self) -> dict[str, object]:
        return {
            **self.evidence.as_record(),
            "evidence_strength": self.strength.value,
            "evidence_reasons": "|".join(self.reasons),
        }


@dataclass(frozen=True)
class SelectionDecision:
    input_id: str
    setor_censitario: str | None
    status_match: MatchStatus
    metodo_match: str
    top_score: float | None
    second_score: float | None
    score_gap: float | None
    score_gap_ratio: float | None
    scores_comparable: bool
    candidate_count: int
    strong_candidate_count: int
    plausible_candidate_count: int
    candidate_sector_count: int
    selected_candidate_id: str | None
    candidate_sectors: tuple[str, ...]
    supporting_candidate_ids: tuple[str, ...]
    top_evidence_strength: EvidenceStrength | None
    evidence: tuple[str, ...]
    assessments: tuple[EvidenceAssessment, ...]

    def as_record(self) -> dict[str, object]:
        return {
            "input_id": self.input_id,
            "setor_censitario": self.setor_censitario,
            "status_match": self.status_match.value,
            "metodo_match": self.metodo_match,
            "top_score": self.top_score,
            "second_score": self.second_score,
            "score_gap": self.score_gap,
            "score_gap_ratio": self.score_gap_ratio,
            "scores_comparable": self.scores_comparable,
            "candidate_count": self.candidate_count,
            "strong_candidate_count": self.strong_candidate_count,
            "plausible_candidate_count": self.plausible_candidate_count,
            "candidate_sector_count": self.candidate_sector_count,
            "selected_candidate_id": self.selected_candidate_id,
            "candidate_sectors": "|".join(self.candidate_sectors),
            "supporting_candidate_ids": "|".join(
                self.supporting_candidate_ids
            ),
            "top_evidence_strength": (
                self.top_evidence_strength.value
                if self.top_evidence_strength
                else None
            ),
            "evidence": "|".join(self.evidence),
        }
def evaluate_candidate(
    address: SelectionAddress,
    candidate: SelectionCandidate,
) -> CandidateEvidence:
    """Compara componentes normalizados sem usar o _score como verdade."""
    street_exact, street_similarity, token_coverage, extra_tokens = (
        _street_metrics(address.street, candidate.street)
    )
    informative_number = bool(address.number and address.number != "sn")
    return CandidateEvidence(
        candidate_id=candidate.candidate_id,
        sector=candidate.sector,
        retrieval_strategy=candidate.retrieval_strategy,
        retrieval_rank=candidate.retrieval_rank,
        es_score=candidate.es_score,
        municipality_match=_same_present(
            address.municipality_code, candidate.municipality_code
        ),
        postal_code_available=bool(address.postal_code),
        postal_code_match=_same_present(
            address.postal_code, candidate.postal_code
        ),
        street_exact=street_exact,
        street_similarity=street_similarity,
        street_input_token_coverage=token_coverage,
        street_candidate_extra_tokens=extra_tokens,
        informative_number_available=informative_number,
        number_match=(
            informative_number
            and _same_present(address.number, candidate.number)
        ),
        no_number_marker_match=(
            address.number == "sn" and candidate.number == "sn"
        ),
        neighborhood_available=bool(address.neighborhood),
        neighborhood_match=_same_present(
            address.neighborhood, candidate.neighborhood
        ),
        complement_available=bool(address.complement),
        complement_match=_same_present(
            address.complement, candidate.complement
        ),
    )


def classify_evidence(
    evidence: CandidateEvidence,
    *,
    policy: SelectionPolicy = DEFAULT_SELECTION_POLICY,
) -> EvidenceAssessment:
    reasons: list[str] = []
    if evidence.municipality_match:
        reasons.append("MUNICIPALITY_EXACT")
    else:
        return EvidenceAssessment(
            evidence=evidence,
            strength=EvidenceStrength.WEAK,
            reasons=("MUNICIPALITY_MISMATCH_OR_MISSING",),
        )

    if evidence.street_exact:
        reasons.append("STREET_EXACT")
    else:
        reasons.append(
            f"STREET_SIMILARITY={evidence.street_similarity:.4f}"
        )
    if evidence.postal_code_match:
        reasons.append("POSTAL_CODE_EXACT")
    if evidence.number_match:
        reasons.append("NUMBER_EXACT")
    if evidence.no_number_marker_match:
        reasons.append("NO_NUMBER_MARKER_BOTH")
    if evidence.neighborhood_match:
        reasons.append("NEIGHBORHOOD_LOCALITY_EXACT")
    if evidence.complement_match:
        reasons.append("COMPLEMENT_EXACT")

    strong_street = (
        evidence.street_similarity >= policy.strong_street_similarity
    )
    strong_numbered = (
        evidence.informative_number_available
        and evidence.number_match
        and (evidence.postal_code_match or evidence.neighborhood_match)
    )
    strong_without_number = (
        not evidence.informative_number_available
        and evidence.postal_code_match
        and evidence.neighborhood_match
    )
    if strong_street and (strong_numbered or strong_without_number):
        reasons.append("STRONG_STRUCTURED_EVIDENCE")
        return EvidenceAssessment(
            evidence=evidence,
            strength=EvidenceStrength.STRONG,
            reasons=tuple(reasons),
        )

    plausible_anchor = (
        evidence.street_exact
        or evidence.postal_code_match
        or evidence.number_match
        or evidence.neighborhood_match
    )
    if (
        evidence.street_similarity >= policy.plausible_street_similarity
        and plausible_anchor
    ):
        reasons.append("PLAUSIBLE_STRUCTURED_EVIDENCE")
        return EvidenceAssessment(
            evidence=evidence,
            strength=EvidenceStrength.PLAUSIBLE,
            reasons=tuple(reasons),
        )

    reasons.append("INSUFFICIENT_STRUCTURED_EVIDENCE")
    return EvidenceAssessment(
        evidence=evidence,
        strength=EvidenceStrength.WEAK,
        reasons=tuple(reasons),
    )


_STRENGTH_PRIORITY = {
    EvidenceStrength.STRONG: 2,
    EvidenceStrength.PLAUSIBLE: 1,
    EvidenceStrength.WEAK: 0,
}

_STRATEGY_PRIORITY = {
    "EXACT_STRUCTURED": 1,
    "TEXT_WITH_GEOGRAPHY": 2,
    "FUZZY_STREET_WITH_MUNICIPALITY": 3,
}


def _assessment_sort_key(
    assessment: EvidenceAssessment,
) -> tuple[object, ...]:
    evidence = assessment.evidence
    anchor_count = sum(
        (
            evidence.postal_code_match,
            evidence.number_match,
            evidence.neighborhood_match,
            evidence.complement_match,
        )
    )
    score_order = (
        -evidence.es_score
        if evidence.es_score is not None
        else float("inf")
    )
    return (
        -_STRENGTH_PRIORITY[assessment.strength],
        -anchor_count,
        -int(evidence.street_exact),
        -evidence.street_similarity,
        _STRATEGY_PRIORITY.get(evidence.retrieval_strategy, 99),
        score_order,
        evidence.candidate_id,
    )


def _score_diagnostics(
    assessments: tuple[EvidenceAssessment, ...],
) -> tuple[float | None, float | None, float | None, float | None, bool]:
    retrieval_order = tuple(
        sorted(
            assessments,
            key=lambda assessment: assessment.evidence.retrieval_rank,
        )
    )
    top_score = (
        retrieval_order[0].evidence.es_score if retrieval_order else None
    )
    second_score = (
        retrieval_order[1].evidence.es_score
        if len(retrieval_order) >= 2
        else None
    )
    comparable = bool(
        len(retrieval_order) >= 2
        and retrieval_order[0].evidence.retrieval_strategy
        == retrieval_order[1].evidence.retrieval_strategy
        and top_score is not None
        and second_score is not None
    )
    if not comparable:
        return top_score, second_score, None, None, False
    gap = top_score - second_score
    ratio = gap / abs(top_score) if top_score else None
    return top_score, second_score, gap, ratio, True


def select_retrieved_candidates(
    address: SelectionAddress,
    candidates: tuple[SelectionCandidate, ...],
    *,
    policy: SelectionPolicy = DEFAULT_SELECTION_POLICY,
) -> SelectionDecision:
    """Decide por classes de evidencia; _score e somente diagnostico."""
    assessments = tuple(
        sorted(
            (
                classify_evidence(
                    evaluate_candidate(address, candidate), policy=policy
                )
                for candidate in candidates
            ),
            key=_assessment_sort_key,
        )
    )
    all_sectors = tuple(
        sorted({assessment.evidence.sector for assessment in assessments})
    )
    strong = tuple(
        assessment
        for assessment in assessments
        if assessment.strength == EvidenceStrength.STRONG
    )
    plausible = tuple(
        assessment
        for assessment in assessments
        if assessment.strength == EvidenceStrength.PLAUSIBLE
    )
    top_score, second_score, gap, ratio, comparable = _score_diagnostics(
        assessments
    )
    common = {
        "input_id": address.input_id,
        "top_score": top_score,
        "second_score": second_score,
        "score_gap": gap,
        "score_gap_ratio": ratio,
        "scores_comparable": comparable,
        "candidate_count": len(candidates),
        "strong_candidate_count": len(strong),
        "plausible_candidate_count": len(plausible),
        "candidate_sector_count": len(all_sectors),
        "candidate_sectors": all_sectors,
        "top_evidence_strength": (
            assessments[0].strength if assessments else None
        ),
        "evidence": assessments[0].reasons if assessments else (),
        "assessments": assessments,
    }
    if not assessments:
        return SelectionDecision(
            setor_censitario=None,
            status_match=MatchStatus.NOT_FOUND,
            metodo_match="ES_NO_CANDIDATES",
            selected_candidate_id=None,
            supporting_candidate_ids=(),
            **common,
        )

    if strong:
        strong_sectors = {
            assessment.evidence.sector for assessment in strong
        }
        if len(strong_sectors) == 1:
            supporting_ids = tuple(
                assessment.evidence.candidate_id for assessment in strong
            )
            return SelectionDecision(
                setor_censitario=next(iter(strong_sectors)),
                status_match=MatchStatus.FOUND,
                metodo_match="ES_STRONG_EVIDENCE_SINGLE_SECTOR",
                selected_candidate_id=(
                    supporting_ids[0] if len(supporting_ids) == 1 else None
                ),
                supporting_candidate_ids=supporting_ids,
                **common,
            )
        return SelectionDecision(
            setor_censitario=None,
            status_match=MatchStatus.AMBIGUOUS,
            metodo_match="ES_STRONG_EVIDENCE_MULTIPLE_SECTORS",
            selected_candidate_id=None,
            supporting_candidate_ids=tuple(
                assessment.evidence.candidate_id for assessment in strong
            ),
            **common,
        )

    if plausible:
        return SelectionDecision(
            setor_censitario=None,
            status_match=MatchStatus.AMBIGUOUS,
            metodo_match="ES_PLAUSIBLE_EVIDENCE_ONLY",
            selected_candidate_id=None,
            supporting_candidate_ids=tuple(
                assessment.evidence.candidate_id for assessment in plausible
            ),
            **common,
        )

    return SelectionDecision(
        setor_censitario=None,
        status_match=MatchStatus.NOT_FOUND,
        metodo_match="ES_INSUFFICIENT_EVIDENCE",
        selected_candidate_id=None,
        supporting_candidate_ids=(),
        **common,
    )


@dataclass(frozen=True)
class CandidateSummary:
    input_id: str
    strategy_name: str
    candidate_count: int
    candidate_row_count: int
    candidate_sectors: tuple[str, ...]
    minimum_candidate_id: str | None


@dataclass(frozen=True)
class MatchDecision:
    input_id: str
    setor_censitario: str | None
    status_match: MatchStatus
    metodo_match: str
    quantidade_candidatos: int
    quantidade_linhas_candidatas: int
    quantidade_setores_candidatos: int
    score_match: float | None
    id_candidato_cnefe: str | None
    setores_candidatos: tuple[str, ...]

    def as_record(self) -> dict[str, object]:
        return {
            "input_id": self.input_id,
            "setor_censitario": self.setor_censitario,
            "status_match": self.status_match.value,
            "metodo_match": self.metodo_match,
            "quantidade_candidatos": self.quantidade_candidatos,
            "quantidade_linhas_candidatas": self.quantidade_linhas_candidatas,
            "quantidade_setores_candidatos": self.quantidade_setores_candidatos,
            "score_match": self.score_match,
            "id_candidato_cnefe": self.id_candidato_cnefe,
            "setores_candidatos": "|".join(self.setores_candidatos),
        }


def select_candidate(summary: CandidateSummary) -> MatchDecision:
    """Aceita setor unico, explicita setores multiplos e nunca escolhe o primeiro."""
    sector_count = len(summary.candidate_sectors)
    if summary.candidate_count == 0 or sector_count == 0:
        return not_found_decision(summary.input_id)
    if sector_count > 1:
        return MatchDecision(
            input_id=summary.input_id,
            setor_censitario=None,
            status_match=MatchStatus.AMBIGUOUS,
            metodo_match=summary.strategy_name,
            quantidade_candidatos=summary.candidate_count,
            quantidade_linhas_candidatas=summary.candidate_row_count,
            quantidade_setores_candidatos=sector_count,
            score_match=None,
            id_candidato_cnefe=None,
            setores_candidatos=summary.candidate_sectors,
        )

    return MatchDecision(
        input_id=summary.input_id,
        setor_censitario=summary.candidate_sectors[0],
        status_match=MatchStatus.FOUND,
        metodo_match=summary.strategy_name,
        quantidade_candidatos=summary.candidate_count,
        quantidade_linhas_candidatas=summary.candidate_row_count,
        quantidade_setores_candidatos=1,
        score_match=None,
        id_candidato_cnefe=(
            summary.minimum_candidate_id
            if summary.candidate_count == 1
            else None
        ),
        setores_candidatos=summary.candidate_sectors,
    )


def not_found_decision(input_id: str) -> MatchDecision:
    return MatchDecision(
        input_id=input_id,
        setor_censitario=None,
        status_match=MatchStatus.NOT_FOUND,
        metodo_match="NO_EXACT_MATCH",
        quantidade_candidatos=0,
        quantidade_linhas_candidatas=0,
        quantidade_setores_candidatos=0,
        score_match=None,
        id_candidato_cnefe=None,
        setores_candidatos=(),
    )
