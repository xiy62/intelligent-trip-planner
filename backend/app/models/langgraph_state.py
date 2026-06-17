"""LangGraph state and supporting models for trip planning."""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

from pydantic import BaseModel, Field

from .schemas import Location, TripPlan, TripRequest, WeatherInfo


class RequestContext(BaseModel):
    """Normalized request summary used across retrieval and planning nodes."""

    city: str
    start_date: str
    end_date: str
    travel_days: int
    transportation: str
    accommodation: str
    preferences: List[str] = Field(default_factory=list)
    free_text_input: str = ""
    memory_context: str = ""
    summary: str = ""


class AttractionCandidate(BaseModel):
    """Structured attraction candidate from retrieval."""

    name: str
    address: str = ""
    location: Optional[Location] = None
    source: str = "google_maps"
    source_id: str = ""
    raw_text: str = ""


class HotelCandidate(BaseModel):
    """Structured hotel candidate from retrieval."""

    name: str
    address: str = ""
    location: Optional[Location] = None
    source: str = "google_maps"
    source_id: str = ""
    raw_text: str = ""


class RAGChunk(BaseModel):
    """Retrieved travel knowledge chunk."""

    chunk_id: str
    source: str
    title: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PlannerInputBundle(BaseModel):
    """Normalized planner payload assembled from graph state."""

    request_context: RequestContext
    attraction_candidates: List[AttractionCandidate] = Field(default_factory=list)
    hotel_candidates: List[HotelCandidate] = Field(default_factory=list)
    weather_info: List[WeatherInfo] = Field(default_factory=list)
    rag_chunks: List[RAGChunk] = Field(default_factory=list)


class EvaluationScores(BaseModel):
    """Structured evaluation scores."""

    schema_score: float = 0.0
    date_coverage_score: float = 0.0
    budget_consistency_score: float = 0.0
    grounding_score: float = 0.0
    pacing_score: float = 0.0
    route_coherence_score: float = 0.0
    preference_match_score: float = 0.0
    attribution_coverage_score: float = 0.0


class UnsupportedEntity(BaseModel):
    """Entity that could not be grounded to retrieval evidence."""

    entity_type: Literal["attraction", "hotel", "claim"]
    name: str
    reason: str = ""


class EvidenceLink(BaseModel):
    """Mapping from a generated itinerary entity to retrieval or RAG evidence."""

    entity_type: Literal["attraction", "hotel", "claim"]
    entity_name: str
    day_index: Optional[int] = None
    evidence_type: Literal["candidate_attraction", "candidate_hotel", "rag_chunk", "none"]
    evidence_id: str = ""
    source_title: str = ""
    source_url: str = ""
    confidence: float = 0.0
    match_reason: str = ""


class EvaluationReport(BaseModel):
    """Graph evaluation result used for routing and debugging."""

    passed: bool = False
    hard_failures: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    quality_warnings: List[str] = Field(default_factory=list)
    scores: EvaluationScores = Field(default_factory=EvaluationScores)
    unsupported_entities: List[UnsupportedEntity] = Field(default_factory=list)
    unsupported_claims: List[str] = Field(default_factory=list)
    evidence_links: List[EvidenceLink] = Field(default_factory=list)
    next_action: Literal[
        "finalize_response",
        "plan_itinerary",
        "retrieve_attractions",
        "retrieve_hotels",
        "fallback_response",
    ] = "plan_itinerary"


class RetryState(BaseModel):
    """Per-node execution counts used for retry routing."""

    prepare_request: int = 0
    retrieve_attractions: int = 0
    retrieve_hotels: int = 0
    retrieve_weather: int = 0
    retrieve_rag_context: int = 0
    plan_itinerary: int = 0
    evaluate_itinerary: int = 0
    finalize_response: int = 0
    fallback_response: int = 0


class RunMetrics(BaseModel):
    """Per-run metrics tracked for observability and later quantification."""

    started_at: float = 0.0
    ended_at: float = 0.0
    end_to_end_ms: float = 0.0
    node_latency_ms: Dict[str, float] = Field(default_factory=dict)
    node_total_latency_ms: Dict[str, float] = Field(default_factory=dict)
    node_attempts: Dict[str, int] = Field(default_factory=dict)
    evaluation_pass_count: int = 0
    evaluation_attempt_count: int = 0
    first_evaluation_pass: Optional[bool] = None
    final_evaluation_pass: Optional[bool] = None
    recovered_after_retry: bool = False
    fallback_count: int = 0
    schema_failure_count: int = 0
    date_coverage_failure_count: int = 0
    budget_consistency_failure_count: int = 0
    grounding_failure_count: int = 0


def merge_retry_state(left: Optional[RetryState], right: Optional[RetryState]) -> RetryState:
    """Merge retry counts from parallel graph branches without double-counting shared history."""
    if left is None:
        return right or RetryState()
    if right is None:
        return left
    merged = left.model_copy(deep=True)
    for field in RetryState.model_fields:
        setattr(merged, field, max(getattr(left, field, 0), getattr(right, field, 0)))
    return merged


def merge_run_metrics(left: Optional[RunMetrics], right: Optional[RunMetrics]) -> RunMetrics:
    """Merge run metrics emitted by parallel graph branches."""
    if left is None:
        return right or RunMetrics()
    if right is None:
        return left
    merged = left.model_copy(deep=True)
    merged.started_at = left.started_at or right.started_at
    merged.ended_at = max(left.ended_at, right.ended_at)
    merged.end_to_end_ms = max(left.end_to_end_ms, right.end_to_end_ms)
    merged.node_latency_ms.update(right.node_latency_ms)
    for key, value in right.node_total_latency_ms.items():
        merged.node_total_latency_ms[key] = max(merged.node_total_latency_ms.get(key, 0.0), value)
    for key, value in right.node_attempts.items():
        merged.node_attempts[key] = max(merged.node_attempts.get(key, 0), value)

    merged.evaluation_pass_count = max(left.evaluation_pass_count, right.evaluation_pass_count)
    merged.evaluation_attempt_count = max(left.evaluation_attempt_count, right.evaluation_attempt_count)
    merged.first_evaluation_pass = (
        left.first_evaluation_pass
        if left.first_evaluation_pass is not None
        else right.first_evaluation_pass
    )
    merged.final_evaluation_pass = (
        right.final_evaluation_pass
        if right.final_evaluation_pass is not None
        else left.final_evaluation_pass
    )
    merged.recovered_after_retry = left.recovered_after_retry or right.recovered_after_retry
    merged.fallback_count = max(left.fallback_count, right.fallback_count)
    merged.schema_failure_count = max(left.schema_failure_count, right.schema_failure_count)
    merged.date_coverage_failure_count = max(left.date_coverage_failure_count, right.date_coverage_failure_count)
    merged.budget_consistency_failure_count = max(
        left.budget_consistency_failure_count,
        right.budget_consistency_failure_count,
    )
    merged.grounding_failure_count = max(left.grounding_failure_count, right.grounding_failure_count)
    return merged


def merge_decision_trace(left: Optional[List[str]], right: Optional[List[str]]) -> List[str]:
    """Append trace events from parallel branches in a stable, compact order."""
    return list(left or []) + list(right or [])


class TripGraphState(TypedDict, total=False):
    """Typed LangGraph state."""

    request: TripRequest
    travel_dates: List[str]
    request_context: RequestContext
    candidate_attractions: List[AttractionCandidate]
    candidate_hotels: List[HotelCandidate]
    weather_info: List[WeatherInfo]
    rag_chunks: List[RAGChunk]
    planner_inputs: PlannerInputBundle
    draft_plan_raw: str
    draft_plan: Optional[TripPlan]
    evaluation_report: EvaluationReport
    evaluation_history: List[EvaluationReport]
    retry_counts: Annotated[RetryState, merge_retry_state]
    decision_trace: Annotated[List[str], merge_decision_trace]
    metrics: Annotated[RunMetrics, merge_run_metrics]
    final_plan: Optional[TripPlan]
    conversation_id: str
    memory_applied: bool
    memory_summary: str
    memory_profile: Dict[str, Any]
